import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_points = list(range(16))
    all_triples = []
    triple_to_idx = {}
    for c in itertools.combinations(all_points, 3):
        mask = 0
        for p in c:
            mask |= (1 << p)
        idx = len(all_triples)
        all_triples.append((c, mask))
        triple_to_idx[c] = idx
        
    num_triples = len(all_triples)
    
    # Precompute conflict matrix
    conflict = [[False]*num_triples for _ in range(num_triples)]
    for i in range(num_triples):
        mi = all_triples[i][1]
        for j in range(i+1, num_triples):
            mj = all_triples[j][1]
            inter = mi & mj
            if inter & (inter - 1): # popcount > 1
                conflict[i][j] = True
                conflict[j][i] = True

    def can_add(active_set, idx):
        for j in active_set:
            if conflict[idx][j]:
                return False
        return True
        
    def get_conflicts(active_set, idx):
        return [j for j in active_set if conflict[idx][j]]

    # Check if there is an independent set of size 6
    # Returns True and the set if found
    def find_independent_set_size_6(active_indices):
        active_masks = [all_triples[i][1] for i in active_indices]
        # Shuffle points to randomize which independent set we find first?
        # Not necessary, just find one.
        for s in itertools.combinations(all_points, 6):
            smask = 0
            for p in s:
                smask |= (1 << p)
            is_independent = True
            for tm in active_masks:
                if (tm & smask) == tm:
                    is_independent = False
                    break
            if is_independent:
                return True, s
        return False, None

    def generate_random_packing():
        active = set()
        indices = list(range(num_triples))
        random.shuffle(indices)
        for idx in indices:
            if can_add(active, idx):
                active.add(idx)
        return active

    best_active = None
    best_alpha = 16
    
    # We only care about alpha <= 5. 
    # If we find a system with no independent set of size 6, alpha is 5 (since alpha>=5).
    
    while time.time() < deadline:
        if best_active is None:
            active = generate_random_packing()
        else:
            # Check if current best is already optimal
            has_ind, ind_set = find_independent_set_size_6(best_active)
            if not has_ind:
                break # Found optimal alpha=5
            
            # Try to break an independent set in best_active
            # Pick a random independent set of size 6
            # We have 'ind_set' from the check above.
            # To diversify, maybe find a different one?
            # For now, use the one found.
            
            # Pick a random triple from ind_set
            sub_triples = list(itertools.combinations(ind_set, 3))
            random.shuffle(sub_triples)
            target_triple = sub_triples[0]
            target_idx = triple_to_idx[target_triple]
            
            # Remove conflicts and add target
            conflicts = get_conflicts(best_active, target_idx)
            for c_idx in conflicts:
                best_active.remove(c_idx)
            best_active.add(target_idx)
            
            # Refill: try to add random valid triples
            # Do a few rounds of random addition
            indices = list(range(num_triples))
            random.shuffle(indices)
            for idx in indices:
                if idx not in best_active and can_add(best_active, idx):
                    best_active.add(idx)
                    # Stop after a few additions to keep time low?
                    # Or just go through a subset.
                    # Let's just break after 10 additions
                    # Actually, just check a few random ones
                    pass 

            # Update best? 
            # Since we modified best_active directly, it is the current state.
            # We don't need to compare if we assume the move improves alpha 
            # (it breaks one independent set, but might create others).
            # However, it's possible we made it worse (increased alpha).
            # To be safe, we should keep a backup and only accept if alpha didn't increase.
            # But computing alpha exactly is expensive.
            # We only know if alpha <= 5.
            # If we break an independent set of size 6, alpha might drop to 5.
            # If we create a new independent set of size 6, alpha is still >= 6.
            # If we remove too many triples, alpha might increase to 7?
            # Removing triples makes it EASIER to have independent sets.
            # So alpha can only increase or stay same when removing triples.
            # Adding triples makes it HARDER to have independent sets.
            # So alpha can only decrease or stay same when adding triples.
            
            # My move: Remove conflicts (alpha might increase), Add target (alpha might decrease).
            # Net effect is uncertain.
            # So I should keep the best known state.
            
            # Let's restructure: 
            # 1. Keep 'current' and 'best'.
            # 2. Try move on 'current'.
            # 3. If 'current' has no ind set of 6, update 'best' and break.
            # 4. Else, maybe accept 'current' if it looks better? 
            #    Hard to judge without full alpha.
            #    Let's just use Simulated Annealing style: 
            #    Accept with prob exp(-delta/T)? 
            #    But we don't have delta.
            #    
            #    Simple strategy: 
            #    If the move breaks an independent set, it's a "good" direction.
            #    Keep it.
            #    If it fails to find alpha=5 in time, return best.

        # End while

    # Final check
    if best_active is None:
        best_active = generate_random_packing()
        
    triples = [list(all_triples[i][0]) for i in best_active]
    return triples

if __name__ == "__main__":
    res = construct()
    print(f"Constructed {len(res)} triples")
    # Verify linearity
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            inter = set(res[i]) & set(res[j])
            if len(inter) > 1:
                raise ValueError(f"Triples {i} and {j} share more than 1 point: {inter}")
    print("Linearity valid")
