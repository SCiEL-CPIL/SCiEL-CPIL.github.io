import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We aim for 140s limit.
    deadline = start_time + 140.0
    
    N = 16
    points = list(range(N))
    
    # Precompute all triples as bitmasks and lists
    # We need both: bitmask for fast independence check, list for pair checking
    all_triples_data = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                mask = (1 << a) | (1 << b) | (1 << c)
                all_triples_data.append(((a, b, c), mask))
                
    # Helper to check linearity using pair sets
    # A linear triple system on triples implies no pair is shared by two triples.
    # So we track used pairs.
    
    def calc_alpha_fast(triples_masks):
        """
        Calculate independence number exactly using bitmask DP.
        A subset S is independent if no triple mask is a subset of S.
        """
        if not triples_masks:
            return N
            
        # Mark all subsets that contain at least one triple as "bad" (not independent)
        # bad[S] = True if S contains a triple
        bad = [False] * (1 << N)
        
        # Optimization: Only mark supersets of each triple mask
        for m in triples_masks:
            # Supersets of m are m | s where s is a subset of ~m
            # We iterate all s subset of full_mask ^ m
            free_mask = ((1 << N) - 1) ^ m
            s = free_mask
            while True:
                bad[m | s] = True
                if s == 0:
                    break
                s = (s - 1) & free_mask
            
        # Find max size of S such that bad[S] is False
        max_alpha = 0
        for S in range(1 << N):
            if not bad[S]:
                cnt = S.bit_count()
                if cnt > max_alpha:
                    max_alpha = cnt
        return max_alpha

    def find_maximal_independent_set(triples_masks, rng):
        """
        Greedily find a large independent set.
        """
        # Randomize point order
        pts = list(range(N))
        rng.shuffle(pts)
        
        chosen = []
        chosen_mask = 0
        chosen_set = set()
        
        for p in pts:
            p_mask = 1 << p
            # Check if adding p creates a triple
            # A triple is formed if there exists a triple mask t such that t is subset of (chosen_mask | p_mask)
            # Since chosen is independent, no triple is in chosen_mask.
            # So we just need to check if (chosen_mask | p_mask) contains any triple.
            new_mask = chosen_mask | p_mask
            
            is_ok = True
            for t_mask in triples_masks:
                if (new_mask & t_mask) == t_mask:
                    is_ok = False
                    break
            
            if is_ok:
                chosen.append(p)
                chosen_mask = new_mask
                chosen_set.add(p)
                
        return chosen

    def try_add_breaking_triple(triples, used_pairs, iset, rng):
        """
        Try to find a triple that:
        1. Is linear with existing triples (no shared pairs).
        2. Intersects the independent set 'iset' (to reduce its size or break it).
        Ideally, it should intersect iset in at least 1 point.
        To actually reduce alpha, it's best if it hits the specific iset we found.
        """
        iset_set = set(iset)
        
        # Candidates: triples that have at least one point in iset
        # We generate these on the fly to avoid storing huge lists, or pre-compute candidates per point?
        # For N=16, iterating all 560 triples is fast enough.
        
        # Shuffle all triples to randomize search
        candidates = list(all_triples_data)
        rng.shuffle(candidates)
        
        for t, t_mask in candidates:
            # Check if it intersects iset
            if not (set(t) & iset_set):
                continue
            
            # Check linearity: no pair in t is in used_pairs
            a, b, c = t
            if (a, b) in used_pairs: continue
            if (a, c) in used_pairs: continue
            if (b, c) in used_pairs: continue
            
            # It's a valid addition
            return t, t_mask
        
        return None

    def add_triple(triples, used_pairs, t, t_mask):
        triples.append((t, t_mask))
        a, b, c = t
        used_pairs.add((a, b))
        used_pairs.add((b, a))
        used_pairs.add((a, c))
        used_pairs.add((c, a))
        used_pairs.add((b, c))
        used_pairs.add((c, b))

    def remove_triple(triples, used_pairs, idx):
        t, t_mask = triples[idx]
        a, b, c = t
        used_pairs.discard((a, b))
        used_pairs.discard((b, a))
        used_pairs.discard((a, c))
        used_pairs.discard((c, a))
        used_pairs.discard((b, c))
        used_pairs.discard((c, b))
        triples.pop(idx)

    best_alpha = N
    best_system = []
    rng = random.Random(42)

    # Phase 1: Multiple Random Greedy Packings
    attempts = 0
    while time.time() < deadline - 10.0: # Reserve 10s for final polish
        attempts += 1
        
        current_triples = []
        used_pairs = set()
        
        # Greedy packing
        shuffled_data = list(all_triples_data)
        rng.shuffle(shuffled_data)
        
        for t, t_mask in shuffled_data:
            a, b, c = t
            if (a, b) in used_pairs: continue
            if (a, c) in used_pairs: continue
            if (b, c) in used_pairs: continue
            add_triple(current_triples, used_pairs, t, t_mask)
        
        # Local Search
        current_alpha = calc_alpha_fast([m for _, m in current_triples])
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_system = [list(t) for t, _ in current_triples]
            
        # Iterate local search
        for step in range(500):
            if time.time() > deadline - 5.0:
                break
                
            # Find a maximal independent set
            iset = find_maximal_independent_set([m for _, m in current_triples], rng)
            
            if len(iset) <= current_alpha:
                # If the greedy independent set is not larger than current alpha, 
                # we might have reached a local optimum for this greedy heuristic.
                # But let's try one more kick if we are stuck.
                if len(iset) <= best_alpha:
                    break
            
            # Try to add a triple that hits iset
            added = try_add_breaking_triple(current_triples, used_pairs, iset, rng)
            
            if added:
                t, t_mask = added
                add_triple(current_triples, used_pairs, t, t_mask)
                # Recalculate alpha (expensive, but N=16 is small)
                new_alpha = calc_alpha_fast([m for _, m in current_triples])
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = [list(t) for t, _ in current_triples]
                # If alpha didn't improve, we might have added a triple that didn't break the *maximum* independent set
                # but broke *this* one. We continue.
            else:
                # Kick: Remove a random triple, try to add a breaking one, restore if failed
                if not current_triples:
                    break
                idx = rng.randrange(len(current_triples))
                removed_t, removed_mask = current_triples[idx]
                remove_triple(current_triples, used_pairs, idx)
                
                added = try_add_breaking_triple(current_triples, used_pairs, iset, rng)
                
                if added:
                    t, t_mask = added
                    add_triple(current_triples, used_pairs, t, t_mask)
                    new_alpha = calc_alpha_fast([m for _, m in current_triples])
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_system = [list(t) for t, _ in current_triples]
                    else:
                        # If alpha didn't improve, revert the kick?
                        # Actually, if we swapped one triple for another, the structure changed.
                        # If alpha is same, it's a neutral move. We can keep it or revert.
                        # Let's revert to avoid drift if no improvement, to stay in a good basin.
                        # Remove the new one, restore the old one.
                        # Find index of new one (it's the last one added)
                        # Actually, simpler: just revert if no strict improvement in alpha.
                        # Find the new triple to remove
                        # The new triple is at the end
                        new_idx = len(current_triples) - 1
                        remove_triple(current_triples, used_pairs, new_idx)
                        # Restore old
                        add_triple(current_triples, used_pairs, removed_t, removed_mask)
                        # Note: add_triple appends, so order changes. 
                        # This is fine as long as we track state correctly.
                        # However, `remove_triple` by index is fragile if we don't track indices carefully.
                        # Let's simplify: if no improvement, just restore the removed one and stop kicking for this step?
                        # Or just don't revert, let it drift? 
                        # Given the small size, let's try to be conservative: revert if no alpha improvement.
                        # To revert properly:
                        # 1. Remove the new triple (last added)
                        # 2. Add the old triple back.
                        # Since we removed new and added old, the set of triples is back to original.
                        # Wait, I removed new (which is last) and added old (which is appended).
                        # So current_triples ends with old_t.
                        # The state is effectively restored (same set of triples).
                        pass
                else:
                    # Could not add anything, restore removed triple
                    add_triple(current_triples, used_pairs, removed_t, removed_mask)
                    # If we can't improve, break local search?
                    # Maybe try a few more kicks?
                    # For now, break to save time for new restart.
                    break

        if best_alpha <= 7: # Theoretical lower bound is around 6-7 for 16 pts?
             # Actually, f(16) is unknown but likely small. 
             # If we get 8 or 9, that's good.
             pass

    return best_system

# Run a quick self-test to ensure it doesn't crash and returns a valid list
if __name__ == "__main__":
    res = construct()
    print(f"Result: {len(res)} triples, sample: {res[:5]}")
