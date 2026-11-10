import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute masks for each triple
    triple_masks = [0] * n_triples
    for i, t in enumerate(all_triples):
        mask = 0
        for v in t:
            mask |= (1 << v)
        triple_masks[i] = mask
        
    # Precompute pairs for each triple to check linearity
    triple_pairs = []
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        triple_pairs.append(pairs)
        
    def calc_alpha(triple_indices):
        """
        Calculate independence number of the hypergraph defined by triple_indices.
        """
        if not triple_indices:
            return 16
            
        n = 16
        active_masks = [triple_masks[i] for i in triple_indices]
        
        # Sort vertices by degree (number of active triples containing them)
        deg = [0] * n
        for mask in active_masks:
            for v in range(n):
                if mask & (1 << v):
                    deg[v] += 1
                    
        order = sorted(range(n), key=lambda v: -deg[v])
        
        best_alpha = [0]
        
        def search(idx, current_mask, current_size):
            if time.time() > deadline - 5:
                return
            
            if current_size > best_alpha[0]:
                best_alpha[0] = current_size
                
            if idx == n:
                return
                
            v = order[idx]
            
            # Option 1: Exclude v
            search(idx + 1, current_mask, current_size)
            
            # Option 2: Include v
            new_mask = current_mask | (1 << v)
            
            valid = True
            for am in active_masks:
                if am & new_mask == am:
                    valid = False
                    break
                    
            if valid:
                search(idx + 1, new_mask, current_size + 1)
                
        search(0, 0, 0)
        return best_alpha[0]

    def fill_system(current_triples, used_pairs):
        """Try to add more triples to make the system maximal."""
        changed = True
        while changed:
            changed = False
            # Shuffle to break symmetry
            indices = list(range(n_triples))
            random.shuffle(indices)
            for t_idx in indices:
                if t_idx in current_triples:
                    continue
                pairs = triple_pairs[t_idx]
                valid = True
                for p in pairs:
                    if p in used_pairs:
                        valid = False
                        break
                if valid:
                    for p in pairs:
                        used_pairs.add(p)
                    current_triples.append(t_idx)
                    changed = True
                    
    best_alpha = 17
    best_solution = None
    
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        random.seed(1000 + restart)
        
        used_pairs = set()
        current_triples = []
        current_triple_set = set()
        
        # Greedy initialization
        indices = list(range(n_triples))
        random.shuffle(indices)
        
        for t_idx in indices:
            pairs = triple_pairs[t_idx]
            valid = True
            for p in pairs:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                for p in pairs:
                    used_pairs.add(p)
                current_triples.append(t_idx)
                current_triple_set.add(t_idx)
                
        # Fill to maximal
        fill_system(current_triples, used_pairs)
        
        # Calculate initial alpha
        alpha = calc_alpha(current_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_solution = list(current_triples)
            
        if best_alpha <= 4:
            break
            
        # Local Search
        steps = 0
        max_steps = 100
        while steps < max_steps:
            if time.time() > deadline - 10:
                break
            steps += 1
            
            if not current_triples:
                break
                
            rem_idx_pos = random.randint(0, len(current_triples) - 1)
            rem_idx = current_triples[rem_idx_pos]
            
            # Remove
            current_triples.pop(rem_idx_pos)
            current_triple_set.discard(rem_idx)
            for p in triple_pairs[rem_idx]:
                used_pairs.discard(p)
                
            # Try to add a random triple
            add_idx = random.randint(0, n_triples - 1)
            if add_idx in current_triple_set:
                # Restore and continue
                current_triples.insert(rem_idx_pos, rem_idx)
                current_triple_set.add(rem_idx)
                for p in triple_pairs[rem_idx]:
                    used_pairs.add(p)
                continue
                
            pairs_add = triple_pairs[add_idx]
            valid = True
            for p in pairs_add:
                if p in used_pairs:
                    valid = False
                    break
                    
            if valid:
                current_triples.append(add_idx)
                current_triple_set.add(add_idx)
                for p in pairs_add:
                    used_pairs.add(p)
                
                # Fill again to see if we can add more
                # Note: fill_system modifies current_triples and used_pairs
                # We need to be careful if we revert
                # For simplicity, we don't fill in every step, only occasionally or if alpha improves
                
                if steps % 10 == 0:
                    new_alpha = calc_alpha(current_triples)
                    if new_alpha < alpha:
                        alpha = new_alpha
                        if alpha < best_alpha:
                            best_alpha = alpha
                            best_solution = list(current_triples)
                            if best_alpha <= 4:
                                break
                    elif new_alpha == alpha:
                        pass
                    else:
                        # Revert
                        current_triples.pop()
                        current_triple_set.discard(add_idx)
                        for p in pairs_add:
                            used_pairs.discard(p)
                        current_triples.insert(rem_idx_pos, rem_idx)
                        current_triple_set.add(rem_idx)
                        for p in triple_pairs[rem_idx]:
                            used_pairs.add(p)
            else:
                current_triples.insert(rem_idx_pos, rem_idx)
                current_triple_set.add(rem_idx)
                for p in triple_pairs[rem_idx]:
                    used_pairs.add(p)
                    
        if best_alpha <= 4:
            break
            
    if best_solution is None:
        return [[0, 1, 2]]
        
    return [list(all_triples[i]) for i in best_solution]
