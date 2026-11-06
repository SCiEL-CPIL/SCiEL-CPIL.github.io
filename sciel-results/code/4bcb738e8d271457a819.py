import random
import time
import math
from itertools import combinations

def construct():
    n = 16
    # Pre-compute all possible triples
    all_triples = []
    triple_to_idx = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                t = (i, j, k)
                idx = len(all_triples)
                all_triples.append(t)
                triple_to_idx[t] = idx
                
    # Pre-compute masks for all triples for fast intersection checks
    triple_masks = [(1<<a) | (1<<b) | (1<<c) for a,b,c in all_triples]
    
    # Pre-compute pairs for each triple
    triple_pairs = []
    for a,b,c in all_triples:
        triple_pairs.append([(a,b), (a,c), (b,c)])
        
    # Pre-compute all subsets of size k for fast alpha calculation
    subsets_by_size = {k: [] for k in range(n+1)}
    for k in range(n+1):
        for combo in combinations(range(n), k):
            mask = 0
            for p in combo:
                mask |= (1 << p)
            subsets_by_size[k].append(mask)
            
    def calc_alpha_fast(active_indices):
        """
        Calculate independence number.
        active_indices: list of indices into all_triples
        """
        if not active_indices:
            return n
            
        # Get the masks of the active triples
        active_masks = [triple_masks[i] for i in active_indices]
        
        # Check from largest size down
        for k in range(n, 0, -1):
            for mask in subsets_by_size[k]:
                is_indep = True
                for tm in active_masks:
                    if (mask & tm) == tm:
                        is_indep = False
                        break
                if is_indep:
                    return k
        return 0

    # Function to generate initial random linear hypergraph
    def gen_initial():
        shuffled_indices = list(range(len(all_triples)))
        random.shuffle(shuffled_indices)
        cur_indices = []
        used_pairs = set()
        for idx in shuffled_indices:
            a, b, c = all_triples[idx]
            pairs = triple_pairs[idx]
            if not any(p in used_pairs for p in pairs):
                cur_indices.append(idx)
                used_pairs.update(pairs)
        return cur_indices, used_pairs

    # Main optimization loop
    start_time = time.time()
    time_limit = 110 # 10s buffer
    
    best_alpha = n
    best_config = []
    
    # Run multiple restarts with Simulated Annealing
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        cur_indices, cur_used_pairs = gen_initial()
        
        # Calculate initial alpha
        cur_alpha = calc_alpha_fast(cur_indices)
        
        if cur_alpha < best_alpha:
            best_alpha = cur_alpha
            best_config = [all_triples[i] for i in cur_indices]
            
        # Simulated Annealing
        T = 1.0
        cooling = 0.995
        min_T = 0.01
        
        while T > min_T and time.time() - start_time < time_limit:
            # Pick a random triple to remove
            if not cur_indices:
                break
            remove_pos = random.randint(0, len(cur_indices) - 1)
            remove_idx = cur_indices[remove_pos]
            
            # Pick a random triple to add
            added = False
            for _ in range(50): # Try up to 50 random candidates
                add_idx = random.randint(0, len(all_triples) - 1)
                
                if add_idx == remove_idx:
                    continue
                if add_idx in cur_indices:
                    continue
                    
                # Check if add_idx conflicts with cur_indices \ {remove_idx}
                # We can check if any pair of add_idx is in cur_used_pairs
                # BUT we must exclude the pairs of remove_idx from this check
                # This is tricky with a simple set.
                
                # Efficient check:
                # 1. Temporarily remove remove_idx's pairs from a copy? No, too slow.
                # 2. Check each pair of add_idx against all pairs of other triples?
                
                # Let's just check against all triples in cur_indices except remove_idx
                conflict = False
                add_pairs = triple_pairs[add_idx]
                for pos, idx in enumerate(cur_indices):
                    if pos == remove_pos:
                        continue
                    other_pairs = triple_pairs[idx]
                    for p1 in add_pairs:
                        for p2 in other_pairs:
                            if p1 == p2:
                                conflict = True
                                break
                        if conflict:
                            break
                    if conflict:
                        break
                        
                if not conflict:
                    # Valid move
                    # Perform move
                    new_indices = cur_indices[:]
                    new_indices[remove_pos] = add_idx
                    # Note: order doesn't matter for set logic, but we need to update used_pairs correctly
                    # Actually, we can just rebuild used_pairs for the new state?
                    # Or update incrementally.
                    # Remove remove_idx pairs, Add add_idx pairs.
                    # But wait, if we just swap, the pairs of remove_idx are freed, and pairs of add_idx are taken.
                    # Since we verified no conflict with others, the new pairs of add_idx are NOT in the pairs of others.
                    # So we can update cur_used_pairs:
                    # Remove pairs of remove_idx
                    for p in triple_pairs[remove_idx]:
                        cur_used_pairs.remove(p)
                    # Add pairs of add_idx
                    for p in add_pairs:
                        cur_used_pairs.add(p)
                        
                    cur_indices = new_indices
                    cur_alpha_new = calc_alpha_fast(cur_indices)
                    
                    delta_alpha = cur_alpha_new - cur_alpha
                    
                    if delta_alpha < 0 or random.random() < math.exp(-delta_alpha / T):
                        # Accept
                        cur_alpha = cur_alpha_new
                        
                        if cur_alpha < best_alpha:
                            best_alpha = cur_alpha
                            best_config = [all_triples[i] for i in cur_indices]
                        
                        added = True
                    else:
                        # Reject: Revert changes
                        # Remove add_idx pairs
                        for p in add_pairs:
                            cur_used_pairs.remove(p)
                        # Add remove_idx pairs
                        for p in triple_pairs[remove_idx]:
                            cur_used_pairs.add(p)
                        cur_indices = cur_indices[:] # Revert list
                        # Actually, we modified cur_indices in place? No, we made new_indices.
                        # So cur_indices is still old. We just need to revert cur_used_pairs.
                        # And we didn't update cur_alpha.
                        pass
                    break # Try only one move per temperature step
                
            T *= cooling
            
    return [list(t) for t in best_config]
