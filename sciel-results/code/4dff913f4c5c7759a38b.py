import random
import time
import sys

def construct():
    n = 16
    # Pre-compute all possible triples
    all_triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                all_triples.append((i, j, k))
                
    # Function to calculate independence number
    def calc_alpha(triples):
        triple_masks = [ (1<<a) | (1<<b) | (1<<c) for a,b,c in triples ]
        max_size = 0
        max_mask = 0
        # Iterate through all masks
        for mask in range(1 << n):
            # Check if mask is independent
            is_indep = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_indep = False
                    break
            if is_indep:
                size = mask.bit_count()
                if size > max_size:
                    max_size = size
                    max_mask = mask
        return max_size, max_mask

    # Helper to get pairs of a triple
    def get_pairs(t):
        a, b, c = t
        return [(a,b), (a,c), (b,c)]

    # Helper to check if a triple can be added given used_pairs
    def can_add(triple, used_pairs):
        for p in get_pairs(triple):
            if p in used_pairs:
                return False
        return True

    # Initial random linear hypergraph
    best_alpha = n
    best_config = []
    
    start_time = time.time()
    time_limit = 110 # 110 seconds to be safe

    for restart in range(200):
        if time.time() - start_time > time_limit:
            break
            
        # Generate a random maximal linear hypergraph
        cur_triples = []
        used_pairs_set = set()
        
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        for t in shuffled:
            if can_add(t, used_pairs_set):
                cur_triples.append(t)
                used_pairs_set.update(get_pairs(t))
                
        # Evaluate alpha
        alpha, max_mask = calc_alpha(cur_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_config = cur_triples[:]
            
        # Local Search
        # We try to improve alpha by swapping triples.
        # A move: Remove T1, Add T2.
        # Constraint: T2 must be valid with (Cur - {T1}).
        # Objective: New Alpha <= Old Alpha (preferably <)
        
        cur_set = set(cur_triples)
        cur_used_pairs = set(used_pairs_set)
        
        for step in range(200):
            if time.time() - start_time > time_limit:
                break
            
            # Try to find a beneficial swap
            # Pick a random triple to remove
            if not cur_triples:
                break
            t_remove = random.choice(cur_triples)
            
            # Temporarily remove
            cur_set.remove(t_remove)
            for p in get_pairs(t_remove):
                cur_used_pairs.remove(p)
                
            # Find a valid triple to add
            # Prioritize triples that intersect the current largest independent set
            # We need the current largest independent set.
            # Re-calculating alpha is expensive. 
            # Heuristic: Try a few random valid triples.
            
            # Get list of valid triples to add
            # This is O(N^3) if we check all. 
            # Optimization: Pre-compute for each pair, which triples use it?
            # Or just iterate all_triples and check can_add.
            
            added = False
            # Try up to 10 random candidates
            candidates = random.sample(all_triples, min(10, len(all_triples)))
            # Also include t_remove itself to potentially revert
            candidates.append(t_remove)
            
            best_candidate = None
            best_new_alpha = alpha + 1 # We want to find one with alpha <= current
            
            for t_add in candidates:
                if t_add == t_remove:
                    continue
                if can_add(t_add, cur_used_pairs):
                    # Tentatively add
                    # Check if this swap improves or maintains alpha
                    # To avoid full calc_alpha every time, we can just accept if we can't evaluate fast.
                    # But calc_alpha is ~2M ops. 200 steps * 200 restarts * 10 candidates * 2M = 800B ops. Too slow.
                    
                    # We need a faster evaluation or fewer candidates.
                    # Let's just do 1 candidate per step.
                    # And only evaluate alpha if the candidate is promising?
                    # Or just evaluate alpha for the swap.
                    
                    # Let's try just one random valid swap per step.
                    # If it's valid, swap and evaluate.
                    # If alpha improves or stays same, keep. Else revert.
                    
                    # For now, let's just try to add t_add
                    cur_set.add(t_add)
                    cur_used_pairs.update(get_pairs(t_add))
                    
                    new_alpha, _ = calc_alpha(list(cur_set))
                    
                    if new_alpha <= alpha:
                        # Accept
                        cur_triples = list(cur_set)
                        alpha = new_alpha
                        added = True
                        if alpha < best_alpha:
                            best_alpha = alpha
                            best_config = cur_triples[:]
                    else:
                        # Revert
                        cur_set.remove(t_add)
                        for p in get_pairs(t_add):
                            cur_used_pairs.remove(p)
                        
            # Restore t_remove if we didn't add anything else
            if not added:
                cur_set.add(t_remove)
                cur_used_pairs.update(get_pairs(t_remove))
                cur_triples = list(cur_set)
                
            # If we improved, break early? No, continue to find better.
            
    # Convert best_config to list of lists
    return [list(t) for t in best_config]
