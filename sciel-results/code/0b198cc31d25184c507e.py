import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    N = 511
    
    # 1. Precompute prime factors for all numbers 1..511
    # factors[i] will be a set of prime factors of i
    factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not factors[i]:  # i is prime
            for j in range(i, N + 1, i):
                factors[j].add(i)
    
    # 2. Prepare candidates and weights
    # We process in descending order of weight (which corresponds to descending a)
    candidates = list(range(N, 0, -1))
    weights = [0.0] * (N + 1)
    for a in candidates:
        weights[a] = 1.0 / (512 - a)
        
    # 3. Beam Search
    # State: (score, frozenset_of_factors, tuple_of_chosen)
    # We use a list of tuples.
    beam_width = 20000
    
    # Initial state: score 0, empty factors, empty chosen
    # Using frozenset() for empty set
    states = [(0.0, frozenset(), ())]
    
    best_score = 0.0
    best_chosen = ()
    
    # We track the best score seen in the entire process, not just the beam
    # But since we prune, the best in the beam is usually the best overall
    # unless we prune a path that leads to a better solution later.
    # However, standard beam search keeps the top K.
    
    for idx, a in enumerate(candidates):
        if time.time() > deadline:
            break
            
        w_a = weights[a]
        fac_a = factors[a]
        
        new_states = []
        
        for score, fac_set, chosen in states:
            # Option 1: Exclude a
            # We don't need to create a new object if we are just carrying it over,
            # but we are building a new list.
            new_states.append((score, fac_set, chosen))
            
            # Option 2: Include a
            if not (fac_a & fac_set):
                new_score = score + w_a
                if new_score > best_score:
                    best_score = new_score
                    best_chosen = chosen + (a,)
                
                new_fac = frozenset(fac_set | fac_a)
                new_chosen = chosen + (a,)
                new_states.append((new_score, new_fac, new_chosen))
        
        # Prune: keep top beam_width states by score
        # Sort by score descending
        # Optimization: If new_states is huge, this is the bottleneck.
        # 20000 states -> 40000 new states. Sorting 40k items is fast.
        new_states.sort(key=lambda x: x[0], reverse=True)
        states = new_states[:beam_width]
        
        # Check deadline again after sort
        if time.time() > deadline:
            break

    # 4. Local Refinement
    # The beam search gives a good candidate. Let's try to improve it locally.
    # Convert best_chosen to a set for faster lookup
    if best_chosen:
        current_set = list(best_chosen)
        current_score = best_score
        in_set = set(current_set)
        
        # Helper to check if a number can be added to a set
        def can_add(a, set_list, set_obj):
            fac_a = factors[a]
            for b in set_list:
                if fac_a & factors[b]:
                    return False
            return True
            
        # Try single swaps: remove i, add j
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current_set)):
                if time.time() > deadline:
                    break
                P = current_set[i]
                # Remove P
                remaining = [x for k, x in enumerate(current_set) if k != i]
                remaining_set = set(remaining)
                
                # Find best candidate to add
                # We want max weight a such that a not in remaining_set and coprime to all in remaining
                # Iterate candidates in descending order of weight (511 down to 1)
                best_add = None
                best_add_score = 0.0
                for a in range(N, 0, -1):
                    if a in remaining_set:
                        continue
                    if factors[a] & factors[P]:
                        # If a shares factor with P, it might still be coprime to others,
                        # but we are replacing P, so P is gone.
                        # We just need a to be coprime to remaining.
                        pass
                    if can_add(a, remaining, remaining_set):
                        # Since we iterate 511->1, the first one found is the best
                        best_add = a
                        best_add_score = weights[a]
                        break
                
                if best_add is not None:
                    delta = best_add_score - weights[P]
                    if delta > 1e-9:
                        # Perform swap
                        current_set[i] = best_add
                        current_score += delta
                        in_set = set(current_set)
                        improved = True
                        break # Restart loop with new set
            
            # Try add-only moves (if beam search didn't saturate)
            if not improved:
                for a in range(N, 0, -1):
                    if a in in_set:
                        continue
                    if can_add(a, current_set, in_set):
                        current_set.append(a)
                        current_score += weights[a]
                        in_set.add(a)
                        improved = True
                        break
                        
        if current_score > best_score:
            best_score = current_score
            best_chosen = tuple(current_set)

    return list(best_chosen)
