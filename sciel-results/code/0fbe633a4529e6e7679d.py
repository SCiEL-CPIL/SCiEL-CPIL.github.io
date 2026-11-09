"""Combined program: fast greedy initialization + efficient local search with deadline guard."""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    n = 256
    
    # Precompute all 3-APs for fast lookup
    # ap_conflicts[x] = set of pairs (a,b) such that {a,x,b} or {x,a,b} or {a,b,x} forms a 3-AP
    # Better: for each x, precompute which other elements conflict with x
    # conflict[x] = set of y such that x and y are in some 3-AP together (with some third element)
    
    # Actually, let's precompute: for each pair (a, b), what is the third element c that would form a 3-AP?
    # If a < b: c = 2b - a (if c > b and c <= n) or c = a - (b-a) = 2a-b (if c < a and c >= 1)
    # So for each pair (a,b) with a < b, the potential 3-APs are:
    #   (a, b, 2b-a) if 2b-a <= n
    #   (2a-b, a, b) if 2a-b >= 1
    
    # Precompute: for each x, the set of "forbidden partners" - elements y such that 
    # adding both x and y to an empty set would create a 3-AP. But that's not quite right.
    # 
    # Better approach: precompute for each x, and for each potential third element z,
    # what pairs (y1, y2) would form a 3-AP with z? This is getting complex.
    #
    # Simplest fast approach: maintain a list of elements in S, and for can_add,
    # iterate over that list (size K ~ 50-80) instead of range(1, n+1) (size 256).
    
    # Precompute for each x, the set of pairs (y, z) that would form a 3-AP with x
    # For fast checking: precompute conflict masks
    # conflict_mask[x] is a bitset where bit y is set if x and y are in some 3-AP
    # But we need to check if adding x creates ANY 3-AP, which requires checking
    # if there exist y, z in S such that {x,y,z} is a 3-AP.
    
    # Better: precompute for each x, list of (y, z) pairs such that {x,y,z} is a 3-AP
    # Then can_add(x) checks if any such pair has both y and z in S.
    
    # Even faster: for each x, precompute all y such that there exists z with {x,y,z} AP
    # and for each such y, store the corresponding z. Then check if z is in S.
    
    # Let's precompute: for each x, a list of (y, z) where {x,y,z} forms a 3-AP
    # and y < z (to avoid duplicates)
    ap_pairs = {}
    for x in range(1, n + 1):
        pairs = []
        for y in range(1, n + 1):
            if y == x:
                continue
            # x, y, z: x is first, y is middle, z is last: z = 2y - x
            z = 2 * y - x
            if z != x and z != y and 1 <= z <= n:
                pairs.append((y, z))
            # x, y, z: y is first, x is middle, z is last: z = 2x - y
            z = 2 * x - y
            if z != x and z != y and 1 <= z <= n:
                pairs.append((y, z))
            # x, y, z: y is first, z is middle, x is last: z = (x + y) / 2
            # This is covered by the above when we swap roles
        ap_pairs[x] = pairs
    
    def can_add_fast(membership, elements_list, x):
        """Check if adding x to set maintains AP-free property using precomputed pairs."""
        for y, z in ap_pairs[x]:
            if membership[y] and membership[z]:
                return False
        return True
    
    # Greedy initialization: add smallest elements first (maximizes reciprocal sum)
    S = set()
    membership = np.zeros(n+1, dtype=bool)
    elements_list = []
    for x in range(1, n + 1):
        if can_add_fast(membership, elements_list, x):
            S.add(x)
            membership[x] = True
            elements_list.append(x)
    
    best_S = set(S)
    best_val = sum(1.0 / x for x in S)
    
    # Also try a greedy that prioritizes by reciprocal value (same as smallest first)
    # and a randomized greedy for diversity
    import random
    random.seed(42)
    
    # Try multiple greedy initializations
    best_init_S = set(S)
    best_init_val = best_val
    
    for trial in range(5):
        S_trial = set()
        membership_trial = np.zeros(n+1, dtype=bool)
        elements_list_trial = []
        
        # Randomized greedy: shuffle but bias toward small elements
        candidates = list(range(1, n + 1))
        # Weighted shuffle: smaller elements more likely to come first
        weights = [1.0 / x for x in candidates]
        total_w = sum(weights)
        
        while candidates:
            # Pick next candidate with probability proportional to weight
            r = random.random() * total_w
            cum = 0
            idx = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    idx = i
                    break
            x = candidates[idx]
            weights.pop(idx)
            total_w -= weights[idx] if idx < len(weights) else 0
            candidates.pop(idx)
            
            if can_add_fast(membership_trial, elements_list_trial, x):
                S_trial.add(x)
                membership_trial[x] = True
                elements_list_trial.append(x)
        
        val_trial = sum(1.0 / x for x in S_trial)
        if val_trial > best_init_val:
            best_init_val = val_trial
            best_init_S = set(S_trial)
            S = set(S_trial)
            membership = membership_trial.copy()
            elements_list = list(elements_list_trial)
            best_S = set(S)
            best_val = best_init_val
    
    # Local search with fast can_add
    while time.time() < deadline:
        improved = False
        excluded = set(range(1, n + 1)) - S
        
        # Phase 1: Try adding each excluded element (prioritize small x for max reciprocal)
        for x in sorted(excluded, key=lambda v: -1.0 / v):
            if time.time() > deadline:
                break
            if can_add_fast(membership, elements_list, x):
                new_val = best_val + 1.0 / x
                if new_val > best_val + 1e-15:
                    S.add(x)
                    membership[x] = True
                    elements_list.append(x)
                    best_S = set(S)
                    best_val = new_val
                    improved = True
                    excluded = set(range(1, n + 1)) - S
        
        if time.time() > deadline:
            break
        
        # Phase 2: Try single swaps (remove u, add v where 1/v > 1/u for improvement)
        if not improved:
            S_list = sorted(S, reverse=True)  # Try removing largest first
            excluded_list = sorted(set(range(1, n + 1)) - S, key=lambda v: -1.0 / v)
            
            for u in S_list:
                if time.time() > deadline:
                    break
                # Temporarily remove u
                membership[u] = False
                elements_list.remove(u)
                temp_val = best_val - 1.0 / u
                
                found_swap = False
                for v in excluded_list:
                    if v == u:
                        continue
                    if 1.0 / v <= 1.0 / u:
                        break  # We want 1/v > 1/u for improvement, i.e., v < u
                    if can_add_fast(membership, elements_list, v):
                        new_val = temp_val + 1.0 / v
                        if new_val > best_val + 1e-15:
                            S.remove(u)
                            S.add(v)
                            membership[u] = False
                            membership[v] = True
                            elements_list.append(v)
                            best_S = set(S)
                            best_val = new_val
                            improved = True
                            found_swap = True
                            break  # Restart swap search with new S
                
                if not found_swap:
                    # Restore u
                    membership[u] = True
                    elements_list.append(u)
                
                if improved:
                    break
        
        if not improved:
            # Phase 3: Try pair swaps (remove u1, u2, add v1, v2)
            if time.time() > deadline - 10.0:
                break
            
            S_list = sorted(S, reverse=True)
            excluded_list = sorted(set(range(1, n + 1)) - S, key=lambda v: -1.0 / v)
            
            found_improvement = False
            if len(S_list) >= 2:
                # Try all pairs of elements to remove (largest first)
                for i in range(len(S_list)):
                    if time.time() > deadline - 5.0:
                        break
                    u1 = S_list[i]
                    for j in range(i + 1, len(S_list)):
                        if time.time() > deadline - 5.0:
                            break
                        u2 = S_list[j]
                        
                        # Temporarily remove u1, u2
                        membership[u1] = False
                        membership[u2] = False
                        elements_list.remove(u1)
                        elements_list.remove(u2)
                        temp_val = best_val - 1.0 / u1 - 1.0 / u2
                        
                        # Try adding pairs of small excluded elements
                        for vi, v1 in enumerate(excluded_list[:40]):
                            if v1 == u1 or v1 == u2:
                                continue
                            if can_add_fast(membership, elements_list, v1):
                                membership[v1] = True
                                elements_list.append(v1)
                                for vj, v2 in enumerate(excluded_list[vi+1:40], start=vi+1):
                                    if v2 == u1 or v2 == u2 or v2 == v1:
                                        continue
                                    if can_add_fast(membership, elements_list, v2):
                                        new_val = temp_val + 1.0 / v1 + 1.0 / v2
                                        if new_val > best_val + 1e-15:
                                            S.discard(u1)
                                            S.discard(u2)
                                            S.add(v1)
                                            S.add(v2)
                                            membership[v1] = True
                                            membership[v2] = True
                                            elements_list.append(v2)
                                            best_S = set(S)
                                            best_val = new_val
                                            found_improvement = True
                                            break
                                # Restore v1 if no improvement found with it
                                if not found_improvement:
                                    membership[v1] = False
                                    elements_list.remove(v1)
                            if found_improvement:
                                break
                        
                        if not found_improvement:
                            # Restore u1, u2
                            membership[u1] = True
                            membership[u2] = True
                            elements_list.append(u1)
                            elements_list.append(u2)
                        
                        if found_improvement:
                            break
                    if found_improvement:
                        break
            
            if found_improvement:
                continue
            else:
                break
    
    # Final polish: try to add any remaining elements that fit
    if time.time() < deadline - 2.0:
        excluded = set(range(1, n + 1)) - best_S
        for x in sorted(excluded, key=lambda v: -1.0 / v):
            if time.time() > deadline - 2.0:
                break
            # Check against best_S
            temp_membership = np.zeros(n+1, dtype=bool)
            for elem in best_S:
                temp_membership[elem] = True
            if can_add_fast(temp_membership, list(best_S), x):
                best_S.add(x)
                best_val += 1.0 / x
    
    return list(best_S)
