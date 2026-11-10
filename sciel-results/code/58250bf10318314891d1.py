import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 12
    m = 40
    total_sets = 1 << n
    
    # Helper function to check strict subset comparability
    # Returns True if a < b or b < a
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b

    def calc_score(S):
        score = 0
        # S is a list of masks
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if is_comparable(S[i], S[j]):
                    score += 1
        return score

    def greedy_init():
        S = []
        in_S = [False] * total_sets
        
        # Start with empty set (mask 0)
        S.append(0)
        in_S[0] = True
        
        # G[y] = number of comparable sets in S
        G = [0] * total_sets
        # 0 is comparable to all
        for y in range(1, total_sets):
            G[y] = 1
        
        for _ in range(m - 1):
            best_y = -1
            best_g = -1
            # Find y with max G[y] not in S
            for y in range(total_sets):
                if not in_S[y]:
                    if G[y] > best_g:
                        best_g = G[y]
                        best_y = y
            
            S.append(best_y)
            in_S[best_y] = True
            
            # Update G for all other candidates
            for u in range(total_sets):
                if not in_S[u]:
                    if is_comparable(u, best_y):
                        G[u] += 1
        return S

    def local_search(S):
        # S is a list of masks
        in_S = [False] * total_sets
        for x in S:
            in_S[x] = True
            
        # Precompute comparable counts within S for each element
        # C[i] = number of j in S such that S[i] is comparable to S[j]
        C = [0] * len(S)
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j and is_comparable(S[i], S[j]):
                    C[i] += 1
        
        current_score = calc_score(S)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_i = -1
            best_y = -1
            
            # Try swapping each element S[i] with some y not in S
            for i in range(len(S)):
                x = S[i]
                # Cost of removing x: we lose comparable pairs with other S[k]
                # This is C[i]
                cost_remove = C[i]
                
                # Find best y to add
                # Gain of adding y: comparable pairs with S[k] for k != i
                # We need to calculate comparable count of y with S \ {x}
                # gain_add = count of k in S, k != i, such that is_comparable(S[k], y)
                
                # We can iterate through all possible y not in S
                # To speed up, we could precompute G[y] for current S, 
                # but G[y] includes comparison with x.
                # G_current[y] = count of k in S such that is_comparable(S[k], y)
                # gain_add = G_current[y] - (1 if is_comparable(x, y) else 0)
                
                # Let's just iterate y. total_sets is 4096. len(S) is 40.
                # 40 * 4096 = 163,840 iterations per pass.
                # is_comparable is 2 ANDs and 2 comparisons. Very fast.
                
                for y in range(total_sets):
                    if in_S[y]:
                        continue
                    
                    # Calculate gain
                    gain = 0
                    for k in range(len(S)):
                        if k == i:
                            continue
                        if is_comparable(S[k], y):
                            gain += 1
                    
                    delta = gain - cost_remove
                    if delta > best_delta:
                        best_delta = delta
                        best_i = i
                        best_y = y
            
            if best_delta > 0:
                # Apply swap
                x_old = S[best_i]
                S[best_i] = best_y
                in_S[x_old] = False
                in_S[best_y] = True
                
                current_score += best_delta
                
                # Update C array
                # C[i] for the new element at best_i is 'gain' calculated above?
                # Actually, C[best_i] should be the number of comparable elements in new S.
                # That is exactly 'gain' + 0 (it doesn't compare to itself).
                # Wait, C[i] counts comparable j in S.
                # For the new element at index best_i, it is comparable to 'gain' other elements.
                # So C[best_i] = gain.
                
                # For other elements k != best_i:
                # If S[k] was comparable to x_old, decrement C[k]
                # If S[k] is comparable to S[best_i] (new), increment C[k]
                
                x_new = S[best_i]
                C[best_i] = gain
                
                for k in range(len(S)):
                    if k == best_i:
                        continue
                    if is_comparable(S[k], x_old):
                        C[k] -= 1
                    if is_comparable(S[k], x_new):
                        C[k] += 1
                
                improved = True
            else:
                improved = False
                
        return S, current_score

    def perturbation(S, num_swaps):
        S_perturbed = S.copy()
        in_S = [False] * total_sets
        for x in S_perturbed:
            in_S[x] = True
            
        for _ in range(num_swaps):
            # Pick random index to remove
            i = random.randint(0, len(S_perturbed) - 1)
            x = S_perturbed[i]
            
            # Pick random y not in S
            y = random.randint(0, total_sets - 1)
            while in_S[y]:
                y = random.randint(0, total_sets - 1)
            
            S_perturbed[i] = y
            in_S[x] = False
            in_S[y] = True
            
        return S_perturbed

    # Main loop
    # 1. Greedy initialization
    S = greedy_init()
    best_S, best_score = local_search(S)
    
    # 2. Iterated Local Search
    current_S = best_S.copy()
    
    while time.time() < deadline - 1.0:
        # Perturb
        # Start with small perturbations, maybe increase if stuck?
        # Simple random walk perturbation
        num_swaps = random.randint(1, 5)
        S_perturbed = perturbation(current_S, num_swaps)
        
        # Local Search
        S_ls, score_ls = local_search(S_perturbed)
        
        if score_ls > best_score:
            best_score = score_ls
            best_S = S_ls
            current_S = best_S.copy()
        else:
            # If we don't improve, we might want to perturb from the current best 
            # or accept worse solutions to escape basins? 
            # Standard ILS usually accepts worse with probability or just restarts from best.
            # Let's stick to restart from best to keep it simple and effective.
            pass

    # Convert best_S to lists of ints
    result = []
    for mask in best_S:
        s = []
        for i in range(n):
            if mask & (1 << i):
                s.append(i)
        result.append(s)
        
    return result

