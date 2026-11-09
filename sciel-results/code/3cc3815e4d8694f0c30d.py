import time
import math

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Candidates 1..511
    N = 511
    candidates = list(range(1, N + 1))
    # Sort by weight descending (larger a -> larger weight)
    candidates.sort(reverse=True)
    
    # Precompute weights
    weights = {a: 1.0 / (512.0 - a) for a in candidates}
    
    # Precompute GCD matrix for speed
    # gcd_matrix[i][j] = gcd(i, j)
    # Using a 2D list of size 512x512
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    selected = []
    selected_set = set()
    
    # 1. Greedy Initialization
    for a in candidates:
        if time.time() > deadline:
            break
        ok = True
        for b in selected:
            if gcd_mat[a][b] != 1:
                ok = False
                break
        if ok:
            selected.append(a)
            selected_set.add(a)

    best_score = sum(weights[a] for a in selected)

    # 2. Local Search (Swap/Insert)
    # We try to improve the set by inserting unselected 'a' 
    # and removing conflicting selected 'b's if the net gain is positive.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate through candidates in descending order of weight
        for a in candidates:
            if a in selected_set:
                continue
            
            # Find conflicts with currently selected items
            conflicts = []
            for b in selected:
                if gcd_mat[a][b] != 1:
                    conflicts.append(b)
            
            # If no conflicts, we can just add it
            if not conflicts:
                selected.append(a)
                selected_set.add(a)
                best_score += weights[a]
                improved = True
                break  # Restart loop to re-evaluate with new state
            
            # If there are conflicts, check if swapping is beneficial
            # Gain = weight(a) - sum(weights(conflicts))
            loss = sum(weights[b] for b in conflicts)
            gain = weights[a] - loss
            
            if gain > 1e-12:
                # Perform the swap
                for b in conflicts:
                    selected.remove(b)
                    selected_set.remove(b)
                
                selected.append(a)
                selected_set.add(a)
                best_score += gain
                improved = True
                break  # Restart loop

    return selected
