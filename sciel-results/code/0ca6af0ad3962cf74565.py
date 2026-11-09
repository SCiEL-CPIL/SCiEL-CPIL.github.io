import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    n = 128
    # Precompute all 4-term APs
    aps_list = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps_list.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    aps = np.array(aps_list, dtype=np.int32)  # shape (num_aps, 4)
    num_aps = len(aps)
    
    # Map index to list of AP indices containing it
    idx_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps_list):
        for idx in ap:
            idx_to_aps[idx].append(i)
    
    idx_to_aps_arr = [np.array(lst, dtype=np.int32) for lst in idx_to_aps]

    def get_bad_aps(colors):
        """Return indices of APs with <= 2 distinct colors."""
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        
        matches = ((c0 == c1).astype(np.int8) + 
                   (c0 == c2).astype(np.int8) + 
                   (c0 == c3).astype(np.int8) + 
                   (c1 == c2).astype(np.int8) + 
                   (c1 == c3).astype(np.int8) + 
                   (c2 == c3).astype(np.int8))
        
        bad = matches >= 2
        return np.where(bad)[0]

    def count_bad(colors):
        """Efficiently count bad APs."""
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        
        matches = ((c0 == c1).astype(np.int8) + 
                   (c0 == c2).astype(np.int8) + 
                   (c0 == c3).astype(np.int8) + 
                   (c1 == c2).astype(np.int8) + 
                   (c1 == c3).astype(np.int8) + 
                   (c2 == c3).astype(np.int8))
        
        return np.sum(matches >= 2)

    best_solution = None
    best_cost = float('inf')
    
    # Try k = 3, 4, 5...
    for k in range(3, 10):
        if time.time() > deadline - 10:
            break
            
        # Simulated Annealing Search
        best_k_solution = None
        best_k_cost = float('inf')
        
        max_restarts = 50
        restart_count = 0
        while time.time() < deadline - 5 and restart_count < max_restarts:
            restart_count += 1
            
            # Random initialization
            colors = np.random.randint(0, k, n).astype(np.int32)
            current_cost = count_bad(colors)
            
            if current_cost == 0:
                best_k_solution = colors.copy()
                best_k_cost = 0
                break
                
            # Simulated Annealing
            T = 5.0
            alpha = 0.9995
            min_T = 0.01
            max_iters = 20000
            no_improve = 0
            max_no_improve = 5000
            
            for iter_count in range(max_iters):
                if time.time() > deadline - 5:
                    break
                if T < min_T:
                    break
                    
                if current_cost == 0:
                    best_k_solution = colors.copy()
                    best_k_cost = 0
                    break
                    
                # Pick a random position
                i = random.randint(0, n - 1)
                old_color = colors[i]
                
                # Pick a new random color
                if k == 1:
                    new_color = old_color
                else:
                    new_color = random.randint(0, k - 1)
                    while new_color == old_color:
                        new_color = random.randint(0, k - 1)
                        
                # Apply move
                colors[i] = new_color
                new_cost = count_bad(colors)
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Accept
                    current_cost = new_cost
                    if current_cost < best_k_cost:
                        best_k_cost = current_cost
                        best_k_solution = colors.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Reject
                    colors[i] = old_color
                    no_improve += 1
                    
                T *= alpha
                
                if no_improve > max_no_improve:
                    break
                    
            if best_k_cost == 0:
                break

        if best_k_cost == 0:
            return [int(c) for c in best_k_solution]
            
    # Fallback: return the best solution found for any k
    if best_solution is not None:
        return [int(c) for c in best_solution]
        
    # Absolute fallback: Greedy construction (same as before)
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
