"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Rank-matching heuristic
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    deadline = 115.0
    
    # 2. Local Search (2-opt) with time-limited passes
    pass_count = 0
    max_passes = 10
    while pass_count < max_passes and (time.time() - start_time) < deadline:
        improved = False
        for i in range(n):
            if (time.time() - start_time) > deadline:
                break
            for j in range(i + 1, n):
                dA = a[i, :] - a[j, :]
                dB = b[p[j], p] - b[p[i], p]
                dB[i] = 0
                dB[j] = 0
                delta = np.dot(dA, dB)
                
                if delta < -1e-10:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
        pass_count += 1
        if not improved:
            break
    
    # 3. Tabu Search with time limit
    tabu_list = {}  # (i, j) -> iteration when it becomes non-tabu
    tabu_tenure = 50
    iters = 0
    max_iters = 20000
    
    while iters < max_iters and (time.time() - start_time) < deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            iters += 1
            continue
        
        # Check tabu status
        move_key = (min(i, j), max(i, j))
        is_tabu = move_key in tabu_list and tabu_list[move_key] > iters
        
        dA = a[i, :] - a[j, :]
        dB = b[p[j], p] - b[p[i], p]
        dB[i] = 0
        dB[j] = 0
        delta = np.dot(dA, dB)
        
        # Accept if improving or not tabu
        if delta < -1e-10 or not is_tabu:
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            # Add to tabu list
            tabu_list[move_key] = iters + tabu_tenure
            
            # Clean old tabu entries
            if len(tabu_list) > 1000:
                tabu_list = {k: v for k, v in tabu_list.items() if v > iters}
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        iters += 1
        
        if iters % 500 == 0 and (time.time() - start_time) > deadline:
            break
    
    # 4. Simulated Annealing for diversification
    T = 5000.0 
    T_end = 0.1
    alpha = 0.995
    
    iters = 0
    max_iters = 30000
    while T > T_end and iters < max_iters and (time.time() - start_time) < deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            iters += 1
            continue
            
        dA = a[i, :] - a[j, :]
        dB = b[p[j], p] - b[p[i], p]
        dB[i] = 0
        dB[j] = 0
        delta = np.dot(dA, dB)
        
        if delta < -1e-10:
            accept = True
        else:
            exp_arg = -delta / T
            if exp_arg > -50:
                accept = np.random.rand() < np.exp(exp_arg)
            else:
                accept = False
        
        if accept:
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
        T *= alpha
        iters += 1
        
        if iters % 1000 == 0 and (time.time() - start_time) > deadline:
            break
    
    # 5. Final polish: local search to clean up
    if (time.time() - start_time) < deadline - 3.0:
        p2 = best_p.copy()
        c2 = best_cost
        for _ in range(3):
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dA = a[i, :] - a[j, :]
                    dB = b[p2[j], p2] - b[p2[i], p2]
                    dB[i] = 0
                    dB[j] = 0
                    delta = np.dot(dA, dB)
                    if delta < -1e-10:
                        p2[i], p2[j] = p2[j], p2[i]
                        c2 += delta
                        improved = True
                        if c2 < best_cost:
                            best_cost = c2
                            best_p = p2.copy()
            if not improved:
                break
            if (time.time() - start_time) > deadline - 3.0:
                break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
