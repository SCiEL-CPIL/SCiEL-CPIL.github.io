import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    # Use 115s to be safe against 120s limit
    deadline = start_time + 115.0
    
    # Precompute all 4-term APs
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    APs = np.array(ap_list, dtype=np.int32)
    
    def get_cost(coloring):
        C = coloring[APs]
        mono_mask = (C[:, 0] == C[:, 1]) & (C[:, 1] == C[:, 2]) & (C[:, 2] == C[:, 3])
        return int(np.sum(mono_mask))

    def thue_morse(n):
        tm = [0]
        for _ in range(n.bit_length()):
            tm = tm + [1 - x for x in tm]
        return np.array(tm[:n], dtype=np.int32)

    def greedy_construct():
        c = np.zeros(n, dtype=np.int32)
        for i in range(n):
            c[i] = 0
            cost0 = get_cost(c)
            c[i] = 1
            cost1 = get_cost(c)
            if cost1 < cost0:
                c[i] = 1
            else:
                c[i] = 0
        return c

    def hill_climb(coloring, time_limit=None):
        current_cost = get_cost(coloring)
        best_coloring = coloring.copy()
        best_cost = current_cost
        
        positions = list(range(n))
        random.shuffle(positions)
        
        improved = True
        while improved:
            improved = False
            for pos in positions:
                old_val = coloring[pos]
                coloring[pos] = 1 - old_val
                
                new_cost = get_cost(coloring)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_coloring = coloring.copy()
                else:
                    coloring[pos] = old_val
                    
                if time_limit and time.time() > time_limit:
                    return best_coloring, best_cost
        return best_coloring, best_cost

    candidates = []
    candidates.append(thue_morse(n))
    candidates.append(greedy_construct())
    
    for _ in range(5):
        candidates.append(np.random.randint(0, 2, size=n, dtype=np.int32))

    best_init = None
    best_init_cost = float('inf')
    
    # Quick hill climb on each candidate
    for c in candidates:
        hc_limit = start_time + 10.0
        hc_result, hc_cost = hill_climb(c.copy(), hc_limit)
        if hc_cost < best_init_cost:
            best_init_cost = hc_cost
            best_init = hc_result
            
        if time.time() > deadline - 10:
            break

    if best_init is None:
        best_init = thue_morse(n)
        best_init_cost = get_cost(best_init)

    current_coloring = best_init.copy()
    current_cost = best_init_cost
    
    # Simulated Annealing
    T = 5.0
    target_T_end = 0.1
    max_iterations = 5_000_000
    cooling_rate = (target_T_end / T) ** (1.0 / max_iterations)
    
    best_sa_cost = current_cost
    best_sa_coloring = current_coloring.copy()
    
    steps_since_improvement = 0
    patience = 10000
    
    # Use local vars for speed
    rng = np.random.default_rng()
    
    for i in range(max_iterations):
        if time.time() > deadline - 5.0:
            break
            
        pos = rng.integers(0, n)
        
        old_val = current_coloring[pos]
        current_coloring[pos] = 1 - old_val
        
        new_cost = get_cost(current_coloring)
        delta = new_cost - current_cost
        
        if delta < 0:
            current_cost = new_cost
            steps_since_improvement = 0
            if new_cost < best_sa_cost:
                best_sa_cost = new_cost
                best_sa_coloring = current_coloring.copy()
        else:
            if T > 0.01:
                if delta < 50 * T:
                    r = random.random()
                    if r < math.exp(-delta / T):
                        current_cost = new_cost
                    else:
                        current_coloring[pos] = old_val
                else:
                    current_coloring[pos] = old_val
            else:
                current_coloring[pos] = old_val
        
        T *= cooling_rate
        
        if steps_since_improvement > patience:
            T = max(T, 2.0)
            steps_since_improvement = 0

    final_coloring, final_cost = hill_climb(best_sa_coloring.copy(), deadline - 1.0)
    
    return final_coloring.tolist()
