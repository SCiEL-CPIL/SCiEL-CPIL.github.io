import numpy as np
import time
import random
import math
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix, vstack

def construct():
    n = 256
    start_time = time.time()
    max_time = 55.0
    
    # Precompute 3-APs and 4-APs
    ap3s = []
    for d in range(1, n):
        for a in range(0, n - 2 * d):
            ap3s.append((a, a + d, a + 2 * d))
            
    ap4s = []
    for d in range(1, n):
        for a in range(0, n - 3 * d):
            ap4s.append((a, a + d, a + 2 * d, a + 3 * d))
            
    num_3aps = len(ap3s)
    num_4aps = len(ap4s)
    
    def count_3aps(state):
        count = 0
        for ap in ap3s:
            if all(state[i] == 1 for i in ap):
                count += 1
        return count
        
    def count_4aps(state):
        count = 0
        for ap in ap4s:
            if all(state[i] == 1 for i in ap):
                count += 1
        return count
        
    # Helper to get neighbors for fast delta calculation in fallback
    elem_3aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap3s):
        for i in ap:
            elem_3aps[i].append(idx)
            
    elem_4aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap4s):
        for i in ap:
            elem_4aps[i].append(idx)

    def fast_local_search(state, time_limit):
        """
        A simple, robust local search from scratch.
        """
        current_state = state[:]
        current_3ap = count_3aps(current_state)
        
        # Repair 4-APs
        while count_4aps(current_state) > 0 and (time.time() - start_time) < time_limit:
            # Find a 4-AP
            ap_found = None
            for ap in ap4s:
                if all(current_state[i] == 1 for i in ap):
                    ap_found = ap
                    break
            if ap_found is None: break
            # Remove one element from it (random or lowest degree)
            # Pick random to avoid bias
            victim = random.choice(ap_found)
            current_state[victim] = 0
            
        # Improvement Loop
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            # Try adding elements
            for i in range(n):
                if current_state[i] == 0:
                    # Check if adding i creates 4-AP
                    ok = True
                    for ap_idx in elem_4aps[i]:
                        ap = ap4s[ap_idx]
                        if all(current_state[j] == 1 for j in ap if j != i):
                            ok = False
                            break
                    if ok:
                        # Check gain in 3-APs
                        gain = 0
                        for ap_idx in elem_3aps[i]:
                            ap = ap3s[ap_idx]
                            if all(current_state[j] == 1 for j in ap if j != i):
                                gain += 1
                        if gain > 0:
                            current_state[i] = 1
                            current_3ap += gain
                            improved = True
            # Try removing elements that don't contribute? (Optional, usually not needed if we maximize)
            # Try swaps? (Complex, skip for simplicity)
            
        return current_state, current_3ap

    # 1. Try ILP
    ilp_success = False
    try:
        # Variables: x (n), y (num_3aps)
        c = np.zeros(n + num_3aps)
        c[n:] = 1.0
        
        rows1 = []
        cols1 = []
        vals1 = []
        for j, (a, b, c_idx) in enumerate(ap3s):
            # y_j - x_a <= 0
            rows1.append(j); cols1.append(a); vals1.append(-1.0)
            rows1.append(j); cols1.append(n + j); vals1.append(1.0)
            # y_j - x_b <= 0
            rows1.append(j); cols1.append(b); vals1.append(-1.0)
            rows1.append(j); cols1.append(n + j); vals1.append(1.0)
            # y_j - x_c <= 0
            rows1.append(j); cols1.append(c_idx); vals1.append(-1.0)
            rows1.append(j); cols1.append(n + j); vals1.append(1.0)
            
        A1 = coo_matrix((vals1, (rows1, cols1)), shape=(num_3aps, n + num_3aps))
        b1 = np.zeros(num_3aps)
        
        rows2 = []
        cols2 = []
        vals2 = []
        for k, (a, b, c_idx, d) in enumerate(ap4s):
            rows2.extend([k, k, k, k])
            cols2.extend([a, b, c_idx, d])
            vals2.extend([1.0, 1.0, 1.0, 1.0])
            
        A2 = coo_matrix((vals2, (rows2, cols2)), shape=(num_4aps, n + num_3aps))
        b2 = np.full(num_4aps, 3.0)
        
        A_ub = vstack([A1, A2]).tocsr()
        b_ub = np.concatenate([b1, b2])
        
        lb = np.zeros(n + num_3aps)
        ub = np.ones(n + num_3aps)
        bounds = Bounds(lb, ub)
        
        integrality = np.zeros(n + num_3aps)
        integrality[:n] = 1.0
        
        constraints = LinearConstraint(A_ub, -np.inf, b_ub)
        
        remaining = max_time - (time.time() - start_time) - 5
        if remaining > 1:
            res = milp(c, integrality=integrality, bounds=bounds, constraints=constraints, options={"time_limit": remaining})
            
            if res.status == 0 or res.status == 1:
                x_sol = res.x[:n]
                x_int = np.round(x_sol).astype(int)
                state = x_int.tolist()
                
                # Verify
                if count_4aps(state) == 0:
                    score = count_3aps(state)
                    # Polish with local search
                    state, score = fast_local_search(state, max_time - 2)
                    return [i+1 for i in range(n) if state[i]==1]
                else:
                    # Invalid, repair
                    state, score = fast_local_search(state, max_time - 2)
                    if count_4aps(state) == 0:
                        return [i+1 for i in range(n) if state[i]==1]
    except Exception:
        pass

    # 2. Fallback: Randomized Greedy + Local Search
    # Run multiple random restarts
    best_state = None
    best_score = -1
    time_left = max_time - (time.time() - start_time)
    
    num_restarts = int(time_left / 2) if time_left > 2 else 1
    num_restarts = max(1, min(num_restarts, 5))
    
    for r in range(num_restarts):
        # Random initialization
        state = [1 if random.random() < 0.5 else 0 for _ in range(n)]
        # Repair and Improve
        state, score = fast_local_search(state, (max_time - (time.time() - start_time)) / (num_restarts - r))
        
        if count_4aps(state) == 0 and score > best_score:
            best_score = score
            best_state = state[:]
            
    if best_state is not None:
        return [i+1 for i in range(n) if best_state[i]==1]
    else:
        # Absolute fallback: single element
        return [1]

