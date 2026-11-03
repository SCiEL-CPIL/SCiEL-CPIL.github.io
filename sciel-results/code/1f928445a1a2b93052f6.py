import numpy as np
import time

def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    pi = p[i]
    pj = p[j]
    # Vectorized delta calculation
    # delta = sum_k a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + a[j,k]*(b[pi,p[k]] - b[pj,p[k]])
    #       + sum_k a[k,i]*(b[p[k],pj] - b[p[k],pi]) + a[k,j]*(b[p[k],pi] - b[p[k],pj])
    #       + a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
    
    # Let's use the existing fast version
    bpj = b[:, pj][p]
    bpi = b[:, pi][p]
    brj = b[pj, :][p]
    bri = b[pi, :][p]
    
    delta = (a[:, i] * (bpj - bpi) +
             a[:, j] * (bpi - bpj) +
             a[i, :] * (brj - bri) +
             a[j, :] * (bri - brj))
    delta[i] = 0
    delta[j] = 0
    direct = a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])
    return int(np.sum(delta) + direct)

def _local_search_2opt(a, b, p, current_cost, deadline, max_passes=100):
    n = len(p)
    for _ in range(max_passes):
        if time.time() >= deadline - 1:
            break
        best_d = 0
        best_i, best_j = -1, -1
        for i in range(n):
            if time.time() >= deadline - 1:
                break
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < best_d:
                    best_d = d
                    best_i, best_j = i, j
        if best_i == -1:
            break
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_d
    return p, current_cost

def solve(a, b):
    n = len(a)
    deadline = time.time() + 115
    rng = np.random.default_rng(42)
    
    # 1. Initialization: Rank Matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Initial Local Search
    p, cost = _local_search_2opt(a, b, p, _qap_cost(a, b, p), deadline)
    best_p = p.copy()
    best_cost = cost
    
    # 2. Intensive Local Search (ILS) with Perturbation
    # We use a perturbation strategy: random swaps followed by local search.
    # To avoid cycling, we can use a simple memory or just random restarts if stuck.
    
    stagnation = 0
    perturbation_strength = 5
    
    while time.time() < deadline - 5:
        # Perturb the current best solution
        p_curr = best_p.copy()
        cost_curr = best_cost
        
        for _ in range(perturbation_strength):
            i, j = rng.integers(0, n, size=2)
            if i != j:
                d = _swap_delta(a, b, p_curr, i, j)
                p_curr[i], p_curr[j] = p_curr[j], p_curr[i]
                cost_curr += d
        
        # Local Search on perturbed solution
        p_curr, cost_curr = _local_search_2opt(a, b, p_curr, cost_curr, deadline)
        
        if cost_curr < best_cost:
            best_cost = cost_curr
            best_p = p_curr.copy()
            stagnation = 0
            # Optionally reduce perturbation strength if improving
            if perturbation_strength > 2:
                perturbation_strength = max(2, perturbation_strength - 1)
        else:
            stagnation += 1
            # Increase perturbation strength if stuck
            if stagnation % 10 == 0:
                perturbation_strength = min(50, perturbation_strength + 5)
    
    # 3. Final Polish
    p_final = best_p.copy()
    cost_final = best_cost
    p_final, cost_final = _local_search_2opt(a, b, p_final, cost_final, deadline)
    
    if cost_final < best_cost:
        best_p = p_final
        best_cost = cost_final

    return best_p.tolist()
