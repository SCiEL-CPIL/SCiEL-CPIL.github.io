"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    # --- Greedy construction: place facilities one by one, choosing the location
    #     that minimizes the incremental cost given already-placed facilities.
    #     Order facilities by descending "importance" (sum of flows).
    order = np.argsort(-a.sum(axis=1), kind="stable")
    p = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    
    # Precompute for speed: a_sum[i] = sum of a[i,:], a_col_sum[i] = sum of a[:,i]
    a_row_sum = a.sum(axis=1)
    a_col_sum = a.sum(axis=0)
    
    for idx in range(n):
        f = order[idx]
        best_loc = -1
        best_cost = np.iinfo(np.int64).max
        for loc in range(n):
            if used[loc]:
                continue
            # Incremental cost of placing f at loc given already placed
            # Sum over already placed k: a[f,k]*b[loc,p[k]] + a[k,f]*b[p[k],loc]
            inc = 0
            for k in range(idx):
                g = order[k]
                inc += a[f, g] * b[loc, p[g]] + a[g, f] * b[p[g], loc]
            if inc < best_cost:
                best_cost = inc
                best_loc = loc
        p[f] = best_loc
        used[best_loc] = True

    # --- Compute current total cost
    def total_cost(p):
        total = 0
        for i in range(n):
            for j in range(n):
                total += a[i, j] * b[p[i], p[j]]
        return total

    current_cost = total_cost(p)

    # --- Tabu Search
    # Precompute some helper arrays for fast swap delta computation
    # For swap of i,j: delta = sum over k!=i,j of terms + direct i-j terms
    
    # We'll use a simple tabu search with random restarts if time permits
    t0 = time.time()
    time_limit = 110.0  # leave 10s margin
    
    tabu_len = max(10, n // 5)
    tabu = [0] * (n * n)  # tabu[i*n+j] = iteration when (i,j) becomes non-tabu
    tabu_set = {}  # (i,j) -> iteration
    
    best_p = p.copy()
    best_cost = current_cost
    
    iteration = 0
    no_improve = 0
    max_no_improve = 2000
    
    # Precompute b rows as lists for faster access
    b_rows = [b[i] for i in range(n)]
    a_rows = [a[i] for i in range(n)]
    a_cols = [a[:, i] for i in range(n)]
    
    while time.time() - t0 < time_limit:
        # Evaluate all possible swaps, pick best non-tabu
        best_delta = np.iinfo(np.int64).max
        best_i, best_j = -1, -1
        
        for i in range(n):
            pi = p[i]
            for j in range(i + 1, n):
                pj = p[j]
                
                # Check tabu
                key = i * n + j
                if tabu_set.get(key, -1) >= iteration:
                    continue
                
                # Compute delta for swapping i and j
                # Direct terms: a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj] - a[i,j]*b[pi,pj] - a[j,i]*b[pj,pi]
                delta = a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])
                
                # Sum over k != i, j
                for k in range(n):
                    if k == i or k == j:
                        continue
                    pk = p[k]
                    # Terms involving i and k:
                    # a[i,k]*(b[pj,pk] - b[pi,pk]) + a[k,i]*(b[pk,pj] - b[pk,pi])
                    # Terms involving j and k:
                    # a[j,k]*(b[pi,pk] - b[pj,pk]) + a[k,j]*(b[pk,pi] - b[pk,pj])
                    delta += a[i, k] * (b[pj, pk] - b[pi, pk]) + a[k, i] * (b[pk, pj] - b[pk, pi])
                    delta += a[j, k] * (b[pi, pk] - b[pj, pk]) + a[k, j] * (b[pk, pi] - b[pk, pj])
                
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        
        if best_i == -1:
            # All moves are tabu; pick the best tabu move (aspiration: allow if it improves best)
            for i in range(n):
                pi = p[i]
                for j in range(i + 1, n):
                    pj = p[j]
                    delta = a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        pk = p[k]
                        delta += a[i, k] * (b[pj, pk] - b[pi, pk]) + a[k, i] * (b[pk, pj] - b[pk, pi])
                        delta += a[j, k] * (b[pi, pk] - b[pj, pk]) + a[k, j] * (b[pk, pi] - b[pk, pj])
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j
        
        if best_i == -1:
            break
        
        # Perform the swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
        # Update tabu
        key = best_i * n + best_j
        tabu_set[key] = iteration + tabu_len
        # Also reverse
        key2 = best_j * n + best_i
        tabu_set[key2] = iteration + tabu_len
        
        iteration += 1
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > max_no_improve:
            # Perturb: do a few random swaps
            for _ in range(5):
                ri = np.random.randint(n)
                rj = np.random.randint(n)
                if ri != rj:
                    p[ri], p[rj] = p[rj], p[ri]
            current_cost = total_cost(p)
            no_improve = 0
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
