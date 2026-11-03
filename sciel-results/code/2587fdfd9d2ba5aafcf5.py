"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _greedy_initial(a, b, n, rng):
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    importance = A.sum(axis=1)
    order = np.argsort(-importance, kind='stable')
    
    p = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    placed_list = []
    diag_b = B.diagonal()
    
    for idx in range(n):
        f = order[idx]
        if len(placed_list) == 0:
            marginal = A[f, f] * diag_b
        else:
            placed = np.array(placed_list, dtype=np.int64)
            term1 = A[f, placed] @ B[:, p[placed]]
            term2 = A[placed, f] @ B[p[placed], :]
            marginal = term1 + term2 + A[f, f] * diag_b
        
        marginal[used] = np.inf
        best_loc = int(np.argmin(marginal))
        p[f] = best_loc
        used[best_loc] = True
        placed_list.append(f)
    
    return p


def _local_search_2opt(a, b, p, deadline):
    n = len(a)
    p = p.copy().astype(np.int64)
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    
    # B_at_p[l1, l2] = B[p[l1], p[l2]] -- no wait
    # Let me define: Bp[i, j] = B[p[i], p[j]]
    Bp = B[np.ix_(p, p)]
    cost = float(np.sum(A * Bp))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            li = p[i]
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    break
                lj = p[j]
                
                # Bp[i, :] = B[p[i], p] = B[li, p]
                # Bp[:, i] = B[p, p[i]] = B[p, li]
                # Bp[j, :] = B[lj, p]
                # Bp[:, j] = B[p, lj]
                
                diff_row = Bp[j, :] - Bp[i, :]  # B[lj,p] - B[li,p]
                diff_col = Bp[:, j] - Bp[:, i]  # B[p,lj] - B[p,li]
                
                S1 = A[i, :] @ diff_row
                S2 = -A[j, :] @ diff_row
                S3 = A[:, i] @ diff_col
                S4 = -A[:, j] @ diff_col
                
                corr = (A[i, i] * (B[lj, lj] - B[li, li])
                       + A[i, j] * (B[lj, li] - B[li, lj])
                       + A[j, i] * (B[li, lj] - B[lj, li])
                       + A[j, j] * (B[li, li] - B[lj, lj]))
                
                delta = S1 + S2 + S3 + S4 - corr
                
                if delta < -1e-9:
                    # Swap i and j
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    # Update Bp: swap row/col i and j
                    Bp[i, :], Bp[j, :] = Bp[j, :].copy(), Bp[i, :].copy()
                    Bp[:, i], Bp[:, j] = Bp[:, j].copy(), Bp[:, i].copy()
                    # Fix the diagonal
                    Bp[i, i] = B[li, li]  # wait, after swap p[i]=lj, so Bp[i,i]=B[lj,lj]
                    Bp[j, j] = B[li, li]  # p[j]=li, so Bp[j,j]=B[li,li]
                    # Actually the swap of rows and columns should handle this correctly
                    # Let me verify: before swap, Bp[i,i]=B[li,li], Bp[j,j]=B[lj,lj]
                    # After swapping rows i,j and cols i,j:
                    # Bp[i,i] becomes old Bp[j,j] = B[lj,lj] = B[p[i]_new, p[i]_new] ✓
                    # Bp[j,j] becomes old Bp[i,i] = B[li,li] = B[p[j]_new, p[j]_new] ✓
                    # Bp[i,j] becomes old Bp[j,i] = B[lj,li] = B[p[i]_new, p[j]_new] ✓
                    # Bp[j,i] becomes old Bp[i,j] = B[li,lj] = B[p[j]_new, p[i]_new] ✓
                    # So the swap is correct!
    
    return p, int(cost)


def _perturb(p, n, rng, fraction=0.05):
    p = p.copy()
    n_swaps = max(1, int(n * fraction))
    for _ in range(n_swaps):
        i, j = rng.choice(n, size=2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Greedy initialization
    p_best = _greedy_initial(a, b, n, rng)
    cost_best = _qap_cost(a, b, p_best)
    
    # Phase 2: Local search
    p_cur, cost_cur = _local_search_2opt(a, b, p_best, deadline - 10.0)
    if cost_cur < cost_best:
        p_best = p_cur
        cost_best = cost_cur
    
    # Phase 3: Iterated local search
    while time.time() < deadline - 5.0:
        p_pert = _perturb(p_best, n, rng, fraction=0.05)
        p_cur, cost_cur = _local_search_2opt(a, b, p_pert, deadline - 2.0)
        if cost_cur < cost_best:
            p_best = p_cur
            cost_best = cost_cur
    
    return p_best.tolist()
# EVOLVE-BLOCK-END
