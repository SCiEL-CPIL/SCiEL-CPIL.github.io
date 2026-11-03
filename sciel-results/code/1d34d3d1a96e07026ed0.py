"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """Compute exact delta for swapping facilities at positions i and j."""
    pi = p[i]
    pj = p[j]
    n = len(p)
    
    # Vectorized computation of the delta
    # For each k != i, j:
    #   a[i,k]*(b[pj,pk] - b[pi,pk]) + a[j,k]*(b[pi,pk] - b[pj,pk])
    #   + a[k,i]*(b[pk,pj] - b[pk,pi]) + a[k,j]*(b[pk,pi] - b[pk,pj])
    # Plus direct i-j terms:
    #   a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
    
    # Let D_row = b[pj, :] - b[pi, :]  (shape n)
    # Let D_col = b[:, pj] - b[:, pi]  (shape n)
    
    D_row = b[pj, :] - b[pi, :]
    D_col = b[:, pj] - b[:, pi]
    
    # Sum over all k of a[i,k]*D_row[k] = a[i,:] @ D_row
    s1 = int(np.dot(a[i, :], D_row))
    # Sum over all k of a[j,k]*(-D_row[k]) = -a[j,:] @ D_row
    s2 = -int(np.dot(a[j, :], D_row))
    # Sum over all k of a[k,i]*D_col[k] = a[:,i] @ D_col
    s3 = int(np.dot(a[:, i], D_col))
    # Sum over all k of a[k,j]*(-D_col[k]) = -a[:,j] @ D_col
    s4 = -int(np.dot(a[:, j], D_col))
    
    # The above sums include k=i and k=j terms which need correction.
    # Let's compute the "clean" delta by excluding i and j from the sums,
    # then adding the direct i-j interaction change.
    
    # For s1 (i's row): exclude k=i and k=j
    # k=i term: a[i,i]*D_row[i] = a[i,i]*(b[pj,pi]-b[pi,pi])
    # k=j term: a[i,j]*D_row[j] = a[i,j]*(b[pj,pj]-b[pi,pj])
    s1_clean = s1 - int(a[i, i] * (b[pj, pi] - b[pi, pi])) - int(a[i, j] * (b[pj, pj] - b[pi, pj]))
    
    # For s2 (j's row): exclude k=i and k=j
    # k=i term: -a[j,i]*D_row[i] = -a[j,i]*(b[pj,pi]-b[pi,pi])
    # k=j term: -a[j,j]*D_row[j] = -a[j,j]*(b[pj,pj]-b[pi,pj])
    s2_clean = s2 - (-int(a[j, i] * (b[pj, pi] - b[pi, pi]))) - (-int(a[j, j] * (b[pj, pj] - b[pi, pj])))
    # = s2 + a[j,i]*(b[pj,pi]-b[pi,pi]) + a[j,j]*(b[pj,pj]-b[pi,pj])
    s2_clean = s2 + int(a[j, i] * (b[pj, pi] - b[pi, pi])) + int(a[j, j] * (b[pj, pj] - b[pi, pj]))
    
    # For s3 (i's col): exclude k=i and k=j
    # k=i term: a[i,i]*D_col[i] = a[i,i]*(b[pi,pj]-b[pi,pi])
    # k=j term: a[j,i]*D_col[j] = a[j,i]*(b[pj,pj]-b[pj,pi])
    s3_clean = s3 - int(a[i, i] * (b[pi, pj] - b[pi, pi])) - int(a[j, i] * (b[pj, pj] - b[pj, pi]))
    
    # For s4 (j's col): exclude k=i and k=j
    # k=i term: -a[i,j]*D_col[i] = -a[i,j]*(b[pi,pj]-b[pi,pi])
    # k=j term: -a[j,j]*D_col[j] = -a[j,j]*(b[pj,pj]-b[pj,pi])
    s4_clean = s4 - (-int(a[i, j] * (b[pi, pj] - b[pi, pi]))) - (-int(a[j, j] * (b[pj, pj] - b[pj, pi])))
    # = s4 + a[i,j]*(b[pi,pj]-b[pi,pi]) + a[j,j]*(b[pj,pj]-b[pj,pi])
    s4_clean = s4 + int(a[i, j] * (b[pi, pj] - b[pi, pi])) + int(a[j, j] * (b[pj, pj] - b[pj, pi]))
    
    # Direct i-j interaction change:
    # Original: a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
    # New:      a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
    direct = int(a[i, j] * (b[pj, pi] - b[pi, pj])) + int(a[j, i] * (b[pi, pj] - b[pj, pi]))
    
    delta = s1_clean + s2_clean + s3_clean + s4_clean + direct
    return delta


def _local_search(a, b, p, deadline, best_cost, best_p):
    """First-improvement 2-opt local search."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                return best_cost, best_p
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    cur += d
                    improved = True
                    if cur < best_cost:
                        best_cost = cur
                        best_p = p.copy()
                    break
        if not improved:
            break
    return best_cost, best_p


def _lap_perturbation(a, b, p, rng):
    """
    LAP-guided perturbation: build a linear surrogate cost matrix from the
    current assignment's structure, solve the assignment problem, and use
    the resulting permutation as a perturbation.
    """
    n = len(p)
    # Build surrogate: c[i, loc] = sum_j a[i,j] * b[loc, p[j]] + sum_j a[j,i] * b[p[j], loc]
    # This is the "linearization" of the QAP around the current assignment.
    # c[i, l] = sum_j a[i,j]*b[l, p[j]] + sum_j a[j,i]*b[p[j], l]
    
    # Precompute for speed:
    # B_p = b[:, p]  (n x n)
    # B_pT = b[p, :] (n x n)
    B_p = b[:, p]  # B_p[l, j] = b[l, p[j]]
    B_pT = b[p, :]  # B_pT[j, l] = b[p[j], l]
    
    # c[i, l] = sum_j a[i,j]*B_p[l,j] + sum_j a[j,i]*B_pT[j,l]
    # = (a @ B_p.T)[i, l] + (a.T @ B_pT)[i, l]
    # B_p.T has shape (n, n): B_p.T[j, l] = B_p[l, j] = b[l, p[j]]
    # So a @ B_p.T gives [i, l] = sum_j a[i,j]*b[l, p[j]]
    # B_pT has shape (n, n): B_pT[j, l] = b[p[j], l]
    # a.T @ B_pT gives [i, l] = sum_j a[j,i]*b[p[j], l]
    
    c = (a @ B_p.T + a.T @ B_pT).astype(np.float64)
    
    # Add small random noise to break ties and diversify
    c += rng.random((n, n)) * 0.01 * np.max(np.abs(c))
    
    row_ind, col_ind = linear_sum_assignment(c)
    new_p = np.zeros(n, dtype=np.int64)
    new_p[row_ind] = col_ind
    return new_p


def _anneal(a, b, p, deadline, best_cost, best_p, rng, t_start=200.0, t_end=1.0, alpha=0.995):
    """Simulated annealing with LAP-guided restarts."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    t = t_start
    no_improve_steps = 0
    lap_interval = 500  # try LAP perturbation every 500 steps if no improvement
    
    while time.time() < deadline and t > t_end:
        i = int(rng.integers(n))
        j = int(rng.integers(n))
        if i == j:
            continue
        
        d = _swap_delta(a, b, p, i, j)
        
        if d < 0 or rng.random() < np.exp(-d / max(t, 1e-10)):
            p[i], p[j] = p[j], p[i]
            cur += d
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1
        
        # LAP-guided perturbation to escape local minima
        if no_improve_steps >= lap_interval:
            no_improve_steps = 0
            new_p = _lap_perturbation(a, b, p, rng)
            new_cost = _qap_cost(a, b, new_p)
            # Accept if better or with SA probability
            if new_cost < cur or rng.random() < np.exp(-(new_cost - cur) / max(t, 1e-10)):
                p = new_p.copy()
                cur = new_cost
                if cur < best_cost:
                    best_cost = cur
                    best_p = p.copy()
        
        t *= alpha
    
    return best_cost, best_p


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 150s kill

    rng = np.random.default_rng(42)
    
    best_cost = np.iinfo(np.int64).max
    best_p = None
    
    # --- Constructive starts ---
    
    # Strategy 1: rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    c1 = _qap_cost(a, b, p1)
    if c1 < best_cost:
        best_cost = c1
        best_p = p1.copy()
    
    # Strategy 2: random
    p2 = rng.permutation(n).astype(np.int64)
    c2 = _qap_cost(a, b, p2)
    if c2 < best_cost:
        best_cost = c2
        best_p = p2.copy()
    
    # Strategy 3: LAP-based initial solution
    # Build surrogate from zero assignment (all at position 0)
    # c[i, l] = sum_j a[i,j]*b[l,j] + sum_j a[j,i]*b[j,l]
    c_init = (a @ b.T + a.T @ b).astype(np.float64)
    row_ind, col_ind = linear_sum_assignment(c_init)
    p3 = np.zeros(n, dtype=np.int64)
    p3[row_ind] = col_ind
    c3 = _qap_cost(a, b, p3)
    if c3 < best_cost:
        best_cost = c3
        best_p = p3.copy()
    
    # --- Local search from best start ---
    p = best_p.copy()
    best_cost, best_p = _local_search(a, b, p, deadline, best_cost, best_p)
    
    # --- Simulated annealing with LAP-guided perturbations ---
    best_cost, best_p = _anneal(a, b, p, deadline, best_cost, best_p, rng)
    
    # --- Final local search ---
    p = best_p.copy()
    best_cost, best_p = _local_search(a, b, p, deadline, best_cost, best_p)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
