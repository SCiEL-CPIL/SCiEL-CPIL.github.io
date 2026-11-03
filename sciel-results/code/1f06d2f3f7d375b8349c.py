"""Combined solver for QAPLIB wil100: Multi-start Iterated Local Search with vectorized delta computation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] is the cost change if we swap positions i and j.
    Uses O(n^3) matrix multiplications, which is fast for n=100.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]

    # M1[i, j] = sum_k a[k, i] * Bp[k, j]
    M1 = a.T @ Bp
    # M2[i, j] = sum_k a[i, k] * Bp[k, j]
    M2 = a @ Bp

    diag_M1 = np.diag(M1)
    diag_M2 = np.diag(M2)
    diag_Bp = np.diag(Bp)
    a_diag = np.diag(a)

    # Raw terms (sum over all k)
    # T1[i,j] = sum_k (a[k,i] - a[k,j]) * (Bp[k,j] - Bp[k,i])
    # This expands to:
    # sum_k a[k,i]Bp[k,j] - sum_k a[k,i]Bp[k,i] - sum_k a[k,j]Bp[k,j] + sum_k a[k,j]Bp[k,i]
    # = M1[i,j] - M1[i,i] - M1[j,j] + M1[j,i]
    T1 = M1 - diag_M1[:, None] - diag_M1[None, :] + M1.T
    
    # T2[i,j] = sum_k (a[i,k] - a[j,k]) * (Bp[j,k] - Bp[i,k])
    # Note: Bp[j,k] = b[p[j], p[k]] and Bp[i,k] = b[p[i], p[k]]
    # sum_k a[i,k]Bp[j,k] - sum_k a[i,k]Bp[i,k] - sum_k a[j,k]Bp[j,k] + sum_k a[j,k]Bp[i,k]
    # = M2[i,j] - M2[i,i] - M2[j,j] + M2[j,i]  (Wait, M2[i,j] = sum_k a[i,k]Bp[k,j] = sum_k a[i,k]b[p[k],p[j]] = sum_k a[i,k]Bp[j,k] since Bp is symmetric? 
    # Actually, Bp[k,j] = b[p[k], p[j]]. If b is symmetric, b[p[k], p[j]] = b[p[j], p[k]] = Bp[j,k].
    # So M2[i,j] is indeed sum_k a[i,k]Bp[j,k].
    # So T2 = M2 - diag_M2[:, None] - diag_M2[None, :] + M2.T
    
    T2 = M2 - diag_M2[:, None] - diag_M2[None, :] + M2.T

    # Corrections for k=i and k=j in the sums above.
    # The sums included k=i and k=j. We need to subtract their contributions.
    # For T1:
    # k=i term: (a[i,i] - a[i,j]) * (Bp[i,j] - Bp[i,i])
    # k=j term: (a[j,i] - a[j,j]) * (Bp[j,j] - Bp[j,i])
    C1_i = (a_diag[:, None] - a) * (Bp - diag_Bp[:, None])
    C1_j = (a.T - a_diag[None, :]) * (diag_Bp[None, :] - Bp)
    # Note: C1_i[i,j] is the k=i term for swap (i,j). C1_j[i,j] is the k=j term.
    # Wait, the indices in C1_i and C1_j need to align with D[i,j].
    # C1_i[i,j] = (a[i,i] - a[i,j]) * (Bp[i,j] - Bp[i,i]). Correct.
    # C1_j[i,j] = (a[j,i] - a[j,j]) * (Bp[j,j] - Bp[j,i]). Correct.
    C1 = C1_i + C1_j

    # For T2:
    # k=i term: (a[i,i] - a[j,i]) * (Bp[j,i] - Bp[i,i])
    # k=j term: (a[i,j] - a[j,j]) * (Bp[j,j] - Bp[i,j])
    C2_i = (a_diag[:, None] - a.T) * (Bp.T - diag_Bp[:, None])
    C2_j = (a - a_diag[None, :]) * (diag_Bp[None, :] - Bp.T)
    C2 = C2_i + C2_j

    # T3: (a[i,j] - a[j,i]) * (Bp[j,i] - Bp[i,j])
    T3 = (a - a.T) * (Bp.T - Bp)

    D = (T1 - C1) + (T2 - C2) + T3
    np.fill_diagonal(D, 0)

    return D

def _local_search(a, b, p, deadline):
    """
    Perform steepest descent local search using the vectorized delta matrix.
    Returns (p, current_cost).
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)

    while time.time() < deadline:
        D = _compute_delta_matrix(a, b, p)
        best_delta = np.min(D)

        if best_delta >= 0:
            break

        best_idx = np.unravel_index(np.argmin(D), D.shape)
        i, j = best_idx
        p[i], p[j] = p[j], p[i]
        current_cost += int(best_delta)

    return p, current_cost

def _kick(p, kick_strength):
    """
    Apply a random kick by swapping 'kick_strength' random pairs.
    """
    n = len(p)
    for _ in range(kick_strength):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i != j:
            p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def compute_cost(p):
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    # Phase 1: Generate multiple diverse initial solutions
    initial_solutions = []
    
    # 1. Hungarian on linear surrogate (row sums)
    flow_cent = a.sum(axis=1)
    dist_cent = b.sum(axis=1)
    surrogate_cost = np.outer(flow_cent, dist_cent)
    row_ind, col_ind = linear_sum_assignment(surrogate_cost)
    p1 = np.empty(n, dtype=np.int64)
    p1[row_ind] = col_ind
    initial_solutions.append(p1)
    
    # 2. Hungarian on linear surrogate (column sums, different perspective)
    flow_cent_col = a.sum(axis=0)
    surrogate_cost2 = np.outer(flow_cent_col, dist_cent)
    row_ind2, col_ind2 = linear_sum_assignment(surrogate_cost2)
    p2 = np.empty(n, dtype=np.int64)
    p2[row_ind2] = col_ind2
    initial_solutions.append(p2)
    
    # 3. Hungarian with noise to break ties
    np.random.seed(42)
    noise = np.random.normal(0, 0.01 * np.max(surrogate_cost), surrogate_cost.shape)
    surrogate_cost3 = surrogate_cost + noise
    row_ind3, col_ind3 = linear_sum_assignment(surrogate_cost3)
    p3 = np.empty(n, dtype=np.int64)
    p3[row_ind3] = col_ind3
    initial_solutions.append(p3)

    # 4. Random permutations
    for _ in range(3):
        initial_solutions.append(np.random.permutation(n).astype(np.int64))

    best_p = initial_solutions[0].copy()
    best_cost = compute_cost(best_p)

    # Phase 2: Iterated Local Search (ILS)
    # We will run local search on each initial solution, then iterate with kicks.
    # To manage time, we allocate a chunk of time to each "thread" of ILS.
    # However, simple sequential ILS on the best solution is often more effective.
    # Let's first refine all initial solutions with a quick local search.
    
    refined_solutions = []
    time_per_initial = (deadline - start_time - 20.0) / len(initial_solutions)
    
    for p_init in initial_solutions:
        if time.time() > deadline - 20.0:
            break
        p_ref, cost_ref = _local_search(a, b, p_init.copy(), time.time() + time_per_initial)
        if cost_ref < best_cost:
            best_cost = cost_ref
            best_p = p_ref.copy()
        refined_solutions.append((p_ref, cost_ref))

    # Sort refined solutions by cost
    refined_solutions.sort(key=lambda x: x[1])
    
    # Phase 3: ILS on the best refined solution
    # Use the remaining time for ILS on the current best.
    p_current = best_p.copy()
    cost_current = best_cost
    
    # We can also try ILS on the 2nd best if time permits, but let's stick to the best for now
    # to maximize depth of search.
    
    while time.time() < deadline - 5.0:
        # Apply a kick
        # Kick strength can vary, but 2-3 swaps is standard for n=100
        p_kicked = _kick(p_current, 3)
        
        # Run local search from the kicked solution
        p_new, cost_new = _local_search(a, b, p_kicked, time.time() + 5.0) # Limit time per LS run
        
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new.copy()
            p_current = p_new.copy()
            cost_current = cost_new
        else:
            # If no improvement, we might still want to move to the new local optimum
            # if it's not too much worse, to explore other basins?
            # Standard ILS accepts if better. Some variants accept with probability.
            # For QAP, sticking to the best found is safer.
            # However, if we stay in the same basin, we are stuck.
            # Let's accept if it's close, or just keep the best.
            # Keeping the best is standard for this type of solver.
            pass

    # Phase 4: Final Polish
    # One last steepest descent on the global best
    if time.time() < deadline - 1.0:
        p_final, cost_final = _local_search(a, b, best_p.copy(), deadline)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
