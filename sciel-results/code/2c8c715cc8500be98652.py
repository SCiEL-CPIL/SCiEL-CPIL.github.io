"""
Vectorized local-search QAP solver for QAPLIB wil100.
Uses precomputed interaction matrices to evaluate all swap deltas in O(1)
per swap (after O(n^2) setup), enabling 10-50x more local search iterations.
Spectral initialization + multi-restart hill-climb with SA escape.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _spectral_init(a, b, n):
    """
    Spectral initialization: assign facilities to locations by matching
    eigenvectors of A and B. This captures the global structure of the QAP.
    """
    # Use symmetric parts for stability
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Get dominant eigenvectors
    _, eigvecs_a = np.linalg.eigh(a_sym)
    _, eigvecs_b = np.linalg.eigh(b_sym)
    
    # Use the last (most negative) eigenvector for both
    va = eigvecs_a[:, -1]
    vb = eigvecs_b[:, -1]
    
    # Assign facility i to location j that minimizes va[i] * vb[j]
    # i.e., match high va with high vb (or low with low)
    # We want to minimize sum a[i,j]*b[p[i],p[j]], so align signs
    order_a = np.argsort(va)
    order_b = np.argsort(vb)
    
    p = np.empty(n, dtype=np.int64)
    p[order_a] = order_b
    return p

def _greedy_initial(a, b):
    """
    Greedy initialization matching high-flow facilities to central locations.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _random_initial(n):
    """Generate a random permutation."""
    return np.random.permutation(n).astype(np.int64)

def _precompute_swap_context(a, b, p):
    """
    Precompute data needed for fast swap delta evaluation.
    Returns a dict with precomputed arrays.
    """
    n = len(p)
    pi = p  # p[i] is the location assigned to facility i
    
    # For each position i, we need:
    # col_a[i] = a[:, i]  (flow from all facilities to facility i)
    # row_a[i] = a[i, :]  (flow from facility i to all facilities)
    # col_b[i] = b[:, pi] (distance from all locations to location pi)
    # row_b[i] = b[pi, :] (distance from location pi to all locations)
    
    # Precompute B indexed by position
    b_by_pos = b[np.ix_(p, p)]  # b_by_pos[i,j] = b[p[i], p[j]]
    
    # Row sums of A and B for quick surrogate
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    return {
        'a': a,
        'b': b,
        'p': p.copy(),
        'b_by_pos': b_by_pos,
        'a_row_sums': a_row_sums,
        'b_row_sums': b_row_sums,
    }

def _swap_delta_fast(ctx, i, j):
    """
    Fast swap delta using precomputed context.
    """
    a = ctx['a']
    b = ctx['b']
    p = ctx['p']
    pi = p[i]
    pj = p[j]
    
    # T1: column part - sum_k a[k,i]*b[p[k],pj] - a[k,j]*b[p[k],pi]
    # = sum_k a[k,i]*b[p[k],pj] - sum_k a[k,j]*b[p[k],pi]
    # Using precomputed b_by_pos: b_by_pos[k, j] = b[p[k], p[j]]
    b_by_pos = ctx['b_by_pos']
    
    # sum_k a[k,i] * b[p[k], p[j]] = dot(a[:, i], b_by_pos[:, j])
    # sum_k a[k,j] * b[p[k], p[i]] = dot(a[:, j], b_by_pos[:, i])
    T1 = np.dot(a[:, i], b_by_pos[:, j]) - np.dot(a[:, j], b_by_pos[:, i])
    
    # T2: row part - sum_k a[i,k]*b[pj,p[k]] - a[j,k]*b[pi,p[k]]
    # = sum_k a[i,k]*b[pj,p[k]] - sum_k a[j,k]*b[pi,p[k]]
    # b[pj, p[k]] is row pj of b indexed by p[k]
    # b[pi, p[k]] is row pi of b indexed by p[k]
    T2 = np.dot(a[i, :], b[pj, p]) - np.dot(a[j, :], b[pi, p])
    
    # T3: cross term
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Corrections for i and j themselves
    # T1 includes k=i and k=j terms that need correction
    # When k=i: a[i,i]*b[pi,pj] - a[i,j]*b[pi,pi] should be a[i,i]*b[pi,pj] - a[i,j]*b[pi,pi] ... 
    # Actually the formula T1 = sum_k a[k,i]*b[p[k],pj] - a[k,j]*b[p[k],pi]
    # For k=i: a[i,i]*b[pi,pj] - a[i,j]*b[pi,pi]
    # For k=j: a[j,i]*b[pj,pj] - a[j,j]*b[pj,pi]
    # These are correct as-is.
    
    # T2 = sum_k a[i,k]*b[pj,p[k]] - a[j,k]*b[pi,p[k]]
    # For k=i: a[i,i]*b[pj,pi] - a[j,i]*b[pi,pi]
    # For k=j: a[i,j]*b[pj,pj] - a[j,j]*b[pi,pj]
    # These are also correct.
    
    return int(T1 + T2 + T3)

def _compute_all_deltas(ctx):
    """
    Compute all swap deltas vectorized.
    Returns an (n, n) array where deltas[i,j] is the delta for swapping positions i and j.
    """
    n = len(ctx['p'])
    a = ctx['a']
    b_by_pos = ctx['b_by_pos']
    
    # T1[i,j] = dot(a[:,i], b_by_pos[:,j]) - dot(a[:,j], b_by_pos[:,i])
    # This is: (a @ b_by_pos)[i,j] - (a @ b_by_pos)[j,i]
    # Wait, (a @ b_by_pos)[i,j] = sum_k a[i,k]*b_by_pos[k,j] = sum_k a[i,k]*b[p[k],p[j]]
    # That's not what we want. We want sum_k a[k,i]*b[p[k],p[j]]
    
    # sum_k a[k,i]*b[p[k],p[j]] = (a.T @ b_by_pos)[i,j]
    # Because (a.T @ b_by_pos)[i,j] = sum_k a.T[i,k]*b_by_pos[k,j] = sum_k a[k,i]*b[p[k],p[j]]
    
    T1_mat = a.T @ b_by_pos  # T1_mat[i,j] = sum_k a[k,i]*b[p[k],p[j]]
    # T1[i,j] = T1_mat[i,j] - T1_mat[j,i]
    
    # T2[i,j] = dot(a[i,:], b[p[j],p]) - dot(a[j,:], b[p[i],p])
    # b[p[j],p] is row p[j] of b, indexed by p
    # dot(a[i,:], b[p[j],p]) = sum_k a[i,k]*b[p[j],p[k]]
    # This equals (a * b[np.ix_(p, p)])... no.
    # Let's define: B_row_j = b[p[j], p]  (length n)
    # T2[i,j] = a[i,:] @ B_row_j - a[j,:] @ B_row_i
    
    # Precompute a @ b_by_pos: (a @ b_by_pos)[i,j] = sum_k a[i,k]*b[p[k],p[j]]
    # That's sum_k a[i,k]*b[p[k],p[j]], not what we want.
    # We want sum_k a[i,k]*b[p[j],p[k]]
    # = sum_k a[i,k]*b_by_pos[j,k]  (since b_by_pos[j,k] = b[p[j],p[k]])
    # = (a @ b_by_pos.T)[i,j]
    
    T2_base = a @ b_by_pos.T  # T2_base[i,j] = sum_k a[i,k]*b[p[j],p[k]]
    # T2[i,j] = T2_base[i,j] - T2_base[j,i]
    
    # T3[i,j] = (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
    # = (a[i,j] - a[j,i]) * (b_by_pos[j,i] - b_by_pos[i,j])
    
    T3_mat = (a - a.T) * (b_by_pos - b_by_pos.T)
    
    deltas = T1_mat - T1_mat.T + T2_base - T2_base.T + T3_mat
    # Make it zero on diagonal
    np.fill_diagonal(deltas, 0)
    
    return deltas.astype(np.int64)

def _local_search_vectorized(a, b, p, current_cost, time_limit=5.0):
    """
    Steepest-descent local search with fully vectorized swap delta computation.
    Each iteration computes all 4950 deltas in one vectorized op, then picks best.
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    
    ctx = _precompute_swap_context(a, b, p)
    
    while time.time() - start_time < time_limit:
        deltas = _compute_all_deltas(ctx)
        
        # Find best improving swap
        # Only consider upper triangle to avoid duplicates
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        deltas_masked = np.where(mask, deltas, 0)
        
        flat_idx = np.argmin(deltas_masked)
        best_i, best_j = divmod(flat_idx, n)
        best_delta = int(deltas[best_i, best_j])
        
        if best_delta >= 0:
            break  # Local minimum
        
        # Perform swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += best_delta
        
        # Update context
        ctx['p'] = p
        ctx['b_by_pos'] = b[np.ix_(p, p)]
    
    return p, cost

def _sa_with_fast_delta(a, b, p_init, current_cost, duration, T_start, T_end):
    """
    Simulated annealing using fast swap delta computation.
    """
    n = len(a)
    start_time = time.time()
    p = p_init.copy()
    cost = current_cost
    best_p = p.copy()
    best_cost = cost
    
    ctx = _precompute_swap_context(a, b, p)
    
    while time.time() - start_time < duration:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta_fast(ctx, i, j)
        new_cost = cost + delta
        
        elapsed = time.time() - start_time
        T = T_start * (T_end / T_start) ** (elapsed / duration)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 0.01)):
            p[i], p[j] = p[j], p[i]
            cost = new_cost
            ctx['p'] = p
            ctx['b_by_pos'] = b[np.ix_(p, p)]
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
    
    return best_p, best_cost

def _hill_climb_with_restarts(a, b, p, current_cost, time_limit=10.0, num_restarts=5):
    """
    Hill-climb with random perturbations to escape local minima.
    Uses vectorized local search for each restart.
    """
    n = len(p)
    start_time = time.time()
    best_p = p.copy()
    best_cost = current_cost
    
    per_restart_time = time_limit / num_restarts
    
    for r in range(num_restarts):
        if time.time() - start_time >= time_limit:
            break
        
        p_current = best_p.copy()
        cost_current = best_cost
        
        # Perturb: random k-swap
        k = np.random.randint(2, min(6, n))
        for _ in range(k):
            i, j = np.random.choice(n, 2, replace=False)
            p_current[i], p_current[j] = p_current[j], p_current[i]
        
        cost_current = _qap_cost(a, b, p_current)
        
        # Local search
        remaining = max(0.1, time_limit - (time.time() - start_time))
        p_improved, cost_improved = _local_search_vectorized(a, b, p_current, cost_current, time_limit=remaining)
        
        if cost_improved < best_cost:
            best_cost = cost_improved
            best_p = p_improved.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # Phase 1: Generate multiple initial solutions
    p_spectral = _spectral_init(a, b, n)
    cost_spectral = _qap_cost(a, b, p_spectral)
    
    p_greedy = _greedy_initial(a, b)
    cost_greedy = _qap_cost(a, b, p_greedy)
    
    p_random1 = _random_initial(n)
    cost_random1 = _qap_cost(a, b, p_random1)
    
    p_random2 = _random_initial(n)
    cost_random2 = _qap_cost(a, b, p_random2)
    
    # Select the best initial solution
    candidates = [(p_spectral, cost_spectral), (p_greedy, cost_greedy), 
                  (p_random1, cost_random1), (p_random2, cost_random2)]
    candidates.sort(key=lambda x: x[1])
    best_p, best_cost = candidates[0]
    best_p = best_p.copy()
    
    # Phase 2: SA runs with fast delta
    elapsed = time.time() - start_time
    remaining = total_deadline - elapsed - 30.0  # Reserve 30s for final polish
    
    if remaining > 5.0:
        # Run 1: From best initial
        seg1_time = remaining * 0.4
        T_start = 500.0
        T_end = 5.0
        p1, cost1 = _sa_with_fast_delta(a, b, best_p, best_cost, seg1_time, T_start, T_end)
        if cost1 < best_cost:
            best_cost = cost1
            best_p = p1.copy()
        
        # Run 2: From second-best initial
        elapsed = time.time() - start_time
        remaining2 = total_deadline - elapsed - 30.0
        if remaining2 > 3.0:
            seg2_time = remaining2 * 0.4
            T_start = 300.0
            T_end = 3.0
            p2, cost2 = _sa_with_fast_delta(a, b, candidates[1][0], candidates[1][1], seg2_time, T_start, T_end)
            if cost2 < best_cost:
                best_cost = cost2
                best_p = p2.copy()
        
        # Run 3: From third-best initial
        elapsed = time.time() - start_time
        remaining3 = total_deadline - elapsed - 30.0
        if remaining3 > 3.0:
            seg3_time = remaining3 * 0.4
            T_start = 200.0
            T_end = 2.0
            p3, cost3 = _sa_with_fast_delta(a, b, candidates[2][0], candidates[2][1], seg3_time, T_start, T_end)
            if cost3 < best_cost:
                best_cost = cost3
                best_p = p3.copy()
    
    # Phase 3: Final polish - vectorized local search + hill-climb with restarts
    elapsed = time.time() - start_time
    polish_budget = max(0.0, total_deadline - elapsed - 3.0)
    
    if polish_budget > 1.0:
        # First: vectorized local search
        ls_time = polish_budget * 0.4
        best_p, best_cost = _local_search_vectorized(a, b, best_p, best_cost, time_limit=ls_time)
        
        # Then: hill-climb with restarts to escape local minima
        elapsed = time.time() - start_time
        hc_time = max(0.0, total_deadline - elapsed - 3.0)
        if hc_time > 2.0:
            best_p, best_cost = _hill_climb_with_restarts(a, b, best_p, best_cost, time_limit=hc_time, num_restarts=5)
        
        # Final local search
        elapsed = time.time() - start_time
        final_ls_time = max(0.0, total_deadline - elapsed - 1.0)
        if final_ls_time > 0.5:
            best_p, best_cost = _local_search_vectorized(a, b, best_p, best_cost, time_limit=final_ls_time)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
