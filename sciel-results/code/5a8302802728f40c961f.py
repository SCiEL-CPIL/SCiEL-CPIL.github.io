"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _compute_cost(a, b, p):
    """Compute the QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    # p is a permutation array of length n
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Efficiently: let Bp = b[p[:, None], p[None, :]]
    # cost = np.sum(a * Bp)
    Bp = b[p[:, None], p[None, :]]
    return int(np.sum(a * Bp))


def _k_optimize(a, b, p, k, time_limit, rng):
    """
    K-opt local search: try moving up to k elements at a time.
    For k=1: try moving each element to every other position.
    For k=2: try swapping pairs.
    Returns (improved_p, best_cost).
    """
    n = len(p)
    p = np.array(p, dtype=np.int64)
    best_cost = _compute_cost(a, b, p)
    
    if k == 1:
        # 1-opt: try moving element at position i to position j
        # This is expensive for n=150, so we do limited passes
        deadline = time.time() + time_limit
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                orig_val = p[i]
                for j in range(n):
                    if i == j:
                        continue
                    # Move p[i] to position j (shifting elements in between)
                    # This is O(n) per move, too slow. Use swap-based instead.
                    # Let's use swap-based 1-opt (which is really 2-opt swap)
                    pass
                break  # Only do swap-based for efficiency
        # Switch to swap-based
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    # Swap p[i] and p[j]
                    p[i], p[j] = p[j], p[i]
                    new_cost = _compute_cost(a, b, p)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        improved = True
                    else:
                        p[i], p[j] = p[j], p[i]
        return p.tolist(), best_cost
    else:
        # For k>1, just do swap-based (2-opt)
        deadline = time.time() + time_limit
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    p[i], p[j] = p[j], p[i]
                    new_cost = _compute_cost(a, b, p)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        improved = True
                    else:
                        p[i], p[j] = p[j], p[i]
        return p.tolist(), best_cost


def _fast_swap_optimize(a, b, p, time_limit):
    """
    Faster swap-based local search using incremental cost updates.
    For a swap of positions i and j (values x=p[i], y=p[j]):
    The change in cost can be computed without full recomputation.
    
    cost = sum_{u,v} a[u,v] * b[p[u], p[v]]
    
    When we swap p[i] and p[j] (x and y), the terms that change are those
    where u or v is i or j.
    
    Let's derive:
    Old contribution involving i or j:
    - For u=i, v=i: a[i,i]*b[x,x]
    - For u=i, v=j: a[i,j]*b[x,y]
    - For u=j, v=i: a[j,i]*b[y,x]
    - For u=j, v=j: a[j,j]*b[y,y]
    - For u=i, v=k (k!=i,j): a[i,k]*b[x, p[k]]
    - For u=k, v=i (k!=i,j): a[k,i]*b[p[k], x]
    - For u=j, v=k (k!=i,j): a[j,k]*b[y, p[k]]
    - For u=k, v=j (k!=i,j): a[k,j]*b[p[k], y]
    
    After swap (x<->y):
    - For u=i, v=i: a[i,i]*b[y,y]
    - For u=i, v=j: a[i,j]*b[y,x]
    - For u=j, v=i: a[j,i]*b[x,y]
    - For u=j, v=j: a[j,j]*b[x,x]
    - For u=i, v=k: a[i,k]*b[y, p[k]]
    - For u=k, v=i: a[k,i]*b[p[k], y]
    - For u=j, v=k: a[j,k]*b[x, p[k]]
    - For u=k, v=j: a[k,j]*b[p[k], x]
    
    Delta = new - old for these terms.
    """
    n = len(p)
    p = np.array(p, dtype=np.int64)
    best_cost = _compute_cost(a, b, p)
    
    # Precompute row/col sums of a for faster delta computation
    # a_rows[i] = a[i, :] as a vector
    # a_cols[j] = a[:, j] as a vector
    
    deadline = time.time() + time_limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                x = p[i]
                y = p[j]
                # Compute delta
                # Terms with both u,v in {i,j}:
                # old: a[i,i]*b[x,x] + a[i,j]*b[x,y] + a[j,i]*b[y,x] + a[j,j]*b[y,y]
                # new: a[i,i]*b[y,y] + a[i,j]*b[y,x] + a[j,i]*b[x,y] + a[j,j]*b[x,x]
                delta_ii = a[i, i] * (b[y, y] - b[x, x])
                delta_ij = a[i, j] * (b[y, x] - b[x, y])
                delta_ji = a[j, i] * (b[x, y] - b[y, x])
                delta_jj = a[j, j] * (b[x, x] - b[y, y])
                
                # Terms with u=i, v=k (k!=i,j): a[i,k]*(b[y,p[k]] - b[x,p[k]])
                # Terms with u=k, v=i (k!=i,j): a[k,i]*(b[p[k],y] - b[p[k],x])
                # Terms with u=j, v=k (k!=i,j): a[j,k]*(b[x,p[k]] - b[y,p[k]])
                # Terms with u=k, v=j (k!=i,j): a[k,j]*(b[p[k],x] - b[p[k],y])
                
                # Vectorized:
                # mask excluding i and j
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                p_masked = p[mask]
                
                # a[i, k] for k in mask
                ai_masked = a[i, mask]
                aj_masked = a[j, mask]
                am_i = a[mask, i]
                am_j = a[mask, j]
                
                # b[y, p[k]] - b[x, p[k]] for k in mask
                diff_yx = b[y, p_masked] - b[x, p_masked]
                # b[p[k], y] - b[p[k], x] for k in mask
                diff_py = b[p_masked, y] - b[p_masked, x]
                
                delta_i_k = np.dot(ai_masked, diff_yx)
                delta_k_i = np.dot(am_i, diff_py)
                delta_j_k = np.dot(aj_masked, -diff_yx)  # b[x,p[k]] - b[y,p[k]] = -(b[y,p[k]]-b[x,p[k]])
                delta_k_j = np.dot(am_j, -diff_py)
                
                total_delta = delta_ii + delta_ij + delta_ji + delta_jj + delta_i_k + delta_k_i + delta_j_k + delta_k_j
                
                if total_delta < 0:
                    p[i], p[j] = y, x
                    best_cost += total_delta
                    improved = True
                # else: no change
    
    return p.tolist(), best_cost


def _random_start(a, b, n, rng):
    """Generate a random permutation."""
    p = np.arange(n)
    rng.shuffle(p)
    return p.tolist()


def _rank_matching_start(a, b):
    """Rank-matching heuristic."""
    n = len(a)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p.tolist()


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    # Initial solutions
    best_p = None
    best_cost = float('inf')
    
    # 1. Rank matching
    p_rank = _rank_matching_start(a, b)
    cost_rank = _compute_cost(a, b, np.array(p_rank, dtype=np.int64))
    if cost_rank < best_cost:
        best_cost = cost_rank
        best_p = p_rank[:]
    
    # 2. A few random starts
    num_random = 5
    for _ in range(num_random):
        if time.time() > deadline:
            break
        p_rand = _random_start(a, b, n, rng)
        cost_rand = _compute_cost(a, b, np.array(p_rand, dtype=np.int64))
        if cost_rand < best_cost:
            best_cost = cost_rand
            best_p = p_rand[:]
    
    # Local search on best solution
    # Use fast swap optimize with remaining time
    remaining = deadline - time.time()
    if remaining > 5.0:
        # Do local search
        time_for_ls = remaining - 10.0  # leave 10s buffer
        if time_for_ls > 1.0:
            p_ls, cost_ls = _fast_swap_optimize(a, b, best_p, time_for_ls)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls[:]
    
    # Try multiple restarts if time permits
    while time.time() < deadline - 20.0:
        p_rand = _random_start(a, b, n, rng)
        cost_rand = _compute_cost(a, b, np.array(p_rand, dtype=np.int64))
        
        # Quick local search
        remaining_for_ls = deadline - time.time() - 15.0
        if remaining_for_ls > 2.0:
            p_ls, cost_ls = _fast_swap_optimize(a, b, p_rand, min(remaining_for_ls, 10.0))
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls[:]
        elif cost_rand < best_cost:
            best_cost = cost_rand
            best_p = p_rand[:]
    
    # Final polish if time allows
    remaining = deadline - time.time()
    if remaining > 5.0:
        p_final, cost_final = _fast_swap_optimize(a, b, best_p, remaining - 2.0)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final[:]
    
    return best_p
# EVOLVE-BLOCK-END
