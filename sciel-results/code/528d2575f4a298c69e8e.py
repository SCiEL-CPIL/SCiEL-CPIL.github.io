"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute the exact QAP cost for a permutation p."""
    # sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Efficient: cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (sum_j a[i,j] * b[p[i], p[j]])
    # We can vectorize: for each i, the inner sum is a[i,:] dot b[p[i], :]
    # But that's O(n^2) per i -> O(n^3) total.
    # Alternative: cost = trace(a * b_perm) where b_perm[i,j] = b[p[i], p[j]]
    # b_perm = b[np.ix_(p, p)]
    # cost = np.sum(a * b[np.ix_(p, p)])
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _swap_cost_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    p[i] and p[j] are the facilities at locations i and j.
    After swap, facility p[i] goes to location j and facility p[j] goes to location i.
    
    cost = sum_{u,v} a[u,v] * b[p[u], p[v]]
    
    Let x = p[i], y = p[j].
    Before swap: contributions involving i or j:
      - a[i,i]*b[x,x] + a[i,j]*b[x,y] + a[j,i]*b[y,x] + a[j,j]*b[y,y]
      - plus cross terms: for all k != i,j: a[i,k]*b[x,p[k]] + a[k,i]*b[p[k],x] + a[j,k]*b[y,p[k]] + a[k,j]*b[p[k],y]
    
    After swap: x and y swapped at positions i and j:
      - a[i,i]*b[y,y] + a[i,j]*b[y,x] + a[j,i]*b[x,y] + a[j,j]*b[x,x]
      - plus cross terms: for all k != i,j: a[i,k]*b[y,p[k]] + a[k,i]*b[p[k],y] + a[j,k]*b[x,p[k]] + a[k,j]*b[p[k],x]
    
    Delta = after - before
    """
    x = p[i]
    y = p[j]
    
    # Diagonal terms
    delta_diag = (a[i,i] * (b[y,y] - b[x,x]) + 
                  a[j,j] * (b[x,x] - b[y,y]) +
                  a[i,j] * (b[y,x] - b[x,y]) +
                  a[j,i] * (b[x,y] - b[y,x]))
    
    # Cross terms with k != i, j
    # For position i: before has b[x, p[k]] for each k, after has b[y, p[k]]
    # For position j: before has b[y, p[k]] for each k, after has b[x, p[k]]
    # Also the reverse: a[k,i]*b[p[k],x] before, a[k,i]*b[p[k],y] after
    # and a[k,j]*b[p[k],y] before, a[k,j]*b[p[k],x] after
    
    # Vectorized: 
    # delta_cross = sum_{k != i,j} [ a[i,k]*(b[y,p[k]] - b[x,p[k]]) + a[k,i]*(b[p[k],y] - b[p[k],x])
    #                              + a[j,k]*(b[x,p[k]] - b[y,p[k]]) + a[k,j]*(b[p[k],x] - b[p[k],y]) ]
    
    # Let's build vectors for all k, then exclude i and j
    pk = p  # p[k] for all k
    
    # b[y, p[k]] - b[x, p[k]] for all k
    diff_yx = b[y, pk] - b[x, pk]  # shape (n,)
    # b[p[k], y] - b[p[k], x] for all k
    diff_pyx = b[pk, y] - b[pk, x]  # shape (n,)
    
    # a[i,k] * diff_yx[k]
    term1 = a[i, :] * diff_yx
    # a[k,i] * diff_pyx[k]
    term2 = a[:, i] * diff_pyx
    # a[j,k] * (-diff_yx[k])
    term3 = a[j, :] * (-diff_yx)
    # a[k,j] * (-diff_pyx[k])
    term4 = a[:, j] * (-diff_pyx)
    
    delta_cross_all = term1 + term2 + term3 + term4
    
    # Exclude k=i and k=j
    delta_cross_all[i] = 0
    delta_cross_all[j] = 0
    
    return delta_diag + int(np.sum(delta_cross_all))


def _hill_climb_2opt(a, b, p, max_iters=None, time_limit=None, start_time=None):
    """
    Hill-climbing via 2-opt swaps. Returns improved permutation and cost.
    """
    p = np.array(p, dtype=np.int64)
    n = len(p)
    
    if start_time is None:
        start_time = time.time()
    
    # Initial cost
    cost = _qap_cost(a, b, p)
    
    if max_iters is None:
        max_iters = 100000
    
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        if time_limit is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Find best improving swap
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Sample a subset of swaps if n is large
        # For n=256, there are ~32k pairs. Let's try all but with early termination if time is tight.
        n_pairs = n * (n - 1) // 2
        
        # To be efficient, we can do a first-improvement or best-improvement
        # Let's do best-improvement with a cap on evaluations per iteration
        max_evals_per_iter = min(n_pairs, 5000)
        
        # Random sampling of pairs if n_pairs is large
        if n_pairs > max_evals_per_iter:
            # Sample pairs
            rng = np.random.default_rng()
            # Generate random pairs
            i_vals, j_vals = [], []
            while len(i_vals) < max_evals_per_iter:
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i != j:
                    i_vals.append(i)
                    j_vals.append(j)
            i_vals = np.array(i_vals)
            j_vals = np.array(j_vals)
        else:
            i_vals, j_vals = np.triu_indices(n, k=1)
        
        for i, j in zip(i_vals, j_vals):
            delta = _swap_cost_delta(a, b, p, i, j)
            if delta < best_delta:
                best_delta = delta
                best_i = i
                best_j = j
        
        if best_i >= 0:
            # Perform the swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += best_delta
            improved = True
    
    return p, cost


def _simulated_annealing(a, b, p, time_limit, start_time=None):
    """
    Simulated annealing with 2-opt moves.
    """
    if start_time is None:
        start_time = time.time()
    
    p = np.array(p, dtype=np.int64)
    n = len(p)
    cost = _qap_cost(a, b, p)
    
    # Temperature schedule
    T0 = 1e6
    Tend = 1e2
    # We'll run until time limit
    alpha = 0.999  # cooling rate per iteration, but we'll adjust based on time
    
    rng = np.random.default_rng()
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        # Try a random swap
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        
        delta = _swap_cost_delta(a, b, p, i, j)
        
        if delta < 0:
            # Accept
            p[i], p[j] = p[j], p[i]
            cost += delta
        else:
            # Accept with probability exp(-delta / T)
            # T should decrease over time
            elapsed = time.time() - start_time
            # Linear cooling from T0 to Tend over time_limit
            T = max(Tend, T0 * (1.0 - elapsed / time_limit))
            if T > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
    
    return p, cost


def solve(a, b):
    """
    Solve the QAP instance tai256c.
    """
    start_time = time.time()
    total_time_limit = 110.0  # Safety margin from 120s
    
    n = len(a)
    
    # Phase 1: Generate initial solutions and run SA on them
    # We'll do multiple SA runs with different initializations
    
    # Initial solution 1: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    
    # Initial solution 2: Random
    rng = np.random.default_rng(42)
    p2 = rng.permutation(n).astype(np.int64)
    
    # Initial solution 3: Greedy
    p3 = np.arange(n, dtype=np.int64)
    
    # Run SA on each with a time budget
    sa_time_each = 20.0
    solutions = []
    
    for init_p in [p1, p2, p3]:
        if time.time() - start_time > total_time_limit - 30.0:
            break
        p_sa, cost_sa = _simulated_annealing(a, b, init_p, sa_time_each, start_time=start_time)
        solutions.append((p_sa, cost_sa))
    
    # Phase 2: Hill-climb the best solutions
    # Sort by cost
    solutions.sort(key=lambda x: x[1])
    
    best_p = solutions[0][0].copy()
    best_cost = solutions[0][1]
    
    # Hill-climb the top solutions
    for p_sol, cost_sol in solutions:
        if time.time() - start_time > total_time_limit - 15.0:
            break
        p_hc, cost_hc = _hill_climb_2opt(a, b, p_sol, time_limit=10.0, start_time=start_time)
        if cost_hc < best_cost:
            best_cost = cost_hc
            best_p = p_hc.copy()
    
    # Final check: ensure we return a valid permutation
    # Verify it's a permutation
    if len(best_p) != n or len(np.unique(best_p)) != n:
        # Fallback to identity
        best_p = np.arange(n, dtype=np.int64)
        best_cost = _qap_cost(a, b, best_p)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
