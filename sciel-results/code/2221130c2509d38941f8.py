"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.optimize as opt

def _calc_cost(a, b, p):
    """Calculate exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Calculate the cost change when swapping facilities at positions i and j.
    p is the current permutation (facility at position k is p[k]).
    We are swapping the facilities assigned to locations i and j.
    """
    # Let x = p[i], y = p[j]
    x = p[i]
    y = p[j]
    
    # The change in cost comes from terms involving x or y.
    # Cost = sum_{u,v} a[u,v] b[p[u], p[v]]
    
    # Terms involving x (as u or v) and y (as u or v) change.
    # Specifically:
    # Old: a[x,x]b[pi,pi] + a[y,y]b[pj,pj] + a[x,y]b[pi,pj] + a[y,x]b[pj,pi]
    #      + sum_{k!=i,j} (a[x,k]b[pi,pk] + a[k,x]b[pk,pi] + a[y,k]b[pj,pk] + a[k,y]b[pk,pj])
    # New: a[x,x]b[pj,pj] + a[y,y]b[pi,pi] + a[x,y]b[pj,pi] + a[y,x]b[pi,pj]
    #      + sum_{k!=i,j} (a[x,k]b[pj,pk] + a[k,x]b[pk,pj] + a[y,k]b[pi,pk] + a[k,y]b[pk,pi])
    
    # Delta = New - Old
    # Let's group by k.
    
    # Term k=i (where p[i]=x):
    # Old: a[x,x]b[pi,pi] + a[y,x]b[pj,pi]
    # New: a[x,x]b[pj,pj] + a[y,x]b[pi,pj]
    # Note: pi = x, pj = y.
    # Old: a[x,x]b[x,x] + a[y,x]b[y,x]
    # New: a[x,x]b[y,y] + a[y,x]b[x,y]
    
    # Term k=j (where p[j]=y):
    # Old: a[y,y]b[pj,pj] + a[x,y]b[pi,pj]
    # New: a[y,y]b[pi,pi] + a[x,y]b[pj,pi]
    # Old: a[y,y]b[y,y] + a[x,y]b[x,y]
    # New: a[y,y]b[x,x] + a[x,y]b[y,x]
    
    # Terms k != i, j (p[k] = z):
    # Old: a[x,z]b[x,z] + a[z,x]b[z,x] + a[y,z]b[y,z] + a[z,y]b[z,y]
    # New: a[x,z]b[y,z] + a[z,x]b[z,y] + a[y,z]b[x,z] + a[z,y]b[z,x]
    
    # This is getting complicated to vectorize directly without D.
    # Let's stick to the D-based vectorization which is faster.
    pass

def _local_search_vectorized(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search using vectorized delta calculations.
    For each i, computes deltas for all j > i in a single NumPy operation.
    """
    n = len(p)
    p = p.copy()
    current_cost = _calc_cost(a, b, p)
    
    # Precompute symmetric A for delta calculation
    A_sym = a + a.T
    A_diag = np.diag(a)
    
    D = b[np.ix_(p, p)].copy()
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        improved = False
        # Iterate through all i
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            js = np.arange(i + 1, n)
            
            diff_A = A_sym[i] - A_sym[js]
            diff_D = D[js] - D[i]
            
            S = np.sum(diff_A * diff_D, axis=1)
            
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            coeff_j = A_sym[i, js] - A_diag[js]
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            diag_term = (A_diag[i] - A_diag[js]) * (D[js, js] - D[i, i])
            
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S - term_i - term_j + diag_term + cross_term
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _hungarian_init(a, b):
    """
    Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix.
    Cost[i, j] = sum_k a[i,k] b[j,k] + sum_k a[k,i] b[k,j]
    """
    n = len(a)
    cost_matrix = a @ b + a.T @ b.T
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _sa_init(a, b, time_limit):
    """
    Simulated Annealing initialization to escape shallow local minima.
    Uses a simple swap move and exponential cooling.
    """
    n = len(a)
    p = np.random.permutation(n)
    current_cost = _calc_cost(a, b, p)
    
    # Precompute for fast delta? SA needs many random swaps.
    # Let's use the explicit D matrix for O(n) delta updates.
    D = b[np.ix_(p, p)].copy()
    A_sym = a + a.T
    A_diag = np.diag(a)
    
    # Temperature schedule
    t_start = time.time()
    initial_temp = 1000.0
    cooling_rate = 0.99
    
    temp = initial_temp
    
    while time.time() - t_start < time_limit:
        if temp < 1e-6:
            break
        
        # Random swap
        i, j = random.sample(range(n), 2)
        
        # Calculate delta for swap i, j
        # We can reuse the logic from local search for a single pair.
        # diff_A = A_sym[i] - A_sym[j]
        # diff_D = D[j] - D[i]
        # S = sum(diff_A * diff_D)
        
        # But we need the corrections for i and j.
        # It's easier to just calculate the full delta for this pair.
        
        # Let's implement a fast single-swap delta.
        # Delta = sum_k (A_sym[i,k] - A_sym[j,k]) * (D[j,k] - D[i,k])
        #        - (A_sym[i,i] - A_sym[j,i]) * (D[j,i] - D[i,i])
        #        - (A_sym[i,j] - A_sym[j,j]) * (D[j,j] - D[i,j])
        #        + (A_diag[i] - A_diag[j]) * (D[j,j] - D[i,i])
        #        + (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
        
        diff_A = A_sym[i] - A_sym[j]
        diff_D = D[j] - D[i]
        S = np.dot(diff_A, diff_D)
        
        term_i = (A_sym[i, i] - A_sym[j, i]) * (D[j, i] - D[i, i])
        term_j = (A_sym[i, j] - A_sym[j, j]) * (D[j, j] - D[i, j])
        diag_term = (A_diag[i] - A_diag[j]) * (D[j, j] - D[i, i])
        cross_term = (a[i, j] - a[j, i]) * (D[j, i] - D[i, j])
        
        delta = S - term_i - term_j + diag_term + cross_term
        
        if delta < 0 or random.random() < np.exp(-delta / temp):
            # Accept move
            p[i], p[j] = p[j], p[i]
            D[i], D[j] = D[j].copy(), D[i].copy()
            D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
            current_cost += int(delta)
        
        temp *= cooling_rate
        
        # Check time every few steps
        if time.time() - t_start > time_limit:
            break
            
    return p, current_cost

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit (150s kill)
    
    best_p = None
    best_cost = np.inf
    
    # 1. Initialization Phase: Generate diverse starts
    # Start 1: Hungarian
    p_hung = _hungarian_init(a, b)
    
    # Start 2: SA
    # Allocate 20% of time to SA init
    sa_time = 0.2 * (deadline - start_time)
    p_sa, cost_sa = _sa_init(a, b, sa_time)
    
    # Start 3: Random
    p_rand1 = np.random.permutation(n)
    
    initial_solutions = [p_hung, p_sa, p_rand1]
    
    # 2. Initial Refinement
    # Allocate ~30% of time to refining initial solutions to find a good "anchor"
    initial_time_budget = 0.3 * (deadline - start_time)
    time_per_init = initial_time_budget / len(initial_solutions)
    
    for p_init in initial_solutions:
        if time.time() > deadline - 5:
            break
        p_ls, cost_ls = _local_search_vectorized(a, b, p_init, time_per_init, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # 3. Iterated Local Search (ILS) Phase
    # Use remaining time for ILS starting from best_p
    current_p = best_p.copy()
    current_cost = best_cost
    
    stagnation_count = 0
    base_perturb = 4
    max_stagnation = 15
    
    while time.time() < deadline - 10.0:
        # Adaptive perturbation
        if stagnation_count > max_stagnation:
            perturb_k = base_perturb + (stagnation_count // 5)
            # Restart from best if stuck too long
            if stagnation_count > max_stagnation * 2:
                current_p = best_p.copy()
                current_cost = best_cost
                stagnation_count = 0
        else:
            perturb_k = base_perturb
        
        p_pert = _perturb(current_p, perturb_k)
        
        # Refine perturbed solution
        # Give it a decent chunk of time, but ensure we can do multiple iterations
        # Dynamic time limit: min(remaining_time * 0.5, 20s)
        remaining = deadline - time.time()
        ls_time = min(remaining * 0.5, 20.0)
        
        if ls_time < 1.0:
            break
            
        p_ls, cost_ls = _local_search_vectorized(a, b, p_pert, ls_time, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            current_p = p_ls.copy()
            current_cost = cost_ls
            stagnation_count = 0
        else:
            stagnation_count += 1
            # If no improvement, keep the perturbed solution as current for next iteration?
            # Or go back to best? Standard ILS usually keeps the perturbed one if it's not too bad,
            # but going back to best is safer for convergence. Let's stick to best if no improvement.
            current_p = best_p.copy()
            current_cost = best_cost
    
    # Final check
    if best_p is None:
        return list(range(n))
        
    if sorted(best_p.tolist()) != list(range(n)):
        return list(range(n))
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
