"""
Continuous Relaxation Solver for QAPLIB wil100.
Strategy:
1. Represent the assignment as a doubly stochastic matrix X (Birkhoff polytope).
2. Optimize the continuous relaxation C(X) = Tr(A X B X^T) using Projected Gradient Descent.
3. Use Barzilai-Borwein (BB) step size for adaptive step selection.
4. Project onto the Birkhoff polytope using Sinkhorn-Knopp iteration (alternating row/col normalization).
5. Extract a permutation from X using the Hungarian algorithm.
6. Refine the discrete permutation with a fast swap-based local search to exploit the continuous solution's basin.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    O(n) vectorized computation.
    """
    pi = p[i]
    pj = p[j]
    
    # Direct terms (i,j pair)
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Diagonal terms
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract k=i and k=j
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract k=i and k=j
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _sinkhorn_project(X, iters=50):
    """
    Project a positive matrix onto the Birkhoff polytope using Sinkhorn-Knopp.
    """
    X = X.copy()
    X = np.clip(X, 1e-10, None)
    
    for _ in range(iters):
        # Normalize rows
        row_sums = X.sum(axis=1, keepdims=True)
        X = X / row_sums
        # Normalize cols
        col_sums = X.sum(axis=0, keepdims=True)
        X = X / col_sums
        
    # Final adjustment to ensure exact double stochasticity (approximate)
    # For large n, Sinkhorn converges well.
    return X

def _birkhoff_gradient(A, B, X):
    """
    Compute the gradient of C(X) = Tr(A X B X^T) with respect to X.
    Gradient = A X B + A^T X B^T
    """
    # A X B
    term1 = A @ X @ B
    # A^T X B^T
    term2 = A.T @ X @ B.T
    return term1 + term2

def _local_search(a, b, p, time_limit, start_time):
    """
    First-improvement local search with random restarts if stuck.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    no_improve = 0
    max_no_improve = 2000 # Allow some wandering
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta(a, b, p, i, j)
        
        if delta < 0:
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            no_improve = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                # Perturb to escape local minimum
                for _ in range(5):
                    i = np.random.randint(n)
                    j = np.random.randint(n)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                        current_cost = _qap_cost(a, b, p)
                no_improve = 0
                
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0  # Safety margin before 120s limit
    start_time = time.time()
    
    np.random.seed(42)
    
    best_p = None
    best_cost = np.inf
    
    # Precompute constants for gradient
    A = a
    B = b
    
    # Initial X: Random positive matrix, then project
    X = np.random.rand(n, n) + 0.1
    X = _sinkhorn_project(X, iters=100)
    
    # Barzilai-Borwein state
    s_prev = None
    y_prev = None
    bb_step = 1.0 / (n * 10) # Initial small step
    
    num_cont_steps = 200 # Continuous optimization steps
    
    for step in range(num_cont_steps):
        elapsed = time.time() - start_time
        if elapsed > total_deadline - 30: # Reserve 30s for discrete refinement
            break
            
        # Compute Gradient
        G = _birkhoff_gradient(A, B, X)
        
        # Store previous gradient and X for BB
        if step > 0:
            s = X - X_prev
            y = G - G_prev
            ss = np.dot(s.flatten(), s.flatten())
            sy = np.dot(s.flatten(), y.flatten())
            
            if ss > 1e-12 and sy > 1e-12:
                # BB step size: ||s||^2 / <s, y>
                bb_step = ss / sy
                # Limit step size to prevent divergence
                bb_step = min(bb_step, 1.0)
                bb_step = max(bb_step, 1e-6)
            
        X_prev = X.copy()
        G_prev = G.copy()
        
        # Gradient Descent Step
        X_new = X - bb_step * G
        
        # Project onto Birkhoff Polytope
        X_new = _sinkhorn_project(X_new, iters=20)
        
        X = X_new
        
        # Periodically extract permutation and evaluate
        if step % 10 == 0 or step == num_cont_steps - 1:
            # Use Hungarian algorithm on X to get permutation
            # We want to maximize sum X[i, p[i]]? No, X is the assignment matrix.
            # The cost is Tr(A X B X^T).
            # The permutation p is such that X[i, p[i]] is close to 1.
            # So we solve LAP to maximize sum X[i, p[i]] (or minimize -X[i, p[i]])
            row_ind, col_ind = linear_sum_assignment(-X)
            p = np.zeros(n, dtype=np.int64)
            p[row_ind] = col_ind
            
            cost = _qap_cost(a, b, p)
            
            # Local Search Refinement
            ls_time = min(5.0, total_deadline - (time.time() - start_time) - 5)
            if ls_time > 0.5:
                p_ls, cost_ls = _local_search(a, b, p.copy(), ls_time, start_time)
            else:
                p_ls, cost_ls = p.copy(), cost
            
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls.copy()
                
    # Final Polish with Discrete SA/LS from the best continuous-derived solution
    elapsed = time.time() - start_time
    if best_p is not None and elapsed < total_deadline - 2:
        sa_time = total_deadline - elapsed - 1
        p_sa = best_p.copy()
        cost_sa = best_cost
        
        # Simple Simulated Annealing polish
        T = 50.0
        while time.time() - start_time < total_deadline - 0.5:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            delta = _swap_delta(a, b, p_sa, i, j)
            if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                p_sa[i], p_sa[j] = p_sa[j], p_sa[i]
                cost_sa += delta
                if cost_sa < best_cost:
                    best_cost = cost_sa
                    best_p = p_sa.copy()
            T = max(1.0, T * 0.995)
            
    if best_p is None:
        # Fallback: Random permutation
        best_p = np.random.permutation(n)
        
    return best_p.tolist()

# EVOLVE-BLOCK-END
