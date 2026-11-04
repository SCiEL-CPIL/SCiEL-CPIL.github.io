"""
Solver for QAPLIB wil100 using Doubly Stochastic Manifold Optimization.
Relaxes the permutation constraint to the Birkhoff polytope, optimizes the
quadratic cost via gradient descent with Sinkhorn projection, and rounds
the result to a permutation using the Hungarian algorithm.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _sinkhorn(a, b, n, max_iter=2000, tol=1e-6):
    """
    Perform Sinkhorn scaling on a matrix to make it doubly stochastic.
    This projects the current estimate onto the Birkhoff polytope (approximately).
    """
    # Initialize with uniform matrix to avoid zeros
    u = np.ones(n)
    v = np.ones(n)
    
    # Precompute log of matrix elements for stability, but we work in linear domain here for simplicity
    # If values are very small, we might need log-space, but for QAP costs, we can often just scale.
    # Let's use a simple multiplicative update.
    
    # Ensure positivity
    a = np.maximum(a, 1e-10)
    
    for _ in range(max_iter):
        # Row scaling
        row_sums = a.sum(axis=1)
        # Avoid division by zero
        row_sums = np.maximum(row_sums, 1e-10)
        u = 1.0 / row_sums
        a = u[:, np.newaxis] * a
        
        # Column scaling
        col_sums = a.sum(axis=0)
        col_sums = np.maximum(col_sums, 1e-10)
        v = 1.0 / col_sums
        a = a * v[np.newaxis, :]
        
        # Check convergence
        # Max deviation from 1 in row/col sums
        if np.max(np.abs(a.sum(axis=1) - 1.0)) < tol and np.max(np.abs(a.sum(axis=0) - 1.0)) < tol:
            break
            
    return a

def _ds_optimize(a, b, n, time_limit):
    """
    Optimize the QAP cost on the doubly stochastic manifold.
    Cost(X) = sum_{i,j} A[i,j] * (X @ B @ X.T)[i,j]
             = trace(X.T A X B)
    Gradient: G = A X B + A.T X B.T
    """
    start_time = time.time()
    
    # Initialize X as identity (a valid permutation matrix)
    # Or a random doubly stochastic matrix
    X = np.eye(n)
    
    # Learning rate schedule
    lr = 0.01
    best_X = X.copy()
    best_cost = _qap_cost_approx(a, b, X)
    
    # Precompute for efficiency
    # We need to compute gradient: G = A X B + A.T X B.T
    # And update X = X - lr * G
    # Then project X to Birkhoff polytope using Sinkhorn
    
    # To make Sinkhorn faster, we can normalize the matrix before scaling
    # Or just run it.
    
    # We will run gradient steps until time runs out
    steps = 0
    while time.time() - start_time < time_limit:
        # Compute Gradient
        # G = A @ X @ B + A.T @ X @ B.T
        G = a @ X @ b + a.T @ X @ b.T
        
        # Gradient Descent Step
        X_new = X - lr * G
        
        # Project to Birkhoff Polytope
        # Sinkhorn requires positive entries. 
        # If X_new has negative entries, we clip them to a small positive value.
        X_pos = np.maximum(X_new, 1e-8)
        
        # Normalize to sum to 1 (approximate scaling before Sinkhorn)
        total_sum = X_pos.sum()
        if total_sum > 0:
            X_pos = X_pos / total_sum
        
        # Sinkhorn Scaling
        X_proj = _sinkhorn(X_pos, b, n, max_iter=50, tol=1e-5)
        
        # Update X
        X = X_proj
        
        # Compute cost (approximate, using matrix trace)
        # Cost = trace(X.T @ A @ X @ B)
        # This is expensive: O(n^3) per step. 
        # For n=100, n^3 = 1e6. 120s / 1e6 = 120 steps? No, numpy is fast.
        # Let's check time.
        
        cost = np.trace(X.T @ (a @ X @ b))
        
        if cost < best_cost:
            best_cost = cost
            best_X = X.copy()
            
        steps += 1
        
        # Adaptive learning rate: decrease if cost oscillates or time is up
        if steps % 100 == 0:
            lr *= 0.9
            
    return best_X, best_cost

def _qap_cost_approx(a, b, X):
    """Compute approximate QAP cost for a doubly stochastic matrix X."""
    return np.trace(X.T @ (a @ X @ b))

def _round_ds_to_perm(X):
    """
    Round a doubly stochastic matrix to a permutation using the Hungarian algorithm.
    We want to maximize the overlap with X, so we minimize the cost -X.
    """
    n = len(X)
    # Cost matrix for Hungarian: we want to assign i to j where X[i,j] is high.
    # So cost = -X
    row_ind, col_ind = linear_sum_assignment(-X)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _local_search(a, b, p, time_limit):
    """
    Fast local search polish.
    Uses a precomputed structure to speed up delta calculations if possible,
    but for n=100, O(n) vectorized delta is fast enough.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    # Simple First Improvement Local Search
    # To be fast, we can just do a few passes of best improvement or first improvement.
    # Given the time limit, we'll do a robust SA-like local search.
    
    # Precompute rows of a and columns of a for faster access?
    # Actually, the vectorized delta in the previous code was quite good.
    # Let's use a simplified version of the SA polish but with a faster cooling.
    
    t0 = 500.0
    t_final = 0.1
    batch_size = 100
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        frac = min(elapsed / time_limit, 1.0)
        temp = t0 * (t_final / t0) ** frac
        
        for _ in range(batch_size):
            if time.time() - start_time > time_limit:
                break
                
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            pi = p[i]
            pj = p[j]
            
            # Fast delta
            delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
            delta += a[i, i] * (b[pj, pj] - b[pi, pi])
            delta += a[j, j] * (b[pi, pi] - b[pj, pj])
            
            diff_a_row = a[i, :] - a[j, :]
            diff_b_col = b[pj, p] - b[pi, p]
            sum_T1 = np.dot(diff_a_row, diff_b_col)
            sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
            sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
            
            diff_a_col = a[:, i] - a[:, j]
            diff_b_row = b[p, pj] - b[p, pi]
            sum_T2 = np.dot(diff_a_col, diff_b_row)
            sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
            sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
            
            delta += sum_T1 + sum_T2
            delta = int(delta)
            
            if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    deadline = 115.0  # Safety margin
    start = time.time()
    
    np.random.seed(42)
    
    best_p = None
    best_cost = np.inf
    
    # 1. Initial Solution: Random or Hungarian on row sums
    # Let's use a simple heuristic initialization to have a valid permutation
    # Sort by row sums of A and B
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    # Assign high flow to high distance? No, we want to minimize cost.
    # If A[i,j] is high, we want B[p[i], p[j]] to be low.
    # This is tricky. Let's just use random or identity for now, 
    # or better, use the DS optimization result.
    
    # 2. Doubly Stochastic Optimization
    # Allocate 60% of time to DS optimization
    ds_time = 60.0
    time_remaining = deadline - (time.time() - start)
    if time_remaining > ds_time:
        X_opt, cost_opt = _ds_optimize(a, b, n, ds_time)
        
        # Round to permutation
        p_ds = _round_ds_to_perm(X_opt)
        cost_ds = _qap_cost(a, b, p_ds)
        
        if cost_ds < best_cost:
            best_cost = cost_ds
            best_p = p_ds.copy()
    else:
        # Fallback: Identity permutation
        best_p = np.arange(n)
        best_cost = _qap_cost(a, b, best_p)
        
    # 3. Polish with Local Search / SA
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 5 and best_p is not None:
        # Use remaining time for SA polish
        sa_time = min(50.0, time_remaining - 2)
        p_final, cost_final = _local_search(a, b, best_p.copy(), sa_time)
        
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
            
    # If we still have time, do another round of DS from the polished solution?
    # Maybe not, as DS is slow. Let's stick to one DS + Polish.
    
    # Ensure best_p is not None
    if best_p is None:
        best_p = np.arange(n)
        
    return best_p.tolist()

# EVOLVE-BLOCK-END
