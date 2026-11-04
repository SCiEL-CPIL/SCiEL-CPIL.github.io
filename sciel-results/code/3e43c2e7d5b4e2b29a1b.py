"""
Sinkhorn-Knopp Continuous Relaxation Solver for QAPLIB wil100.
Strategy:
1. Formulate the QAP as a minimization over the Birkhoff polytope (doubly stochastic matrices).
   Objective: F(X) = sum_{i,j,k,l} A[i,j] B[k,l] X[i,k] X[j,l]
2. Use the Sinkhorn-Knopp algorithm (iterative row/column normalization) to project onto the 
   Birkhoff polytope while implicitly optimizing the objective via gradient descent on the 
   exponential scaling parameters. 
   Specifically, we iterate X_{t+1} = Sinkhorn(exp(-eta * grad(F, X_t))).
   However, a simpler and more robust approach for QAP is to view it as finding a permutation 
   matrix X that minimizes Tr(X^T C X) where C is a derived matrix, but standard QAP is 
   sum A_ij B_{pi pj}.
   
   Let's use the standard gradient flow on the manifold of doubly stochastic matrices.
   We maintain a matrix M and iterate:
   1. Compute gradient G of the QAP objective w.r.t X.
   2. Update M = M * exp(-lr * G)
   3. Sinkhorn normalize M to get X.
   
   The gradient of F(X) = sum A_ij B_{pi pj} X_{i,pi} X_{j,pj} is:
   G_{i,k} = sum_{j,l} A_{ij} B_{k,l} X_{j,l} + sum_{j,l} A_{ji} B_{l,k} X_{j,l}
           = (A @ (B * X.T).T)_{i,k} + (A.T @ (B.T * X).T)_{i,k} 
           Wait, let's derive carefully.
   F = sum_{i,j} A_{ij} sum_{k,l} B_{kl} X_{ik} X_{jl}
   dF/dX_{ik} = sum_j A_{ij} B_{kl} X_{jl} + sum_j A_{ji} B_{lk} X_{jl}
   Let Y = B * X.T (elementwise product). Then sum_l B_{kl} X_{jl} = (B * X.T)[k, j]? No.
   Let's use matrix notation.
   F = Tr(X^T A^T X B^T) ? No.
   F = sum_{i,j,k,l} A_{ij} B_{kl} X_{ik} X_{jl}.
   This is equal to sum_{i,k} X_{ik} (sum_{j,l} A_{ij} B_{kl} X_{jl}).
   Let C(X) be the matrix with C_{ik} = sum_{j,l} A_{ij} B_{kl} X_{jl}.
   C(X) = A @ (B * X.T).T ? 
   (B * X.T)_{k,j} = B_{k,j} X_{j,k}. No, X.T_{k,j} = X_{j,k}.
   We need sum_l B_{k,l} X_{j,l}. This is the (j,k) entry of X @ B.T.
   So sum_j A_{ij} (X @ B.T)_{j,k} = (A @ (X @ B.T))_{i,k}.
   The second term: sum_{j,l} A_{ji} B_{lk} X_{jl}.
   Note B_{lk} = B.T_{kl}.
   sum_l B.T_{kl} X_{j,l} = (X @ B)_{j,k}.
   So sum_j A_{ji} (X @ B)_{j,k} = (A.T @ (X @ B))_{i,k}.
   So Gradient G = A @ (X @ B.T) + A.T @ (X @ B).
   
   We perform gradient descent on the log-scale matrix M, then Sinkhorn normalize.
   M <- M - lr * G
   X <- Sinkhorn(M)
   
   This is a different design family (continuous manifold optimization) from discrete swaps.
"""

import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _sinkhorn(m, n_iter=50, eps=1e-8):
    """
    Perform Sinkhorn-Knopp normalization on matrix m.
    m is an n x n matrix.
    Returns a doubly stochastic matrix X.
    """
    # To avoid underflow/overflow, subtract max
    m = m - np.max(m)
    x = np.exp(m)
    
    for _ in range(n_iter):
        # Normalize rows
        row_sums = x.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        x = x / row_sums
        # Normalize columns
        col_sums = x.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        x = x / col_sums
        
    return x

def _assign_from_stochastic(x):
    """
    Convert a doubly stochastic matrix to a permutation using the Hungarian algorithm.
    """
    from scipy.optimize import linear_sum_assignment
    # We want to maximize the sum of x[i, p[i]], so minimize -x
    row_ind, col_ind = linear_sum_assignment(-x)
    p = np.empty(len(row_ind), dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _polish(a, b, p, time_limit=10.0):
    """
    Brief 2-opt polish to fix any issues from the discrete assignment.
    """
    n = len(a)
    start_time = time.time()
    best_p = p.copy()
    best_cost = _qap_cost(a, b, p)
    current_cost = best_cost
    current_p = p.copy()
    
    # Precompute for speed if possible, but n=100 is small
    # Simple loop
    while time.time() - start_time < time_limit:
        # Try a few random swaps
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        # Calculate delta
        pi = current_p[i]
        pj = current_p[j]
        
        # Direct
        delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
        # Diagonal
        delta += a[i, i] * (b[pj, pj] - b[pi, pi])
        delta += a[j, j] * (b[pi, pi] - b[pj, pj])
        
        # Cross
        diff_a_row = a[i, :] - a[j, :]
        diff_b_col = b[pj, current_p] - b[pi, current_p]
        sum_T1 = np.dot(diff_a_row, diff_b_col)
        sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
        sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
        
        diff_a_col = a[:, i] - a[:, j]
        diff_b_row = b[current_p, pj] - b[current_p, pi]
        sum_T2 = np.dot(diff_a_col, diff_b_row)
        sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
        sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
        
        delta += sum_T1 + sum_T2
        delta = int(delta)
        
        if delta < 0:
            current_p[i], current_p[j] = current_p[j], current_p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
                
    return best_p

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    np.random.seed(42)
    
    # Initialize M (log-scale matrix)
    # Start with a random or structured initialization
    # Using row sums of A and B as a heuristic initial bias
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    # We want high flow to match low distance? No, QAP minimizes sum A_ij B_pi pj.
    # If A is large, we want B to be small.
    # So C[i,j] = -A_row[i] * B_row[j] would maximize product, which minimizes cost if we treat it as such?
    # Actually, just start with zeros or small random noise to let the dynamics decide.
    # Zeros might get stuck in uniform distribution. Add small noise.
    M = np.zeros((n, n), dtype=np.float64)
    M += np.random.normal(0, 1e-3, (n, n))
    
    # Learning rate for gradient descent on M
    # The gradient magnitude can be large. Scale it.
    # Let's estimate a scale.
    scale = np.max(np.abs(a)) * np.max(np.abs(b)) * n
    if scale == 0:
        scale = 1.0
    
    lr = 1.0 / scale * 10 # Empirical scaling
    
    best_p = None
    best_cost = float('inf')
    
    # Continuous phase
    # We will run Sinkhorn iterations for a while, checking cost periodically
    # To save time, we don't check cost every iteration, but every 10 iterations.
    
    # Adaptive learning rate or fixed? Fixed with decay.
    current_lr = lr
    
    while time.time() - start_time < total_deadline - 15.0: # Reserve 15s for polish
        # Compute Gradient G
        # G = A @ (X @ B.T) + A.T @ (X @ B)
        # We need X. We get X from M via Sinkhorn.
        # To save time, we can use the previous X for the gradient step, then update X.
        # Or do a few Sinkhorn steps, then one gradient step.
        
        # Let's do:
        # 1. Sinkhorn M to get X
        # 2. Compute G
        # 3. Update M = M - lr * G
        # 4. Repeat
        
        # Optimization: We can keep X from the previous iteration as an approximation for G,
        # but it's better to normalize M first to get a valid X for G.
        
        x = _sinkhorn(M, n_iter=20) # Fewer iterations for speed, more frequent updates
        
        # Compute Gradient
        # term1 = A @ (X @ B.T)
        # term2 = A.T @ (X @ B)
        # Note: B is symmetric in wil100? QAPLIB instances usually have symmetric B.
        # Let's assume general case.
        
        xb_t = x @ b.T
        xb = x @ b
        
        g = a @ xb_t + a.T @ xb
        
        # Update M
        M -= current_lr * g
        
        # Decay learning rate
        current_lr *= 0.999
        
        # Check cost every 10 iterations (approx)
        # We need a permutation to check cost.
        # Hungarian on 100x100 is fast (~ms).
        # Let's check every 5 iterations to balance overhead.
        # We don't have an iteration counter in the while loop, so let's use time or a counter.
        
        # Actually, let's just check cost every time we finish a Sinkhorn pass if we are not too frequent.
        # But Hungarian is O(n^3) = 10^6 ops. 100 times is 10^8 ops. Fine.
        
        # To avoid doing Hungarian too often, let's count iterations.
        # We can't easily count inside the while without a variable.
        # Let's restructure: run a fixed number of inner iterations?
        # No, time-based is better.
        
        # Let's check cost roughly every 50ms.
        # This is hard to track without a timer inside the loop.
        
        # Alternative: Just check cost at the end of the continuous phase?
        # No, we might drift away from a good solution.
        
        # Let's add a counter.
        # Since we can't easily edit the loop structure without rewriting, let's just
        # assume the while loop runs fast enough that we can check cost every iteration
        # if we optimize the Hungarian call.
        # Actually, 100x100 Hungarian is very fast in SciPy.
        
        p = _assign_from_stochastic(x)
        cost = _qap_cost(a, b, p)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
    # Polish phase
    if best_p is not None:
        best_p = _polish(a, b, best_p, time_limit=10.0)
        # Final cost check
        final_cost = _qap_cost(a, b, best_p)
        if final_cost < best_cost:
            best_cost = final_cost
            # best_p is already updated by polish
    else:
        # Fallback if continuous phase failed to produce a valid permutation (shouldn't happen)
        best_p = np.arange(n)
        
    return best_p.tolist()

# EVOLVE-BLOCK-END
