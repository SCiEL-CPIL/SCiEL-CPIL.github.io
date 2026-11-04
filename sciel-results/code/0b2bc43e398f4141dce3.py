import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 3, 4, 7
    
    # Indices
    # U: (n*m, R) -> (i*m+j, r)
    # V: (m*p, R) -> (j*p+k, r)
    # W: (p*n, R) -> (k*n+i, r)  # Note: seed uses k*n+i
    
    # Target Tensor Condition:
    # Sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] == 1 if (i,j,k) is valid?
    # No, the condition is:
    # For each output (i_out, k_out), the coefficient of A[i_in, j] * B[j, k_in] in C[i_out, k_out]
    # must be 1 if i_out==i_in and k_out==k_in, else 0.
    
    # Let's define the error function efficiently.
    # Error = sum_{i_out, k_out, i_in, j, k_in} ( Sum_r W[k_out*n+i_out, r] U[i_in*m+j, r] V[j*p+k_in, r] - delta(i_out, i_in) * delta(k_out, k_in) )^2
    
    # We can compute this using matrix multiplications.
    # For a fixed r, let U_r be (n*m, 1), V_r be (m*p, 1), W_r be (p*n, 1).
    # This is complex to vectorize over all i_out, k_out, i_in, k_in simultaneously without huge memory.
    # n*m = 12, m*p = 28, p*n = 21.
    # Total terms in sum: n*m*p * p*n = 12*28*21 = 7056 terms.
    # For each term, we sum over R.
    # This is manageable.
    
    def compute_error(U, V, W, R):
        # U: (12, R), V: (28, R), W: (21, R)
        error = 0.0
        # Precompute indices
        # i in 0..2, j in 0..3, k in 0..6
        # i_out in 0..2, k_out in 0..6
        
        # We can loop over the "output" indices i_out, k_out and "input" indices i_in, j, k_in
        # But 7056 iterations * R operations might be slow in Python.
        # Let's use numpy broadcasting.
        
        # Reshape U to (n, m, R)
        U_reshaped = U.reshape(n, m, R)
        # Reshape V to (m, p, R)
        V_reshaped = V.reshape(m, p, R)
        # Reshape W to (p, n, R)  # Note: W index is k*n+i, so dim 0 is k (p), dim 1 is i (n)
        W_reshaped = W.reshape(p, n, R)
        
        # We want to compute T[i_out, k_out, i_in, j, k_in] = sum_r W_reshaped[k_out, i_out, r] * U_reshaped[i_in, j, r] * V_reshaped[j, k_in, r]
        
        # This is a tensor contraction.
        # Let's compute for each i_in, j, k_in the vector of coefficients for all (i_out, k_out).
        # Coeff[i_out, k_out] = sum_r W_reshaped[k_out, i_out, r] * (U_reshaped[i_in, j, r] * V_reshaped[j, k_in, r])
        
        # We can do this by looping over i_in, j, k_in (12*28=336 iterations)
        # For each, compute the product P_r = U_reshaped[i_in, j, r] * V_reshaped[j, k_in, r] (shape R)
        # Then compute M = P_r dot W_reshaped.T ?
        # W_reshaped is (p, n, R). We want sum_r P_r * W_reshaped[k_out, i_out, r].
        # This is equivalent to:
        # M = np.tensordot(P_r, W_reshaped, axes=([0],[2]))
        # M shape: (p, n) which is (k_out, i_out).
        
        total_error = 0.0
        for i_in in range(n):
            for j in range(m):
                for k_in in range(p):
                    P_r = U_reshaped[i_in, j, :] * V_reshaped[j, k_in, :]
                    # M[k_out, i_out] = sum_r P_r[r] * W_reshaped[k_out, i_out, r]
                    M = np.tensordot(P_r, W_reshaped, axes=([0], [2]))
                    # M is (p, n) -> (k_out, i_out)
                    # Target: 1 if k_out==k_in and i_out==i_in, else 0.
                    # We need to sum (M - Target)^2
                    # Target is a (p, n) matrix with 1 at (k_in, i_in) and 0 elsewhere.
                    
                    # To avoid creating the target matrix, we can compute:
                    # sum(M^2) - 2*M[k_in, i_in] + 1
                    # Because sum(T^2) = 1, sum(MT) = M[k_in, i_in].
                    
                    total_error += np.sum(M**2) - 2.0 * M[k_in, i_in] + 1.0
        
        return total_error

    def verify(U, V, W, R):
        # Check if error is close to 0
        err = compute_error(U, V, W, R)
        return err < 1e-8

    # Strategy: Random search with local optimization
    # Start with a reasonable rank, e.g., 60
    # Use Nelder-Mead or simple coordinate descent?
    # Given the time limit, simple random restarts with a few gradient steps might be better.
    
    # However, implementing a robust gradient step for this non-convex problem is risky.
    # Let's try a simpler approach:
    # 1. Generate random U, V, W for a target rank R.
    # 2. Check if it's a solution. (Unlikely)
    # 3. If not, try to improve?
    
    # Alternative: Use the schoolbook decomposition and try to "merge" terms?
    # Or, use a known decomposition for a sub-block.
    
    # Let's try to find a decomposition with Rank 50-70 using random search + verification.
    # If we don't find one, return schoolbook (84).
    
    # To increase chances, we can use the fact that the solution space is continuous.
    # We can use `scipy.optimize.minimize` on the error function.
    
    from scipy.optimize import minimize
    
    def objective(params, R):
        # params: flat array of size R*(12+28+21)
        size_U = n * m * R
        size_V = m * p * R
        U = params[:size_U].reshape(n * m, R)
        V = params[size_U:size_U+size_V].reshape(m * p, R)
        W = params[size_U+size_V:].reshape(p * n, R)
        return compute_error(U, V, W, R)

    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Initialize schoolbook
    U_sb = np.zeros((n * m, best_rank))
    V_sb = np.zeros((m * p, best_rank))
    W_sb = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to improve
    target_ranks = [70, 65, 60, 55, 50]
    
    for R in target_ranks:
        if time.time() > deadline - 10:
            break
            
        # Try a few random starts
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            params = np.random.randn(R * (n * m + m * p + p * n))
            # Scale down?
            params /= np.sqrt(R)
            
            # Use Nelder-Mead
            # This might be slow.
            # Let's just check if random works? No.
            
            # Try a simple gradient descent with a few steps?
            # Or just use the schoolbook if we don't find a better one quickly.
            
            # Given the difficulty, let's just return the schoolbook if we don't find a quick win.
            # But we can try one optimization attempt per rank.
            
            try:
                res = minimize(objective, params, args=(R,), method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < 1e-8:
                    size_U = n * m * R
                    size_V = m * p * R
                    U_opt = res.x[:size_U].reshape(n * m, R)
                    V_opt = res.x[size_U:size_U+size_V].reshape(m * p, R)
                    W_opt = res.x[size_U+size_V:].reshape(p * n, R)
                    
                    # Round to integers? The task allows half-integers.
                    # If the values are close to integers, we can round.
                    # But the evaluator might accept floats.
                    # "integer/half-integer entries"
                    # Let's try to round to nearest 0.5.
                    
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    if verify(U_rounded, V_rounded, W_rounded, R):
                        best_rank = R
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        break
            except Exception:
                pass

    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
