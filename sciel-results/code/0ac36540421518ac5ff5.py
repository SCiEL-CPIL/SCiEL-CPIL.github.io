import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build the target tensor T[i, j, k] where:
    # i in [0, n), j in [0, m), k in [0, p)
    # T[i,j,k] = 1 if output C[i,k] uses A[i,j] and B[j,k]
    # Actually, the tensor for matmul: T[i,j,k] = delta_{output(i,k), input_A(i,j), input_B(j,k)}
    # The tensor is: T[i, j, k] = 1 iff the (i,j) element of A and (j,k) element of B contribute to (i,k) of C
    # More precisely: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for each triple (i,j,k) that appears in the sum
    # Wait, let me re-read the task.
    
    # The tensor decomposition: C[i,k] = sum_r U[i*m + j? ...]
    # Actually from the current code:
    # U has shape (n*m, rank) - indexed by (i,j)
    # V has shape (m*p, rank) - indexed by (j,k)  
    # W has shape (p*n, rank) - indexed by (k,i) with W[k*n+i]
    # The decomposition says: for each (i,j,k), the contribution to C[i,k] is
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 for the correct (i,j,k) triples and 0 otherwise
    # But actually, C[i,k] = sum_j A[i,j]*B[j,k], so for fixed (i,k), 
    # sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should give the coefficient
    # which is 1 for each j (since each A[i,j]*B[j,k] contributes with coefficient 1)
    
    # So the constraint is: for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Wait no. Let me re-examine.
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * T[i,j,k,r] where T is the decomposition
    # Actually: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j,r] * A[i,j]) * ... no
    
    # Let me think again. The standard tensor decomposition for matmul:
    # C[i,k] = sum_r sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
    # Wait, that doesn't work dimensionally.
    
    # From the current code, the trivial decomposition sets:
    # For each (i,j,k): U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # So C[i,k] = sum_col W[k*n+i, col] * (something)
    
    # I think the decomposition is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # No, that's not right either.
    
    # Actually, I believe the correct interpretation is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # Hmm, but that would make W just a scalar multiplier per r.
    
    # Let me look at it differently. The tensor T for matmul has entries:
    # T[i, j, k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, but W is indexed by (k,i) not (i,k). The task says W[k*n+i, r].
    
    # So: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # That makes sense! The tensor is all ones for the valid indices, and we need to
    # decompose it as a sum of rank-1 tensors.
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 75 equations. We want to find the minimum rank r
    # such that there exist U, V, W satisfying these equations.
    
    # Let me set up the problem and try to solve it.
    
    # The trivial solution has r = 75 (one basis vector per equation).
    # We want to find if r < 75 is possible.
    
    # Approach: Try to find a low-rank decomposition using numerical methods.
    # We can set up a least-squares problem or use iterative refinement.
    
    def build_tensor():
        """Build the target tensor values: T[i,j,k] = 1 for all valid (i,j,k)"""
        # Actually, we don't need to build the full tensor, we just need to satisfy
        # the constraints: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    def try_rank(r, n, m, p, time_limit=10):
        """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
        start = time.time()
        n_iter = 5000
        lr = 0.01
        
        # Initialize randomly
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        for it in range(n_iter):
            if time.time() > start + time_limit:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # f = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
                        # df/dU[i*m+j,r] = V[j*p+k,r]*W[k*n+i,r]
                        # df/dV[j*p+k,r] = U[i*m+j,r]*W[k*n+i,r]
                        # df/dW[k*n+i,r] = U[i*m+j,r]*V[j*p+k,r]
                        for rr in range(r):
                            u = U[i*m+j, rr]
                            v = V[j*p+k, rr]
                            w = W[k*n+i, rr]
                            f = u * v * w
                            # We want sum over r of f = 1, so error = sum_r f - 1
                            # But for gradient descent, we can use the per-term gradient
                            # scaled by the total error
                            # Actually, let's compute the total error and use it
                            pass
            
            # This is too slow for large r. Let me use a vectorized approach.
            # Actually, for small dimensions, let me just do it carefully.
            
            # Better: compute the residual for each (i,j,k) and use that to update
            # Let's use a simpler approach: for each (i,j,k), we have:
            # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is a trilinear equation.
            
            # Alternative: fix two matrices and solve for the third.
            # For fixed U, V: sum_r W[k*n+i,r] * (U[i*m+j,r]*V[j*p+k,r]) = 1
            # This is linear in W for each (i,j,k) but coupled across r.
            
            # Let me try a different approach: use the fact that we can scale.
            # If (U,V,W) is a solution, so is (cU, dV, eW) with cde=1.
            
            # Let me try a coordinate descent approach:
            # 1. Fix V and W, solve for U (linear system)
            # 2. Fix U and W, solve for V
            # 3. Fix U and V, solve for W
            
            # For fixed V, W: for each (i,j), we have:
            # sum_k sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = p  (sum over k of 1)
            # Wait no, for each (i,j,k) separately:
            # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
            # So for fixed (i,j), we have p equations (one per k) in r unknowns (U[i*m+j,0..r-1])
            # If p >= r, we can solve this as a least squares problem.
            
            # Let's do coordinate descent:
            for it2 in range(3):
                if time.time() > start + time_limit:
                    break
                
                if it2 == 0:  # Update U
                    for i in range(n):
                        for j in range(m):
                            # Build system: for each k, sum_r U[i*m+j,r] * A_k[r] = 1
                            # where A_k[r] = V[j*p+k,r] * W[k*n+i,r]
                            A = np.zeros((p, r))
                            for k in range(p):
                                for rr in range(r):
                                    A[k, rr] = V[j*p+k, rr] * W[k*n+i, rr]
                            # Solve A @ U[i*m+j,:] = 1
                            if p >= r:
                                sol, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                                U[i*m+j, :] = sol
                            else:
                                # Underdetermined, use minimum norm solution
                                sol, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                                U[i*m+j, :] = sol
                
                elif it2 == 1:  # Update V
                    for j in range(m):
                        for k in range(p):
                            # For each i, sum_r V[j*p+k,r] * U[i*m+j,r] * W[k*n+i,r] = 1
                            A = np.zeros((n, r))
                            for i in range(n):
                                for rr in range(r):
                                    A[i, rr] = U[i*m+j, rr] * W[k*n+i, rr]
                            if n >= r:
                                sol, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                                V[j*p+k, :] = sol
                            else:
                                sol, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                                V[j*p+k, :] = sol
                
                else:  # Update W
                    for k in range(p):
                        for i in range(n):
                            # For each j, sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r] = 1
                            A = np.zeros((m, r))
                            for j in range(m):
                                for rr in range(r):
                                    A[j, rr] = U[i*m+j, rr] * V[j*p+k, rr]
                            if m >= r:
                                sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                                W[k*n+i, :] = sol
                            else:
                                sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                                W[k*n+i, :] = sol
            
            # Check convergence
            if it % 100 == 0:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for rr in range(r):
                                s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                            err += (s - 1.0)**2
                if err < 1e-12:
                    break
        
        # Final verification
        if verify(U, V, W, n, m, p):
            return U, V, W
        return None
    
    # Try ranks from low to high
    # Known lower bounds: for m×n×p matmul, rank >= max(mn, mp, np) = max(15, 25, 15) = 25
    # Also rank >= ceil(mnp / something)... 
    # Let's try from 25 up to 75
    
    best_result = None
    
    # First, try some promising ranks
    ranks_to_try = list(range(25, 76))
    # Shuffle to try diverse ranks, but prioritize lower ones
    # Actually, let's try in order but with time limits
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        time_per_rank = max(1, (deadline - time.time()) / max(1, len(ranks_to_try) - ranks_to_try.index(r)))
        
        result = try_rank(r, n, m, p, time_limit=time_per_rank)
        if result is not None:
            best_result = result
            break  # Found a valid decomposition, this is the minimum rank (since we're going in order)
    
    # If we didn't find anything better, return the trivial rank-75 decomposition
    if best_result is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    else:
        U, V, W = best_result
        # Convert to list of lists with integer/half-integer entries if possible
        # The entries should be exact, so let's round to nearest rational with small denominator
        def to_rational(x, max_den=1000):
            """Convert a float to a fraction with small denominator if possible"""
            from fractions import Fraction
            fr = Fraction(x).limit_denominator(max_den)
            if abs(fr - x) < 1e-10:
                return float(fr)
            return x
        
        U_list = [[to_rational(x) for x in row] for row in U]
        V_list = [[to_rational(x) for x in row] for row in V]
        W_list = [[to_rational(x) for x in row] for row in W]
        
        return {"U": U_list, "V": V_list, "W": W_list}

