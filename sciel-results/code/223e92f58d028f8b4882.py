"""Matrix multiplication tensor decomposition for 3x5 * 5x5."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT - SAFETY_MARGIN
    
    # Schoolbook decomposition: rank = n*m*p = 75
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower-rank decomposition using structured search
    # Strategy: Use Kronecker product structure to explore rank reductions
    # For 3x5 * 5x5, try ranks below 75 using tensor rank algorithms
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Check time before attempting search
    if time.time() > deadline - 10:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Attempt structured decomposition search
    # The tensor T[i,j,k] = delta[i][k] for matrix mult (A is n x m, B is m x p)
    # Actually T[i,j,k] corresponds to C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor has T[i,j,k] = 1 if we're looking at the (i,j,k) term
    # The tensor is: T[i,j,k] = 1 for all (i,j,k) in the multiplication pattern
    # Wait, the matrix multiplication tensor is:
    # T[i,j,k] = 1 if the (i,j) entry of A multiplies the (j,k) entry of B
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is actually the all-ones tensor of shape (n,m,p)!
    
    # No wait, that's not right. The matrix multiplication tensor M_{n,m,p} is defined as:
    # M[i,j,k] = 1 if j is the shared index, i.e., for each (i,j,k) the term A[i,j]*B[j,k]
    # So M[i,j,k] = 1 for all valid (i,j,k). The rank of this tensor is what we're minimizing.
    
    # Actually, the standard matrix multiplication tensor has:
    # T[i,j,k] = 1 for the "structure" of matrix multiplication
    # Specifically, T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to C[i,k]
    # This means T[i,j,k] = 1 for ALL (i,j,k) in [0,n)x[0,m)x[0,p)
    
    # So we need the rank of the all-ones tensor of shape (3,5,5)
    # The rank of the all-ones tensor of shape (n,m,p) is known to be n*m*p for the schoolbook
    # But can it be lower?
    
    # For the all-ones tensor J_{n,m,p}, the tensor rank is at least max(n,m,p)
    # and at most n*m*p.
    
    # Let's try to find a decomposition of rank r < n*m*p
    # We need vectors u_c, v_c, w_c such that:
    # sum_c u_c[i] * v_c[j] * w_c[k] = 1 for all (i,j,k)
    
    # This is equivalent to finding the tensor rank of the all-ones tensor.
    
    # For small cases, we can try random search or structured approaches.
    
    # Try Kronecker product approach:
    # If we can write the all-ones tensor as a sum of rank-1 tensors
    # J_{n,m,p} = sum_{c=1}^r u_c \otimes v_c \otimes w_c
    
    # One approach: use the fact that J_{n,m,p} = J_{n,1,1} \otimes J_{1,m,1} \otimes J_{1,1,p}
    # But this gives rank 1, which is wrong because the tensor rank is not multiplicative this way.
    
    # Actually, let me reconsider. The matrix multiplication tensor is NOT the all-ones tensor.
    # Let me re-read the problem.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix by a 5 by 5 matrix."
    # The output factor follows the official trace convention W[k*n+i,r].
    
    # The matrix multiplication tensor T has entries T[i,j,k] where:
    # - i indexes rows of A (0 to n-1)
    # - j indexes columns of A / rows of B (0 to m-1)  
    # - k indexes columns of B (0 to p-1)
    
    # And T[i,j,k] = 1 for all (i,j,k) because every (i,j) entry of A multiplies every (j,k) entry of B
    # when j is the shared index.
    
    # So yes, it IS the all-ones tensor of shape (n,m,p) = (3,5,5).
    
    # The tensor rank of the all-ones tensor J_{n,m,p}:
    # - Lower bound: max(n, m, p) = 5
    # - Upper bound: n*m*p = 75 (schoolbook)
    
    # Known results:
    # - For J_{2,2,2}, rank = 7 (Strassen)
    # - For J_{n,n,n}, the rank is between n^2 and n^3
    
    # For J_{3,5,5}, let's try to find a good decomposition.
    
    # Approach: Use the fact that J_{n,m,p} can be decomposed using:
    # J_{n,m,p} = sum_{a=1}^n sum_{b=1}^m sum_{c=1}^p e_a \otimes e_b \otimes e_c
    # This is the schoolbook with rank n*m*p.
    
    # Can we do better? Let's try a random search for lower rank decompositions.
    
    def try_decomposition(rank_target, n, m, p, max_iters=1000):
        """Try to find a rank-1 decomposition of the all-ones tensor."""
        # We need: sum_{c=1}^{rank_target} u_c[i] * v_c[j] * w_c[k] = 1 for all (i,j,k)
        
        # Initialize with random vectors
        best_error = float('inf')
        best_result = None
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random initialization
            U_r = np.random.randn(n, rank_target)
            V_r = np.random.randn(m, rank_target)
            W_r = np.random.randn(p, rank_target)
            
            # Normalize to help convergence
            # We want sum_c u_c[i] v_c[j] w_c[k] = 1
            # This is a 3-way tensor completion problem
            
            # Use alternating least squares
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    # U[i,:] should satisfy: sum_c U[i,c] V[j,c] W[k,c] = 1 for all j,k
                    # For fixed i: sum_c U[i,c] (V[:,c] * W[:,c])^T = ones(m*p)
                    # Let M_c = V[:,c] * W[:,c] (element-wise), shape (m*p,)
                    # Then U[i,:] @ M = ones(m*p) where M is (m*p, rank_target)
                    M = (V_r * W_r).reshape(m * p, rank_target)
                    try:
                        U_r[i, :] = np.linalg.lstsq(M, np.ones(m * p), rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for j in range(m):
                    M = (U_r * W_r).reshape(n * p, rank_target)
                    try:
                        V_r[j, :] = np.linalg.lstsq(M, np.ones(n * p), rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for k in range(p):
                    M = (U_r * V_r).reshape(n * m, rank_target)
                    try:
                        W_r[k, :] = np.linalg.lstsq(M, np.ones(n * m), rcond=None)[0]
                    except:
                        pass
            
            # Check error
            T_approx = np.zeros((n, m, p))
            for c in range(rank_target):
                T_approx += np.outer(U_r[:, c], np.outer(V_r[:, c], W_r[:, c]))
            error = np.abs(T_approx - 1.0).max()
            
            if error < best_error:
                best_error = error
                best_result = (U_r.copy(), V_r.copy(), W_r.copy())
            
            if error < 1e-6:
                break
        
        return best_result, best_error
    
    # Try ranks from lower to higher
    for r in range(max(n, m, p), min(rank, 40)):
        if time.time() > deadline - 5:
            break
        
        result, error = try_decomposition(r, n, m, p, max_iters=100)
        
        if result is not None and error < 1e-6:
            U_r, V_r, W_r = result
            # Convert to list format with integer/half-integer entries if possible
            # Round to nearest 0.5
            U_new = [[round(U_r[i, c] * 2) / 2 for c in range(r)] for i in range(n * m)]
            V_new = [[round(V_r[j, c] * 2) / 2 for c in range(r)] for j in range(m * p)]
            W_new = [[round(W_r[k, c] * 2) / 2 for c in range(r)] for k in range(p * n)]
            
            # Verify
            T_check = np.zeros((n, m, p))
            for c in range(r):
                u_col = [U_new[i * m + j][c] for i in range(n) for j in range(m)]
                v_col = [V_new[j * p + k][c] for j in range(m) for k in range(p)]
                w_col = [W_new[k * n + i][c] for k in range(p) for i in range(n)]
                # Reconstruct
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_check[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
            
            if np.abs(T_check - 1.0).max() < 1e-6:
                best_rank = r
                best_U, best_V, best_W = U_new, V_new, W_new
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
