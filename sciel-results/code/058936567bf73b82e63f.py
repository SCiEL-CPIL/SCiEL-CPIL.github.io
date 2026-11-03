"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 10.0
    n, m, p = DIMENSIONS
    
    # Schoolbook rank 27 as guaranteed fallback
    def schoolbook():
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
    
    # Verify a candidate decomposition
    def verify(U, V, W, tol=1e-6):
        r = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_{i,k'} * delta_{j,l'}
        # More precisely, the multiplication tensor T[i,j,k] where output[i,j] = sum_k A[i,k]*B[k,j]
        # Actually the tensor for matmul: T_{i,j,k} = 1 if this term appears in C[i,j]
        # C[i,j] = sum_k A[i,k] B[k,j]
        # So T[i,j,k] = 1 for all i,j,k (each term A[i,k]*B[k,j] contributes to C[i,j])
        # Wait, that's not right. The tensor decomposition is:
        # T[i,j,k] = sum_c U[i,j,c] * V[k,c] * W[c]  ... no
        # 
        # Standard: C = A * B, C[i,j] = sum_k A[i,k] * B[k,j]
        # Tensor form: T[i,j,k] = A[i,k] * B[k,j] * delta_{i,j}... no
        # 
        # The matrix multiplication tensor T has T[i,j,k] = 1 if (i,j,k) corresponds to a term in the product
        # Actually T[i,j,k] = 1 for all i,j,k in [0,n)x[0,p)x[0,m) where the term A[i,k]*B[k,j] goes to C[i,j]
        # 
        # In the factorization: T[i,j,k] = sum_c U[i,j,c] * V[k,j,c] * W[i,c]
        # Wait, let me re-read the task.
        #
        # "The output factor follows the official trace convention W[k*n+i,r]"
        # So W is indexed by (k*n + i), meaning W[k*n+i, r]
        # U is (n*m, rank) -> U[i*m+j, r]
        # V is (m*p, rank) -> V[j*p+k, r]
        # W is (p*n, rank) -> W[k*n+i, r]
        #
        # The decomposition should satisfy:
        # For the matmul tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where T[i,j,k] = 1 if the term A[i,k]*B[k,j] contributes to C[i,j]
        #
        # In standard matmul C[i,j] = sum_k A[i,k] B[k,j], every (i,j,k) triple has T[i,j,k] = 1
        # Wait no. Let me think again.
        #
        # A is n x m, B is m x p, C is n x p
        # C[i,j] = sum_{k=0}^{m-1} A[i,k] * B[k,j]
        #
        # The tensor T has dimensions (n, p, m) with T[i,j,k] = 1 for all valid i,j,k
        # because each term A[i,k]*B[k,j] contributes to C[i,j].
        #
        # So T[i,j,k] = 1 for all i in [0,n), j in [0,p), k in [0,m)
        #
        # And the factorization is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        #
        # Wait, that doesn't look right dimensionally. Let me re-examine.
        #
        # Actually, I think the convention might be different. Let me check with the schoolbook:
        # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
        # So for a given (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # = 1 (from the term where col corresponds to (i,j,k))
        #
        # So T[i,j,k] = 1 for all i,j,k, which matches C[i,j] = sum_k A[i,k]*B[k,j]
        # where each term has coefficient 1.
        #
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        max_err = 0.0
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    s = 0.0
                    for r in range(r):
                        s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < tol
    
    # Try to find rank 26 (Strassen-like or known optimal for 3x3x3)
    # The known minimum rank for 3x3x3 matrix multiplication is 23 (over some fields) 
    # but over rationals/reals it's known to be at most 26 (Strassen gives 23 for 2x2, 
    # and various constructions exist). Actually for 3x3x3, the best known is 26.
    
    # Let's try a search approach with time bounds
    best_result = schoolbook()
    
    # Try known rank-26 construction if available
    # For now, do a randomized search for lower rank
    np.random.seed(42)
    
    def random_search(target_rank, max_iters=10000):
        """Try to find a decomposition of target_rank"""
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Initialize randomly
        U[:] = np.random.randn(n*m, target_rank)
        V[:] = np.random.randn(m*p, target_rank)
        W[:] = np.random.randn(p*n, target_rank)
        
        # Gradient descent on the constraint that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # Loss = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > DEADLINE - SAFETY_MARGIN:
                return None
            
            # Compute gradients
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(p):
                    for k in range(m):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err = s - 1.0
                        loss += err * err
                        for r in range(target_rank):
                            grad_U[i*m+j, r] += 2 * err * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += 2 * err * U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += 2 * err * U[i*m+j, r] * V[j*p+k, r]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if loss < 1e-8:
                # Convert to fractions if possible
                U_frac = [[Fraction(int(round(U[i,r]*1000000)), 1000000) for r in range(target_rank)] for i in range(n*m)]
                V_frac = [[Fraction(int(round(V[i,r]*1000000)), 1000000) for r in range(target_rank)] for i in range(m*p)]
                W_frac = [[Fraction(int(round(W[i,r]*1000000)), 1000000) for r in range(target_rank)] for i in range(p*n)]
                
                # Verify with fractions
                U_check = [[float(U_frac[i][r]) for r in range(target_rank)] for i in range(n*m)]
                V_check = [[float(V_frac[i][r]) for r in range(target_rank)] for i in range(m*p)]
                W_check = [[float(W_frac[i][r]) for r in range(target_rank)] for i in range(p*n)]
                
                if verify(U_check, V_check, W_check, tol=1e-4):
                    return U_frac, V_frac, W_frac
                else:
                    # Try with the float values
                    if verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-6):
                        return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try rank 26
    if time.time() < DEADLINE - SAFETY_MARGIN:
        result = random_search(26, max_iters=5000)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, tol=1e-4):
                best_result = {"U": U, "V": V, "W": W}
    
    # Try rank 25
    if time.time() < DEADLINE - SAFETY_MARGIN:
        result = random_search(25, max_iters=3000)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, tol=1e-4):
                best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
