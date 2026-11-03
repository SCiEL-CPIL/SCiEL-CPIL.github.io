"""Rank-minimal tensor decomposition for 3x3 * 3x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    
    The tensor T has indices (i, j, k) where:
    - i in {0,1,2} (row of output, n=3)
    - j in {0,1,2} (column of first matrix / row of second, m=3)  
    - k in {0,1,2,3,4} (column of second matrix, p=5)
    
    T[i,j,k] = 1 if the (i,j) entry of A times the (j,k) entry of B contributes to C[i,k]
    
    We want to write T as sum over r of U_r ⊗ V_r ⊗ W_r where:
    - U_r is a vector of length n*m = 9
    - V_r is a vector of length m*p = 15
    - W_r is a vector of length p*n = 15
    
    The score is the rank (number of terms), lower is better.
    """
    n, m, p = 3, 3, 5
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # This is the "standard" matrix multiplication tensor.
    
    # Strategy: Use a structured approach based on known low-rank decompositions.
    # For 3x3x3, Strassen gives rank 23.
    # For 3x3x5, we can try to extend this.
    
    # Let's use an optimization-based approach to find a low-rank decomposition.
    # We'll use a greedy/iterative method.
    
    # First, let's try to construct a rank-40 decomposition using a known method.
    # The idea: decompose the 3x5 output into a 3x3 part and a 3x2 part.
    # Use 23 terms for the 3x3x3 part (Strassen-like)
    # Use 17 terms for the 3x3x2 part
    
    # Actually, let's try a more systematic approach using numerical optimization.
    
    # The tensor T can be represented as a 3D array of shape (n, m, p) = (3, 3, 5)
    T = np.ones((n, m, p))
    
    # We want to find rank r such that T = sum_{l=0}^{r-1} u_l ⊗ v_l ⊗ w_l
    # where u_l ∈ R^{n*m}, v_l ∈ R^{m*p}, w_l ∈ R^{p*n}
    
    # Actually, the standard formulation is:
    # C = A * B where A is n×m, B is m×p, C is n×p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T[i,j,k] = 1 for all i,j,k (since each term A[i,j]*B[j,k] contributes)
    
    # In the tensor decomposition framework:
    # We need U (n*m × r), V (m*p × r), W (p*n × r) such that
    # for each (i,k): C[i,k] = sum_j A[i,j] B[j,k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
    
    # Wait, that's not quite right. Let me reconsider.
    
    # The standard tensor decomposition for matrix multiplication:
    # T ∈ R^{n×m×p} where T[i,j,k] = δ_{i,k} (no, that's not right either)
    
    # Actually, for matrix multiplication C = AB:
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # This can be written as: C = sum_l (U_l A) * (B V_l) * W_l
    # where U_l is n×n, V_l is p×p, W_l is a scalar... no.
    
    # Let me use the correct formulation. The tensor for matrix multiplication is:
    # T[i, j, k] = 1 for all valid indices (since every A[i,j]*B[j,k] pair contributes)
    
    # But that's not quite it either. The correct tensor is defined such that:
    # <T, A ⊗ B> = C where A is reshaped to n*m, B to m*p, C to n*p
    
    # So T is a 3D array of shape (n*m, m*p, n*p) but that's not standard.
    
    # Let me use the standard formulation from the task:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The output factor follows W[k*n+i, r]
    
    # The decomposition should satisfy:
    # For all A (n×m) and B (m×p), C = A*B satisfies:
    # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
    
    # Wait, that doesn't look right either. Let me think more carefully.
    
    # Actually, I think the correct interpretation is:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j
    
    # No, that's not standard. Let me look at this differently.
    
    # The matrix multiplication tensor T has entries T[i,j,k] where:
    # - i indexes the row of A and C (0 to n-1)
    # - j indexes the column of A and row of B (0 to m-1)  
    # - k indexes the column of B and C (0 to p-1)
    
    # T[i,j,k] = 1 for all i,j,k (since each A[i,j]*B[j,k] contributes to C[i,k])
    
    # The rank is the minimum number of terms in a decomposition:
    # T = sum_{r=0}^{rank-1} u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^n, v_r ∈ R^m, w_r ∈ R^p
    
    # But the task specifies U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # This suggests a different indexing convention.
    
    # Let me re-read the task: "The output factor follows the official trace convention W[k*n+i,r]"
    
    # I think the correct interpretation is:
    # We want to compute C = A * B where A is n×m, B is m×p
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # The decomposition uses rank r terms:
    # C[i,k] = sum_{r=0}^{r-1} (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
    
    # Hmm, that still doesn't look right. Let me try a different approach.
    
    # Actually, I think the standard formulation is:
    # Define linear forms:
    # For each r, define:
    #   f_r(A) = sum_{i,j} U[i*m+j, r] * A[i,j]
    #   g_r(B) = sum_{j,k} V[j*p+k, r] * B[j,k]  
    #   h_r(C) = sum_{i,k} W[k*n+i, r] * C[i,k]
    
    # And the decomposition satisfies: h_r(C) = f_r(A) * g_r(B) for all A, B, and for each r
    # And C = sum_r h_r(C) (in some sense)
    
    # Actually, I think the correct condition is:
    # For all A, B: C = A*B satisfies
    # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
    
    # No wait, that has an extra sum over j inside. Let me think again.
    
    # I believe the correct formulation is:
    # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
    
    # But this has two separate sums over j, which doesn't make sense for matrix multiplication.
    
    # Let me try yet another interpretation. Perhaps:
    # C[i,k] = sum_r sum_j A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This would mean that for each (i,j,k), we have a coefficient:
    # coeff[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And we need coeff[i,j,k] = 1 for all i,j,k.
    
    # Yes! That makes sense. The tensor T[i,j,k] = 1 for all i,j,k, and we want to decompose it.
    
    # So the problem is: find U (9×r), V (15×r), W (15×r) such that
    # for all i in 0..2, j in 0..2, k in 0..4:
    #   sum_{r=0}^{r-1} U[i*3+j, r] * V[j*5+k, r] * W[k*3+i, r] = 1
    
    # This is a tensor decomposition problem. The rank is r.
    
    # Let's use a numerical optimization approach to find a low-rank decomposition.
    
    # We'll use an iterative method: start with some initial guess and refine.
    
    # Actually, let's try to use known constructions. For 3x3x3, Strassen gives rank 23.
    # For 3x3x5, we can try to extend this.
    
    # Let's first try to construct a rank-40 decomposition, then see if we can reduce it.
    
    # Simple approach: use the schoolbook decomposition with rank 45, then try to reduce.
    
    # Schoolbook: for each (i,j,k), use one term.
    rank_schoolbook = n * m * p  # 45
    
    # Let's try to find a better decomposition using optimization.
    
    # We'll parameterize U, V, W and minimize the error.
    
    # Start with rank 40 and see if we can achieve zero error.
    
    def try_rank(r, max_iter=1000, tol=1e-10):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42)
        
        # Initialize U, V, W with small random values
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        best_error = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute the error: for each (i,j,k), 
            # err[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
            err = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r_idx in range(r):
                            val += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                        err[i,j,k] = val - 1.0
            
            error = np.sum(err**2)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < tol:
                return best_U, best_V, best_W
            
            # Gradient descent step
            lr = 0.01 / (1 + it * 0.01)
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = err[i,j,k]
                        for r_idx in range(r):
                            dU[i*m+j, r_idx] += e * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                            dV[j*p+k, r_idx] += e * U[i*m+j, r_idx] * W[k*n+i, r_idx]
                            dW[k*n+i, r_idx] += e * U[i*m+j, r_idx] * V[j*p+k, r_idx]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_error < 1e-6:
            return best_U, best_V, best_W
        return None
    
    # Try ranks from low to high
    for r in range(25, 46):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iter=500)
        if result is not None:
            U, V, W = result
            # Verify the decomposition
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r_idx in range(r):
                            val += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-8:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If optimization didn't find a solution, fall back to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
