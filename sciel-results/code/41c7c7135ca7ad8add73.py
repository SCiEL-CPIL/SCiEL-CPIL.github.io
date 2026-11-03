"""Rewritten program: rank-minimal exact tensor decomposition for 3x3x7 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The multiplication tensor T[i,j,k] = 1 if we compute C[i,k] = sum_j A[i,j]*B[j,k].
    We need U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For 3x3x7, the schoolbook rank is 3*3*7 = 63.
    We can potentially do better by using Strassen-like methods for the 3x3x3 part
    and then handling the 3x3x4 part.
    
    Actually, let's think about this more carefully. The tensor is:
    T[i,j,k] for i in [0,3), j in [0,3), k in [0,7)
    
    We can split k into two parts: k in [0,3) and k in [3,7).
    For k in [0,3): this is a 3x3x3 tensor, which has rank 23 (Strassen).
    For k in [3,7): this is a 3x3x4 tensor.
    
    But these are not independent - they share the same A and B factors.
    
    Let me try a different approach: use the known minimal rank for 3x3x3 (which is 23)
    and combine it with a decomposition for the remaining 4 columns.
    
    Actually, the cleanest approach: the tensor T[i,j,k] for k=0..6 can be seen as 
    7 copies of the 3x3 multiplication, but they share structure.
    
    Let me use a direct approach: try to find a decomposition of rank r where
    r < 63. I'll use the fact that for the 3x3x3 part, Strassen gives rank 23.
    
    For the full 3x3x7 tensor, one approach:
    - Decompose into 7 separate 3x3x1 tensors (each of rank 3 in schoolbook, but the full
      3x3x3 needs 23).
    
    Actually, let me just implement a numerical search for a low-rank decomposition.
    
    Strategy: Use the tensor T[i,j,k] = 1 for all i,j,k (since every output C[i,k] 
    depends on every A[i,j] and B[j,k] in the standard multiplication).
    
    Wait, no. The tensor T[i,j,k] is the structure tensor of matrix multiplication:
    T[i,j,k] = 1 if the (i,k)-th entry of C = A*B involves A[i,j]*B[j,k].
    So T[i,j,k] = 1 for all i,j,k in the appropriate ranges.
    
    For 3x3x7: i in [0,3), j in [0,3), k in [0,7). T[i,j,k] = 1 for all i,j,k.
    
    We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    This is equivalent to: for each (i,k), the vector (U[:,i*m+*] * V[:,*] * W[k*n+i,:]) 
    summed over c gives the product structure.
    
    Let me use a different formulation. Define:
    U is 9 x r (n*m = 9)
    V is 21 x r (m*p = 21)  
    W is 21 x r (p*n = 21)
    
    Constraint: for all i in [0,3), j in [0,3), k in [0,7):
    sum_c U[i*3+j, c] * V[j*7+k, c] * W[k*3+i, c] = 1
    
    This is a system of 3*3*7 = 63 equations in 9r + 21r + 21r = 51r unknowns.
    
    For r=63 (schoolbook), we have plenty of freedom. We want to minimize r.
    
    Known results: The rank of 3x3x3 matrix multiplication is 23 (Strassen).
    For 3x3x7, I believe the rank is at most 23 + 3*4 = 35 if we can separate, but
    it might be lower due to shared structure.
    
    Let me try a constructive approach using Strassen for the first 3 k-values and
    schoolbook for the last 4, giving rank 23 + 12 = 35.
    
    Actually, let me just do a numerical optimization to find the minimal rank.
    I'll start with r=35 and try to verify, or use a smarter construction.
    
    Let me use the known Strassen decomposition for 3x3x3 and extend it.
    """
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin for 150s limit
    
    n, m, p = 3, 3, 7
    
    def schoolbook_decomposition():
        """Return the schoolbook decomposition of rank 63."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    def verify(U, V, W, tol=1e-10):
        """Verify that the decomposition is exact."""
        n, m, p = 3, 3, 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def strassen_3x3x3():
        """
        Return Strassen decomposition for 3x3x3 matrix multiplication.
        U: 9 x 23, V: 21 x 23, W: 9 x 23 (for 3x3x3, p=3 so W is 9x23)
        
        Strassen's algorithm for 2x2x2 has rank 7. For 3x3x3, we can use
        the known minimal decomposition of rank 23.
        
        Let me use a known explicit Strassen-type decomposition for 3x3.
        
        Actually, let me construct it by embedding the 2x2 Strassen and handling
        the extra row/column.
        
        For simplicity, let me use the fact that rank(3x3x3) = 23 is known,
        and construct an explicit decomposition.
        
        Here's one approach: use the 7-term Strassen for the top-left 2x2 block,
        then handle the interactions with the 3rd row and column.
        """
        # This is complex. Let me use a different strategy.
        # I'll use a numerical approach to find a rank-23 decomposition.
        # Or better, let me just use the schoolbook for now and try to improve.
        pass
    
    # Let me try a direct numerical approach.
    # Start with schoolbook rank 63, then try to reduce.
    
    # First, get the schoolbook decomposition as a baseline.
    U_sb, V_sb, W_sb = schoolbook_decomposition()
    
    if time.time() > deadline - 10:
        # Not enough time, return schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find a lower rank decomposition using optimization.
    # The constraint is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a non-convex problem. Let me try a gradient-based approach.
    
    # Actually, let me try a smarter constructive approach.
    # 
    # Key insight: The 3x3x7 tensor can be viewed as the tensor product structure.
    # For each k, we have a 3x3 slice. The tensor is:
    # T[i,j,k] = 1 for all i,j,k
    #
    # This means the matrix multiplication tensor for 3x3x7 has all entries equal to 1.
    #
    # Wait, that's not right. Let me re-examine.
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The structure tensor T has T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k].
    # For standard matrix multiplication, T[i,j,k] = 1 for ALL i,j,k (since every A[i,j]
    # multiplies every B[j,k] to contribute to C[i,k]... wait no.
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the term A[i,j]*B[j,k] contributes to C[i,k].
    # Therefore T[i,j,k] = 1 for all valid i,j,k.
    #
    # Yes! The structure tensor of matrix multiplication has T[i,j,k] = 1 for all i,j,k.
    #
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # Let me try to find a decomposition of rank r < 63.
    # 
    # Approach: Use the fact that this is equivalent to finding a bilinear algorithm.
    # 
    # Let me try rank 35 first (23 for 3x3x3 + 12 for 3x3x4 schoolbook).
    
    # Actually, let me try a simpler approach: use random initialization and gradient descent.
    
    def try_rank(r, max_iter=5000, lr=0.01):
        """Try to find a rank-r decomposition using gradient descent."""
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(42)
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # The loss is: sum over (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        # 
        # Gradient w.r.t. U[i*m+j, c]:
        # d/dU[i*m+j,c] = 2 * (sum_c' U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c'] - 1) * V[j*p+k,c] * W[k*n+i,c]
        # summed over j,k for fixed i... actually let me just compute it numerically.
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err = val - 1.0
                        grad_U[i*m+j, :] += 2 * err * V[j*p+k, :] * W[k*n+i, :]
                        grad_V[j*p+k, :] += 2 * err * U[i*m+j, :] * W[k*n+i, :]
                        grad_W[k*n+i, :] += 2 * err * U[i*m+j, :] * V[j*p+k, :]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if it % 50 == 0:
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                if max_err < 1e-10:
                    return U, V, W
        
        # Check if we converged
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-10:
            return U, V, W
        return None
    
    # Try ranks from 30 to 62
    best_result = None
    for r in range(30, 63):
        if time.time() > deadline - 15:
            break
        result = try_rank(r, max_iter=2000, lr=0.005)
        if result is not None:
            U, V, W = result
            best_result = (r, U, V, W)
            break  # Found the smallest rank that works
    
    if best_result is not None:
        r, U, V, W = best_result
        # Round to nearest half-integer for cleaner output
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if verify(U_rounded, V_rounded, W_rounded):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
