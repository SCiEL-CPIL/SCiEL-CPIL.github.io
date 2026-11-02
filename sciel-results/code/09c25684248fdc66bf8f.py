"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x4 matrix multiplication.
    
    Strategy:
    1. Use the known rank-23 decomposition for 3x3x3 (Bürgisser) as a base.
    2. Extend to 3x3x4 by handling the extra output column.
    3. Verify exactness with the correct tensor equation.
    4. If verification fails, fall back to a lower-rank search or trivial decomposition.
    
    The tensor equation is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    trivial_rank = n * m * p  # 36
    
    def verify_decomposition(U, V, W, n, m, p):
        """
        Verify that T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        """
        rank = U[0].__len__() if isinstance(U, list) else U.shape[1]
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True
    
    def trivial_decomposition(n, m, p):
        """Return the schoolbook decomposition of rank n*m*p."""
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
        return {"U": U, "V": V, "W": W}
    
    def try_burgisser_3x3x4(n, m, p):
        """
        Attempt to construct a rank-26 decomposition for 3x3x4.
        
        The idea: 3x3x4 = 3x3x3 + 3x3x1.
        Use the known rank-23 algorithm for 3x3x3, then add 3 more terms for the extra column.
        However, this naive approach gives rank 26.
        
        We'll use a structured ansatz based on the Strassen-like patterns.
        """
        # Known rank-23 decomposition for 3x3x3 (simplified pattern)
        # This is a heuristic; a full Burgisser implementation is complex.
        # For now, we'll try a structured random search with a lower rank target.
        
        best_rank = None
        best_result = None
        
        # Try ranks from 25 down to 20
        for target_rank in range(25, 19, -1):
            if time.time() > deadline:
                break
            
            # Try multiple random initializations
            for seed in range(20):
                if time.time() > deadline:
                    break
                
                np.random.seed(seed)
                
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
                
                if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
            
            if best_result is not None:
                break
        
        return best_result
    
    def try_structured_3x3x4(n, m, p):
        """
        Try a structured decomposition exploiting the rectangular asymmetry.
        
        For 3x3x4, we can think of the multiplication as:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        We'll try to construct a decomposition with rank < 36 using
        a pattern that shares terms across the extra column.
        """
        # Start with the trivial decomposition and try to merge terms
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Assign each (i,j,k) to a unique column initially
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Try to find linear dependencies to reduce rank
        # This is a simplified version of a proper rank-reduction algorithm
        
        # Convert to numpy for linear algebra
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # For each column, check if it can be expressed as a combination of others
        # This is computationally expensive, so we'll use a heuristic
        
        # Actually, let's try a different approach: use the fact that for fixed i,
        # the terms across k might have structure.
        
        # For now, return the trivial decomposition as a fallback
        return None
    
    # First, try the structured 3x3x4 approach
    result = try_structured_3x3x4(n, m, p)
    if result is not None:
        return result
    
    # Try the Burgisser-based approach
    result = try_burgisser_3x3x4(n, m, p)
    if result is not None:
        return result
    
    # Fallback: trivial decomposition
    return trivial_decomposition(n, m, p)
# EVOLVE-BLOCK-END
