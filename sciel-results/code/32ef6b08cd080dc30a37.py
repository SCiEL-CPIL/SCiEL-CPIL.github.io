"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x4 matrix multiplication.
    
    The tensor equation is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    where n=3, m=3, p=4.
    
    Strategy:
    1. Construct the trivial rank-36 decomposition.
    2. Use structured ansatz and linear algebra to reduce rank.
    3. Use random search with gradient-like refinement for lower ranks.
    4. Verify exactness and return the best result found.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    trivial_rank = n * m * p  # 36
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """
        Verify that T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        """
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
                    if not np.isclose(val, 1.0, atol=tol):
                        return False
        return True
    
    def trivial_decomposition(n, m, p):
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def try_burgisser_style(n, m, p):
        """
        Try to construct a low-rank decomposition using known patterns.
        
        For 3x3x3, the rank is 23 (Bürgisser). For 3x3x4, we expect rank ~26-28.
        We'll use a structured approach:
        1. Start with the 3x3x3 rank-23 pattern.
        2. Extend to handle the 4th output column.
        """
        # Known rank-23 decomposition for 3x3x3 (simplified)
        # This is a heuristic construction; a full Burgisser implementation is complex.
        # We'll use a pattern-based approach.
        
        # For 3x3x3, one known decomposition uses 23 terms.
        # Let's construct a pattern for 3x3x4.
        
        # Approach: Use the fact that 3x3x4 = 3x3x3 + 3x3x1
        # The 3x3x1 part can be done with 3 additional terms (one for each i,j).
        # So total rank = 23 + 3 = 26.
        
        # Let's try to construct this.
        # First, get a rank-23 decomposition for 3x3x3.
        # We'll use a simplified version based on Strassen-like patterns.
        
        # Actually, let's use a more direct approach:
        # Random search with structured initialization.
        
        best_rank = None
        best_result = None
        
        # Try ranks from 26 down to 22
        for target_rank in range(26, 21, -1):
            if time.time() > deadline:
                break
            
            # Try multiple random initializations
            for seed in range(50):
                if time.time() > deadline:
                    break
                
                np.random.seed(seed)
                
                # Initialize with structured pattern
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                
                # First 23 terms: use a pattern for 3x3x3
                # For k=0,1,2, use the 3x3x3 pattern
                # For k=3, add 3 more terms
                
                # Simplified: random initialization with some structure
                U = np.random.randn(n * m, target_rank) * 0.5
                V = np.random.randn(m * p, target_rank) * 0.5
                W = np.random.randn(p * n, target_rank) * 0.5
                
                # Try to verify
                if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
            
            if best_result is not None:
                break
        
        return best_result
    
    def try_optimization(n, m, p):
        """
        Use optimization to find a low-rank decomposition.
        
        We formulate this as: minimize ||T - sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]||
        subject to rank = r.
        
        We'll use a simple iterative approach.
        """
        best_rank = None
        best_result = None
        
        # Try ranks from 28 down to 24
        for target_rank in range(28, 23, -1):
            if time.time() > deadline:
                break
            
            # Try multiple random initializations
            for seed in range(30):
                if time.time() > deadline:
                    break
                
                np.random.seed(seed)
                
                U = np.random.randn(n * m, target_rank)
                V = np.random.randn(m * p, target_rank)
                W = np.random.randn(p * n, target_rank)
                
                # Iterative refinement
                for iteration in range(100):
                    if time.time() > deadline:
                        break
                    
                    # Compute residuals
                    residuals = {}
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                                residuals[(i, j, k)] = 1.0 - val
                    
                    # Check if converged
                    max_residual = max(abs(r) for r in residuals.values())
                    if max_residual < 1e-10:
                        if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                            best_rank = target_rank
                            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                            break
                    
                    # Update U, V, W using gradient-like steps
                    # This is a simplified approach; a proper gradient would be more complex.
                    
                    # For each term r, update U, V, W to reduce residuals
                    for r in range(target_rank):
                        # Compute gradient-like updates
                        U_grad = np.zeros((n * m, 1))
                        V_grad = np.zeros((m * p, 1))
                        W_grad = np.zeros((p * n, 1))
                        
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    res = residuals[(i, j, k)]
                                    U_grad[i * m + j, 0] += res * V[j * p + k, r] * W[k * n + i, r]
                                    V_grad[j * p + k, 0] += res * U[i * m + j, r] * W[k * n + i, r]
                                    W_grad[k * n + i, 0] += res * U[i * m + j, r] * V[j * p + k, r]
                        
                        # Update
                        learning_rate = 0.01
                        U[:, r] -= learning_rate * U_grad.flatten()
                        V[:, r] -= learning_rate * V_grad.flatten()
                        W[:, r] -= learning_rate * W_grad.flatten()
                
                if best_result is not None:
                    break
            
            if best_result is not None:
                break
        
        return best_result
    
    # First, try the optimization approach
    result = try_optimization(n, m, p)
    if result is not None:
        return result
    
    # Try the Burgisser-style approach
    result = try_burgisser_style(n, m, p)
    if result is not None:
        return result
    
    # Fallback: trivial decomposition
    return trivial_decomposition(n, m, p)
# EVOLVE-BLOCK-END
