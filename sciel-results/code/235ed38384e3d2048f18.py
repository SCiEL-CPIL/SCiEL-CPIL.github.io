"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    Strategy: Use block decomposition. View the 3x6 right operand as two 3x3 blocks.
    If we have a rank-r decomposition for 3x3x3 matrix multiplication, we can construct
    a rank-2r decomposition for 3x3x6 by applying the same decomposition to each block.
    
    The key insight: C = A * B where A is 3x3 and B is 3x6.
    Split B = [B1, B2] where B1, B2 are 3x3.
    Then C = [A*B1, A*B2].
    If A*B1 = sum_c u_c * v_c * w_c^T (rank-r decomposition),
    and A*B2 = sum_c u_c * v'_c * w_c^T,
    then C = sum_c u_c * [v_c, v'_c] * w_c^T (rank-2r decomposition).
    
    We need to find a low-rank decomposition for 3x3x3 matrix multiplication.
    The known minimum rank for 3x3x3 is 23 (from literature).
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    deadline = time.time() + 115
    
    def validate(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def schoolbook_decomposition():
        """Return the schoolbook rank n*m*p decomposition."""
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
    
    def find_3x3x3_decomposition(rank_target, max_restarts=10, timeout=30):
        """
        Find a rank-r decomposition for 3x3x3 matrix multiplication using optimization.
        Returns (U, V, W) as numpy arrays or None if not found.
        """
        n3, m3, p3 = 3, 3, 3
        start_time = time.time()
        
        # The tensor: T[i,j,k] = 1 for all valid (i,j,k) in 3x3x3
        # We need sum_c U[i*m3+j, c] * V[j*p3+k, c] * W[k*n3+i, c] = 1 for all (i,j,k)
        
        triples = [(i, j, k) for i in range(n3) for j in range(m3) for k in range(p3)]
        idx_U = np.array([i * m3 + j for i, j, k in triples])
        idx_V = np.array([j * p3 + k for i, j, k in triples])
        idx_W = np.array([k * n3 + i for i, j, k in triples])
        
        best_solution = None
        best_error = float('inf')
        
        for restart in range(max_restarts):
            if time.time() - start_time > timeout:
                break
            
            # Initialize with random values
            np.random.seed(random.randint(0, 2**31 - 1))
            U = np.random.uniform(-1, 1, (n3 * m3, rank_target))
            V = np.random.uniform(-1, 1, (m3 * p3, rank_target))
            W = np.random.uniform(-1, 1, (p3 * n3, rank_target))
            
            lr = 0.1
            for it in range(2000):
                if time.time() - start_time > timeout:
                    break
                
                U_sel = U[idx_U, :]
                V_sel = V[idx_V, :]
                W_sel = W[idx_W, :]
                
                S = np.sum(U_sel * V_sel * W_sel, axis=1)
                diff = S - 1.0
                error = np.sum(diff * diff)
                
                if error < 1e-12:
                    best_solution = (U.copy(), V.copy(), W.copy())
                    best_error = error
                    return best_solution
                
                if error < best_error:
                    best_error = error
                    best_solution = (U.copy(), V.copy(), W.copy())
                
                # Compute gradients
                term_U = 2.0 * diff[:, None] * V_sel * W_sel
                term_V = 2.0 * diff[:, None] * U_sel * W_sel
                term_W = 2.0 * diff[:, None] * U_sel * V_sel
                
                grad_U = np.zeros((n3 * m3, rank_target))
                grad_V = np.zeros((m3 * p3, rank_target))
                grad_W = np.zeros((p3 * n3, rank_target))
                
                np.add.at(grad_U, idx_U, term_U)
                np.add.at(grad_V, idx_V, term_V)
                np.add.at(grad_W, idx_W, term_W)
                
                # Adaptive learning rate
                grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if grad_norm > 1e-10:
                    step_size = lr / grad_norm
                else:
                    step_size = lr
                
                U -= step_size * grad_U
                V -= step_size * grad_V
                W -= step_size * grad_W
                
                lr *= 0.9995
        
        if best_error < 1e-8 and best_solution is not None:
            return best_solution
        return None
    
    def try_block_approach():
        """
        Try to find a rank-23 decomposition for 3x3x3, then construct rank-46 for 3x3x6.
        """
        # Try different target ranks for 3x3x3
        for r3 in [19, 20, 21, 22, 23, 24, 25]:
            if time.time() > deadline - 20:
                break
            
            result = find_3x3x3_decomposition(r3, max_restarts=5, timeout=15)
            if result is None:
                continue
            
            U3, V3, W3 = result
            
            # Construct 3x3x6 decomposition from 3x3x3 decomposition
            # B is 3x6, split into B1 (cols 0-2) and B2 (cols 3-5)
            # C = A * [B1, B2] = [A*B1, A*B2]
            
            rank6 = 2 * r3
            U6 = np.zeros((n * m, rank6))
            V6 = np.zeros((m * p, rank6))
            W6 = np.zeros((p * n, rank6))
            
            for c in range(r3):
                # First block (cols 0-2 of B, corresponding to k=0,1,2 in 3x3x3)
                U6[:, 2*c] = U3[:, c]
                for j in range(m):
                    for k in range(3):
                        V6[j * p + k, 2*c] = V3[j * p3 + k, c]
                for k in range(3):
                    for i in range(n):
                        W6[k * n + i, 2*c] = W3[k * n3 + i, c]
                
                # Second block (cols 3-5 of B, corresponding to k=0,1,2 in 3x3x3)
                U6[:, 2*c+1] = U3[:, c]
                for j in range(m):
                    for k in range(3):
                        V6[j * p + (k + 3), 2*c+1] = V3[j * p3 + k, c]
                for k in range(3):
                    for i in range(n):
                        W6[(k + 3) * n + i, 2*c+1] = W3[k * n3 + i, c]
            
            # Validate
            U6_list = U6.tolist()
            V6_list = V6.tolist()
            W6_list = W6.tolist()
            
            if validate(U6_list, V6_list, W6_list):
                return {"U": U6_list, "V": V6_list, "W": W6_list}
            
            # Try rounding to half-integers
            U6_r = np.round(U6 * 2) / 2
            V6_r = np.round(V6 * 2) / 2
            W6_r = np.round(W6 * 2) / 2
            if validate(U6_r.tolist(), V6_r.tolist(), W6_r.tolist()):
                return {"U": U6_r.tolist(), "V": V6_r.tolist(), "W": W6_r.tolist()}
        
        return None
    
    # Try the block approach
    result = try_block_approach()
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
