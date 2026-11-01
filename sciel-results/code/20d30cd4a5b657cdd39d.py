"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 2, 9
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    def schoolbook():
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
    
    def build_rank32():
        """Build the rank-32 decomposition using Strassen blocks."""
        U = [[0] * 32 for _ in range(n * m)]
        V = [[0] * 32 for _ in range(m * p)]
        W = [[0] * 32 for _ in range(p * n)]
        
        # Strassen 2x2x2 block entries
        strassen_u = [
            [1, 0, 1, 0],   # M1
            [0, 0, 1, 1],   # M2
            [1, 0, 0, 0],   # M3
            [0, 0, 0, 1],   # M4
            [1, 1, 0, 0],   # M5
            [-1, 0, 1, 0],  # M6
            [0, 1, 0, -1],  # M7
        ]
        strassen_v = [
            [1, 0, 0, 1],   # M1
            [1, 0, 0, 0],   # M2
            [0, 0, 1, -1],  # M3
            [-1, 1, 0, 0],  # M4
            [0, 0, 0, 1],   # M5
            [1, 0, 1, 0],   # M6
            [0, 1, 0, 1],   # M7
        ]
        strassen_w = [
            [1, 0, 0, 1],   # M1
            [0, 0, 1, 0],   # M2
            [0, 1, 0, 0],   # M3
            [-1, 1, 0, 0],  # M4
            [1, 0, 0, 0],   # M5
            [0, 0, 1, 0],   # M6
            [0, 0, 0, 1],   # M7
        ]
        
        rank_offset = 0
        # 4 Strassen blocks for k=0,1; k=2,3; k=4,5; k=6,7
        for block_idx in range(4):
            k_offset = block_idx * 2
            for c in range(7):
                col = rank_offset + c
                for idx in range(4):
                    U[idx][col] = strassen_u[c][idx]
                for idx in range(4):
                    j = idx // 2
                    k_local = idx % 2
                    k_global = k_offset + k_local
                    V[j * p + k_global][col] = strassen_v[c][idx]
                for idx in range(4):
                    k_local = idx // 2
                    i = idx % 2
                    k_global = k_offset + k_local
                    W[k_global * n + i][col] = strassen_w[c][idx]
            rank_offset += 7
        
        # 1 schoolbook block for k=8
        for c in range(4):
            col = rank_offset + c
            i = c // 2
            j = c % 2
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
        rank_offset += 4
        
        return U, V, W
    
    U32, V32, W32 = build_rank32()
    
    if verify(U32, V32, W32):
        # Try to find rank-31 via numerical optimization
        if time.time() - start_time < deadline - 30:
            try:
                # Convert to numpy for optimization
                U_arr = np.array(U32, dtype=float)
                V_arr = np.array(V32, dtype=float)
                W_arr = np.array(W32, dtype=float)
                
                # Try removing one column and adjusting
                # We'll try to find if we can merge two columns
                # by finding a linear dependency
                
                # The tensor equation: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
                # If we can find c1, c2 such that we can combine them...
                
                # Alternative: try a direct rank-31 construction
                # Use 3 Strassen blocks (21) + a 2x2x3 block
                
                # For 2x2x3: can we do better than schoolbook (6)?
                # The 2x2x3 tensor has 12 nonzero entries.
                # Lower bound: max(2*3, 2*3, 2*3) = 6, but we might do better.
                # Actually for 2x2x3, the rank is known to be 7 (same as 2x2x2 + 1)
                # or possibly less. Let's try rank 6 for 2x2x3.
                
                # Try: 3 Strassen (21) + rank-6 for 2x2x3 = 27? No, that's not right.
                # 3 blocks of 2x2x2 (21) + 1 block of 2x2x3
                # 2x2x3 schoolbook = 6, total = 27
                # But we need to check if 2x2x3 can be done in rank < 6
                
                # Actually, let's try a completely different grouping:
                # 2x2x9 = 2x2x(3+3+3)
                # If 2x2x3 can be done in rank 5, total = 15
                # If 2x2x3 can be done in rank 6, total = 18
                
                # Let me try to construct a rank-6 decomposition for 2x2x3
                # T[i,j,k] = 1 for i,j in {0,1}, k in {0,1,2}
                
                # Try using the fact that 2x2x3 is a "slice" of 2x2x3
                # We can use:
                # M1 = A00*B00, M2 = A00*B01, M3 = A00*B02
                # M4 = A10*B00, M5 = A10*B01, M6 = A10*B02
                # M7 = A11*B00, M8 = A11*B01, M9 = A11*B02
                # But that's rank 9.
                
                # Better: use Strassen-like for 2x2x2 part + 1 for the extra column
                # 2x2x2 in rank 7 + 2x2x1 in rank 4 = 11? No, that's worse.
                
                # Let me just try numerical search for rank 31
                from scipy.optimize import minimize
                
                # Flatten the rank-32 solution and try to zero out one column
                # by adjusting the others
                
                # Actually, let's try a smarter approach:
                # The tensor is T = sum_c u_c ⊗ v_c ⊗ w_c
                # where u_c is in R^4, v_c is in R^18, w_c is in R^18
                
                # For rank 31, we need 31 such terms.
                # Start from rank 32 and try to eliminate one term.
                
                # Pick the last column (schoolbook term for i=0,j=0,k=8)
                # and try to absorb it into the Strassen terms.
                
                # This is complex. Let me just return rank 32 for now
                # and try a simple perturbation search.
                
                # Try: can we do 2x2x9 with rank 30?
                # 2x2x9 = 2x2x(2+2+2+2+1) -> 4*7 + 4 = 32 (current)
                # 2x2x9 = 2x2x(3+3+3) -> 3*R(2x2x3)
                # If R(2x2x3) = 7, total = 21
                # If R(2x2x3) = 6, total = 18
                # If R(2x2x3) = 5, total = 15
                
                # Let me try to construct 2x2x3 in rank 6
                # This would give total rank 18, which is much better!
                
                def build_2x2x3_rank6():
                    """Try to build a rank-6 decomposition for 2x2x3."""
                    # T[i,j,k] = 1 for i,j in {0,1}, k in {0,1,2}
                    # U: 4x6, V: 6x6 (2*3=6), W: 6x4 (3*2=6)
                    
                    # Use a combination of Strassen and schoolbook
                    # For k=0,1: use Strassen (rank 7) -> too many
                    # Instead, try a custom decomposition
                    
                    # Let's try:
                    # M1 = A00*(B00+B01+B02)
                    # M2 = A11*(B00+B01+B02)  
                    # M3 = (A00+A11)*(B00-B01)
                    # M4 = (A00+A11)*(B01-B02)
                    # M5 = A00*(B00-B01)
                    # M6 = A11*(B01-B02)
                    
                    # Then:
                    # C00 = M1 - M3 + M5
                    # C01 = M1 + M3 - M4 - M6 + M5  -- this is getting complicated
                    
                    # Let me just try a direct numerical approach
                    # Initialize with random values and optimize
                    
                    np.random.seed(42)
                    U_try = np.random.randn(4, 6) * 0.1
                    V_try = np.random.randn(6, 6) * 0.1
                    W_try = np.random.randn(6, 4) * 0.1
                    
                    def loss(params):
                        U_p = params[0:24].reshape(4, 6)
                        V_p = params[24:60].reshape(6, 6)
                        W_p = params[60:84].reshape(6, 4)
                        
                        # Compute tensor: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
                        # i in {0,1}, j in {0,1}, k in {0,1,2}
                        total_loss = 0
                        for i in range(2):
                            for j in range(2):
                                for k in range(3):
                                    val = 0
                                    for c in range(6):
                                        val += U_p[i*2+j, c] * V_p[j*3+k, c] * W_p[k*2+i, c]
                                    total_loss += (val - 1) ** 2
                        return total_loss
                    
                    params0 = np.concatenate([U_try.flatten(), V_try.flatten(), W_try.flatten()])
                    result = minimize(loss, params0, method='Nelder-Mead', 
                                     options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
                    
                    if result.fun < 1e-8:
                        U_p = result.x[0:24].reshape(4, 6)
                        V_p = result.x[24:60].reshape(6, 6)
                        W_p = result.x[60:84].reshape(6, 4)
                        return U_p.tolist(), V_p.tolist(), W_p.tolist()
                    
                    return None
                
                U23, V23, W23 = build_2x2x3_rank6()
                
                if U23 is not None:
                    # Build full 2x2x9 from three 2x2x3 blocks
                    U_full = [[0] * 18 for _ in range(4)]
                    V_full = [[0] * 18 for _ in range(18)]
                    W_full = [[0] * 18 for _ in range(18)]
                    
                    for block in range(3):
                        k_offset = block * 3
                        for c in range(6):
                            col = block * 6 + c
                            for idx in range(4):
                                U_full[idx][col] = U23[idx][c]
                            for idx in range(6):
                                j = idx // 3
                                k_local = idx % 3
                                k_global = k_offset + k_local
                                V_full[j * 9 + k_global][col] = V23[idx][c]
                            for idx in range(6):
                                k_local = idx // 2
                                i = idx % 2
                                k_global = k_offset + k_local
                                W_full[k_global * 2 + i][col] = W23[idx][c]
                    
                    if verify(U_full, V_full, W_full):
                        return {"U": U_full, "V": V_full, "W": W_full}
                        
            except Exception as e:
                pass
        
        return {"U": U32, "V": V32, "W": W32}
    
    return schoolbook()
# EVOLVE-BLOCK-END
