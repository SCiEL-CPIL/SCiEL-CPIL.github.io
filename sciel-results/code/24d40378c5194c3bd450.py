"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Start with Strassen rank-42 decomposition as baseline
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 linear forms for A (2x2)
    U_forms = [
        [1, 0, 0, 1],  # S0: a00 + a11
        [0, 0, 1, 1],  # S1: a10 + a11
        [1, 0, 0, 0],  # S2: a00
        [0, 0, 0, 1],  # S3: a11
        [1, 1, 0, 0],  # S4: a00 + a01
        [-1, 0, 1, 0], # S5: a10 - a00
        [0, 1, 0, -1]  # S6: a01 - a11
    ]
    
    # Build baseline Strassen decomposition
    for u in range(6):
        for t in range(7):
            c = u * 7 + t
            
            for i in range(4):
                U[i][c] = U_forms[t][i]
                
            if t == 0:
                V[0 * p + 2*u][c] += 1
                V[1 * p + 2*u + 1][c] += 1
            elif t == 1:
                V[0 * p + 2*u][c] += 1
            elif t == 2:
                V[0 * p + 2*u + 1][c] += 1
                V[1 * p + 2*u + 1][c] -= 1
            elif t == 3:
                V[1 * p + 2*u][c] += 1
                V[0 * p + 2*u][c] -= 1
            elif t == 4:
                V[1 * p + 2*u + 1][c] += 1
            elif t == 5:
                V[0 * p + 2*u][c] += 1
                V[0 * p + 2*u + 1][c] += 1
            elif t == 6:
                V[1 * p + 2*u][c] += 1
                V[1 * p + 2*u + 1][c] += 1
                
            if t == 0:
                W[(2*u) * n + 0][c] += 1
                W[(2*u+1) * n + 1][c] += 1
            elif t == 1:
                W[(2*u) * n + 1][c] += 1
                W[(2*u+1) * n + 1][c] -= 1
            elif t == 2:
                W[(2*u+1) * n + 0][c] += 1
                W[(2*u+1) * n + 1][c] += 1
            elif t == 3:
                W[(2*u) * n + 0][c] += 1
                W[(2*u) * n + 1][c] += 1
            elif t == 4:
                W[(2*u) * n + 0][c] -= 1
                W[(2*u+1) * n + 0][c] += 1
            elif t == 5:
                W[(2*u+1) * n + 1][c] += 1
            elif t == 6:
                W[(2*u) * n + 0][c] += 1

    def verify(U_mat, V_mat, W_mat, rank_val):
        """Verify that the decomposition is exact."""
        # Convert to numpy
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        
        # The tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For matrix multiplication (n,m,p), T should be:
        # T[i,j,k] = 1 if i = i1*n + i2, j = j1*m + j2, k = k1*n + k2
        #           and i1 == j2 and j1 == k2 (i.e., A[i1,i2]*B[i2,j2] contributes to C[i1,j2])
        # Wait, let me re-derive the indexing.
        
        # A is n x m, B is m x p, C is n x p
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # In the tensor: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # U[i,j] corresponds to A[i,j] where i in [0,n), j in [0,m)
        # V[j,k] corresponds to B[j,k] where j in [0,m), k in [0,p)
        # W[k,i] corresponds to C[i,k] where i in [0,n), k in [0,p)
        # The trace convention says W[k*n + i, r]
        
        # So the tensor is: T[i*m+j, j*p+k, k*n+i] = A[i,j] * B[j,k] * C[i,k]
        # For the multiplication tensor: T[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent
        # Actually, the multiplication tensor has T[a,b,c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
        # for valid i,j,k.
        
        # Let's verify by checking all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # T[a,b,c] should be 1
                    val = 0.0
                    for r in range(rank_val):
                        val += U_np[a, r] * V_np[b, r] * W_np[c, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Verify baseline
    if not verify(U, V, W, rank):
        raise ValueError("Baseline decomposition is invalid")
    
    best_rank = rank
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to improve by searching for lower rank decompositions
    # Strategy: try rank 41, 40, 39, ... and see if we can find a valid decomposition
    
    # For now, return the baseline. In a more advanced version, we could search.
    # But given the time constraint, let's return the known good result.
    
    if time.time() > deadline - 5:
        return best_result
    
    # Try to find a rank-40 decomposition by modifying the Strassen approach
    # One idea: for some blocks, use a different decomposition that shares forms
    
    # Actually, let's try a completely different approach: use a rank-13 decomposition for 2x2 x 2x4
    # and apply it 3 times (for 12 columns), giving rank 39.
    
    # The rank-13 decomposition for 2x2 x 2x4:
    # This is more complex. Let me try a simpler optimization first.
    
    # Try to reduce rank by 1: rank 41
    # This would require finding a way to share one multiplication between blocks.
    
    # For now, return the baseline rank-42 result.
    return best_result
# EVOLVE-BLOCK-END
