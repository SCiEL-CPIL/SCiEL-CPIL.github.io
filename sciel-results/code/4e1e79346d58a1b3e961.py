"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 140  # Safety margin
    
    def validate(U, V, W, rank):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def strassen_2x2x2(U, V, W, col, i0, i1, j0, j1, k0, k1):
        """Add Strassen decomposition for 2x2x2 block. Returns new col."""
        # P1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        U[i0*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        # P2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        U[i1*m+j0, col] += 1; U[i1*m+j1, col] += 1
        V[j0*p+k0, col] += 1
        W[k0*n+i1, col] += 1; W[k1*n+i1, col] -= 1
        col += 1
        # P3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
        U[i0*m+j0, col] += 1
        V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
        W[k1*n+i0, col] += 1; W[k1*n+i1, col] += 1
        col += 1
        # P4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
        U[i1*m+j1, col] += 1
        V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
        W[k0*n+i0, col] += 1; W[k0*n+i1, col] += 1
        col += 1
        # P5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        U[i0*m+j0, col] += 1; U[i0*m+j1, col] += 1
        V[j1*p+k1, col] += 1
        W[k0*n+i0, col] -= 1; W[k1*n+i0, col] += 1
        col += 1
        # P6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        U[i1*m+j0, col] += 1; U[i0*m+j0, col] -= 1
        V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
        W[k1*n+i1, col] += 1
        col += 1
        # P7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        U[i0*m+j1, col] += 1; U[i1*m+j1, col] -= 1
        V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
        W[k0*n+i0, col] += 1
        col += 1
        return col
    
    def schoolbook_block(U, V, W, col, i_range, j_range, k_range):
        """Add schoolbook terms for a block. Returns new col."""
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i*m+j, col] += 1
                    V[j*p+k, col] += 1
                    W[k*n+i, col] += 1
                    col += 1
        return col
    
    # Build the rank-54 decomposition
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for j0, j1 in [(0,1), (2,3)]:
        for k0, k1 in [(0,1), (2,3), (4,5)]:
            col = strassen_2x2x2(U, V, W, col, 0, 1, j0, j1, k0, k1)
    col = schoolbook_block(U, V, W, col, [0,1], [4], range(p))
    
    # Verify
    if not validate(U, V, W, rank):
        # Fallback to pure schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = schoolbook_block(U, V, W, 0, range(n), range(m), range(p))
    
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    # Try to reduce rank by finding linear dependencies
    # The tensor T is defined by T[i*m+j, j*p+k, k*n+i] = 1
    # We have 54 rank-1 terms. Check if we can eliminate some.
    
    # Approach: Build the "flattened" tensor and check rank
    # T_flat[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    # For our decomposition: sum_c U[a,c]*V[b,c]*W[c,c] = T[a,b,c]
    
    # Try: for each subset of columns, check if removing them still works
    # This is expensive, so try a smarter approach
    
    # Key insight: The tensor has structure. For fixed (i,k), the slice over (j,j')
    # is delta_{j,j'} (identity). This means the tensor is "diagonal" in j.
    
    # Let me try a different block decomposition that might give lower rank
    
    # Strategy: Try 2x3x3 blocks with potentially rank 17
    if time.time() < deadline - 30:
        def try_2x3x3_blocks():
            # Split m into {0,1,2} and {3,4}, p into {0,1,2} and {3,4,5}
            # 2x3x3 + 2x3x3 + 2x2x3 + 2x2x3
            # For 2x3x3: try rank 17 (if possible)
            # For 2x2x3: R(2,2,3) <= 7+4 = 11
            # Total: 17+17+11+11 = 56 (worse than 54)
            return None
        
        # Try m={0,1},{2,3,4}, p={0,1,2},{3,4,5}
        # 2x2x3 + 2x2x3 + 2x3x3 + 2x3x3
        # = 11+11+18+18 = 58 (worse)
        pass
    
    # Try: Can we do 2x5x6 with rank < 54?
    # Known: R(2,5,6) >= ? Lower bound is hard.
    # Let's try to find if rank 53 works by checking linear dependencies.
    
    # Build the system: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a multilinear system. Hard to solve directly.
    
    # Alternative: try a completely different decomposition
    # Use the fact that R(2,5,6) <= R(2,5,3)*2
    # R(2,5,3) <= R(2,2,3) + R(2,3,3) = 11 + 18 = 29
    # Total: 29*2 = 58 (worse)
    
    # Or R(2,5,6) <= R(2,4,6) + R(2,1,6)
    # R(2,4,6) <= R(2,2,3)*4 = 44, R(2,1,6) = 12, total 56 (worse)
    
    # Let me try to verify if the rank-54 solution is optimal or if we can do 53
    
    # Quick check: try to see if any single column can be removed
    # and compensated by adjusting others. This is complex.
    
    # For now, return the rank-54 solution
    U_list = [[int(U[i, c]) if abs(U[i, c] - round(U[i, c])) < 1e-9 else float(U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[int(V[i, c]) if abs(V[i, c] - round(V[i, c])) < 1e-9 else float(V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[int(W[i, c]) if abs(W[i, c] - round(W[i, c])) < 1e-9 else float(W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
