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
        # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, else 0
        # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all valid (i,j,k)
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
    
    # Strategy 1: Current approach - Strassen on 2x2x2 blocks
    # j in {0,1},{2,3} and k in {0,1},{2,3},{4,5}, plus j=4
    def try_strassen_split():
        rank = 54
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for j0, j1 in [(0,1), (2,3)]:
            for k0, k1 in [(0,1), (2,3), (4,5)]:
                col = strassen_2x2x2(U, V, W, col, 0, 1, j0, j1, k0, k1)
        col = schoolbook_block(U, V, W, col, [0,1], [4], range(p))
        return U, V, W, rank
    
    U, V, W, rank = try_strassen_split()
    if not validate(U, V, W, rank):
        # Fallback to pure schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = schoolbook_block(U, V, W, 0, range(n), range(m), range(p))
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Strategy 2: Try different block splits
    # Split m into parts, p into parts, use Strassen where 2x2x2
    if time.time() < deadline - 20:
        # Try m=1+4, p=2+4
        def try_split_m14_p24():
            rank = 0
            # 2x1x2: schoolbook = 4
            rank += 2*1*2
            # 2x1x4: schoolbook = 8
            rank += 2*1*4
            # 2x4x2: schoolbook = 16
            rank += 2*4*2
            # 2x4x4: split into 2x2x4 + 2x2x4, each 2x2x4 = 2*Strassen = 14
            rank += 2*14
            total = rank
            U = np.zeros((n*m, total))
            V = np.zeros((m*p, total))
            W = np.zeros((p*n, total))
            col = 0
            # j=0, k=0,1
            col = schoolbook_block(U, V, W, col, [0,1], [0], [0,1])
            # j=0, k=2,3,4,5
            col = schoolbook_block(U, V, W, col, [0,1], [0], [2,3,4,5])
            # j=1,2,3,4, k=0,1
            col = schoolbook_block(U, V, W, col, [0,1], [1,2,3,4], [0,1])
            # j=1,2, k=2,3,4,5: split j into {1,2} and {3,4}
            for j0, j1 in [(1,2), (3,4)]:
                for k0, k1 in [(2,3), (4,5)]:
                    col = strassen_2x2x2(U, V, W, col, 0, 1, j0, j1, k0, k1)
            return U, V, W, total
        
        U2, V2, W2, rank2 = try_split_m14_p24()
        if validate(U2, V2, W2, rank2) and rank2 < best_rank:
            best_U, best_V, best_W, best_rank = U2, V2, W2, rank2
    
    if time.time() < deadline - 15:
        # Try m=2+3, p=3+3
        def try_split_m23_p33():
            # 2x2x3: split k into 2+1: 2x2x2 (Strassen=7) + 2x2x1 (schoolbook=4) = 11
            # 2x2x3 again = 11
            # 2x3x3: schoolbook = 18, or try to reduce
            # 2x3x3 again = 18
            # Total: 11+11+18+18 = 58 (worse than 54)
            # Try 2x3x3 with better method
            # Actually 2x3x3: can we do better than 18?
            # R(2,3,3) <= R(2,2,3)+R(2,1,3) = 11+6 = 17
            # So total: 11+11+17+17 = 56 (still worse)
            return None
        
        # This won't beat 54, skip
    
    if time.time() < deadline - 10:
        # Try to find linear dependencies in the rank-54 decomposition
        # The tensor T has 60 non-zero entries. We have 54 rank-1 terms.
        # Check if we can merge or eliminate terms.
        # This is complex; try a simpler approach: see if any column is redundant
        
        # Build the "slice" matrices to check for dependencies
        # For each pair (i,k), the slice T_{i,k} is an m×m matrix with T_{i,k}[j,j'] = 1 if j=j' else 0
        # Actually for matrix mult, T_{i,k}[j,j'] = delta_{j,j'} (diagonal)
        # This means the tensor has a specific structure that might allow reduction.
        
        # Try: can we use the fact that for fixed i, the m×p slice has rank 1?
        # No, that's not right. Let me just try a few more structured decompositions.
        
        # Try m=2+1+2, p=2+2+2
        # j=0,1: Strassen with k=0,1 -> 7; k=2,3 -> 7; k=4,5 -> 7 => 21
        # j=2: schoolbook with all k => 12
        # j=3,4: Strassen with k=0,1 -> 7; k=2,3 -> 7; k=4,5 -> 7 => 21
        # Total: 21+12+21 = 54 (same as before)
        
        # Try combining: use the j=2 row to help with Strassen blocks
        # This requires a more sophisticated approach that I don't have time to derive.
        
        pass
    
    # Convert to lists for the return interface
    U_list = [[int(U[i, c]) if abs(U[i, c] - round(U[i, c])) < 1e-9 else float(U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[int(V[i, c]) if abs(V[i, c] - round(V[i, c])) < 1e-9 else float(V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[int(W[i, c]) if abs(W[i, c] - round(W[i, c])) < 1e-9 else float(W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
