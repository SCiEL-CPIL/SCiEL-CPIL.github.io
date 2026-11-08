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
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = schoolbook_block(U, V, W, 0, range(n), range(m), range(p))
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Strategy 2: Try m=1+4, p=2+4
    if time.time() < deadline - 20:
        def try_split_m14_p24():
            rank = 2*1*2 + 2*1*4 + 2*4*2 + 2*14
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            col = 0
            col = schoolbook_block(U, V, W, col, [0,1], [0], [0,1])
            col = schoolbook_block(U, V, W, col, [0,1], [0], [2,3,4,5])
            col = schoolbook_block(U, V, W, col, [0,1], [1,2,3,4], [0,1])
            for j0, j1 in [(1,2), (3,4)]:
                for k0, k1 in [(2,3), (4,5)]:
                    col = strassen_2x2x2(U, V, W, col, 0, 1, j0, j1, k0, k1)
            return U, V, W, rank
        
        U2, V2, W2, rank2 = try_split_m14_p24()
        if validate(U2, V2, W2, rank2) and rank2 < best_rank:
            best_U, best_V, best_W, best_rank = U2, V2, W2, rank2
    
    # Strategy 3: Try m=2+3, p=2+4
    if time.time() < deadline - 15:
        def try_split_m23_p24():
            # 2x2x2: Strassen=7
            # 2x2x4: 2*Strassen=14
            # 2x3x2: schoolbook=12
            # 2x3x4: 2x3x2 + 2x3x2 = 24, or try better
            # Total: 7+14+12+24 = 57 (worse)
            # Try 2x3x4 with Strassen on 2x2x2 blocks:
            # j in {1,2}, k in {2,3}: Strassen=7; j in {1,2}, k in {4,5}: Strassen=7 => 14
            # So 2x3x4 = 14, total = 7+14+12+14 = 47!
            rank = 7 + 14 + 12 + 14
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            col = 0
            # j=0,1; k=0,1: Strassen
            col = strassen_2x2x2(U, V, W, col, 0, 1, 0, 1, 0, 1)
            # j=0,1; k=2,3,4,5: two Strassen blocks
            for k0, k1 in [(2,3), (4,5)]:
                col = strassen_2x2x2(U, V, W, col, 0, 1, 0, 1, k0, k1)
            # j=2,3,4; k=0,1: schoolbook
            col = schoolbook_block(U, V, W, col, [0,1], [2,3,4], [0,1])
            # j=2,3; k=2,3,4,5: two Strassen blocks
            for k0, k1 in [(2,3), (4,5)]:
                col = strassen_2x2x2(U, V, W, col, 0, 1, 2, 3, k0, k1)
            return U, V, W, rank
        
        U3, V3, W3, rank3 = try_split_m23_p24()
        if validate(U3, V3, W3, rank3) and rank3 < best_rank:
            best_U, best_V, best_W, best_rank = U3, V3, W3, rank3
    
    # Strategy 4: Try m=1+2+2, p=2+2+2
    if time.time() < deadline - 10:
        def try_split_m122_p222():
            # j=0: schoolbook with all k = 12
            # j=1,2: Strassen with k=0,1 -> 7; k=2,3 -> 7; k=4,5 -> 7 => 21
            # j=3,4: Strassen with k=0,1 -> 7; k=2,3 -> 7; k=4,5 -> 7 => 21
            # Total: 12+21+21 = 54 (same as before)
            return None
        
        # This won't beat 54, skip
    
    # Strategy 5: Try m=2+1+2, p=1+4+1
    if time.time() < deadline - 5:
        def try_split_m212_p141():
            # j=0,1; k=0: schoolbook=4
            # j=0,1; k=1,2,3,4: 2x2x4 = 14
            # j=0,1; k=5: schoolbook=4
            # j=2; k=0: schoolbook=2
            # j=2; k=1,2,3,4: schoolbook=8
            # j=2; k=5: schoolbook=2
            # j=3,4; k=0: schoolbook=4
            # j=3,4; k=1,2,3,4: 2x2x4 = 14
            # j=3,4; k=5: schoolbook=4
            # Total: 4+14+4+2+8+2+4+14+4 = 56 (worse)
            return None
    
    # Strategy 6: Try to reduce rank by exploiting structure
    # For 2x5x6, the tensor has a specific structure.
    # Let's try a different approach: use the fact that n=2 is small.
    # For each (j,k), the contribution to output is A[:,j]*B[j,k]
    # This means we can potentially share computations across j.
    
    # Try: group by k first
    if time.time() < deadline - 3:
        def try_group_by_k():
            # For each k, we need to compute sum_j A[:,j]*B[j,k]
            # This is like multiplying a 2x5 matrix by a 5x1 vector
            # R(2,5,1) = 10 (schoolbook)
            # But we can do better by sharing across k values.
            # Actually, let's try a different grouping.
            
            # Group j into pairs and use Strassen-like tricks
            # j=0,1 with k=0,1,2,3,4,5:
            #   Use Strassen for 2x2x2 blocks: (0,1)x(0,1), (0,1)x(2,3), (0,1)x(4,5) => 3*7=21
            # j=2,3 with k=0,1,2,3,4,5:
            #   Same: 21
            # j=4 with k=0,1,2,3,4,5:
            #   Schoolbook: 12
            # Total: 21+21+12 = 54 (same as before)
            return None
    
    # Convert to lists for the return interface
    U_list = [[int(U[i, c]) if abs(U[i, c] - round(U[i, c])) < 1e-9 else float(U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[int(V[i, c]) if abs(V[i, c] - round(V[i, c])) < 1e-9 else float(V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[int(W[i, c]) if abs(W[i, c] - round(W[i, c])) < 1e-9 else float(W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
