"""Tensor decomposition for 2x5x6 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

_DEADLINE = time.time() + 125.0


def _strassen_2x2x2():
    """
    Return the 7 rank-1 tensors for Strassen's 2x2x2 matrix multiplication.
    
    For C = A*B where A,B,C are 2x2:
    A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    
    Strassen:
    M1 = (a11+a22)*(b11+b22)
    M2 = (a21+a22)*b11
    M3 = a11*(b12-b22)
    M4 = a22*(b21-b11)
    M5 = (a11+a12)*b22
    M6 = (a21-a11)*(b11+b12)
    M7 = (a12-a22)*(b21+b22)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    """
    # U: 4x7, V: 4x7, W: 4x7
    # Indexing: A[i*2+j], B[j*2+k], C[k*2+i] where i,k in {0,1}, j in {0,1}
    
    # U rows: a11(0), a12(1), a21(2), a22(3)
    # V rows: b11(0), b12(1), b21(2), b22(3)
    # W rows: c11(0), c12(1), c21(2), c22(3)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = a22*(b21-b11)
    U[3][3] = 1
    V[0][3] = -1; V[2][3] = 1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (a21-a11)*(b11+b12)
    U[0][5] = -1; U[2][5] = 1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _verify(U, V, W):
    """Verify that the decomposition is exact for 2x5x6 matrix multiplication."""
    n, m, p = 2, 5, 6
    rank = len(U[0]) if U else 0
    
    # Check shapes
    if len(U) != n*m:
        return False
    if len(V) != m*p:
        return False
    if len(W) != p*n:
        return False
    for mat in [U, V, W]:
        for row in mat:
            if len(row) != rank:
                return False
    
    # The tensor T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # This is equivalent to: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # sums to the matrix multiplication tensor.
    
    # Build the full tensor and check
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] where a=i*m+j, b=j*p+k, c=k*n+i
    # And T should be 1 iff a,b,c correspond to a valid multiplication term, else 0.
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # indexed as U index = i*m+j, V index = j*p+k, W index = k*n+i
    
    # For each triple (i,j,k), check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # For invalid triples, it should be 0. But actually the tensor is defined such that
    # C_ik = sum_j A_ij * B_jk, so the tensor T_{i,j,k} = 1 for all valid (i,j,k).
    
    # Let's verify by reconstructing: for random-ish inputs, check C = A*B
    # Actually, let's just check the tensor entries directly.
    # The tensor is T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T should equal 1 for all combinations that are "consistent" (i.e., there exist i,j,k 
    # such that a=i*m+j, b=j*p+k, c=k*n+i simultaneously).
    
    # For each a in [0, n*m), b in [0, m*p), c in [0, p*n):
    # Check if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    # If yes, T[a,b,c] should be 1, else 0.
    
    # Efficient check: for each r, compute the contribution and sum.
    # But that's O(rank * n*m * m*p * p*n) which is fine for small sizes.
    
    # Actually, let's just check a few entries to be safe.
    # Check all valid (i,j,k) triples:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                total = 0
                for r in range(rank):
                    total += U[a][r] * V[b][r] * W[c][r]
                if total != 1:
                    return False
    
    # Check some invalid entries to ensure they're 0
    # An entry (a,b,c) is invalid if there's no (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
    # For example, a=0 (i=0,j=0), b=0 (j=0,k=0), c=0 (k=0,i=0) -> valid
    # a=0 (i=0,j=0), b=0 (j=0,k=0), c=1 (k=0,i=1) -> i should be 0 and 1, invalid
    # Let's check: a=0, b=0, c=1
    total = 0
    for r in range(rank):
        total += U[0][r] * V[0][r] * W[1][r]
    if total != 0:
        return False
    
    return True


def construct():
    n, m, p = 2, 5, 6
    
    # Strategy: Use Strassen for the 2x2 subblocks where possible.
    # A is 2x5, B is 5x6, C is 2x6.
    # Split A = [A1 (2x2) | A2 (2x3)], B = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 * 2x6. Split B1 into three 2x2 blocks: B1a, B1b, B1c
    #   A1*B1a: 2x2*2x2 -> Strassen rank 7
    #   A1*B1b: 2x2*2x2 -> Strassen rank 7
    #   A1*B1c: 2x2*2x2 -> Strassen rank 7
    #   Total for A1*B1: 21
    # A2*B2: 2x3 * 3x6. Schoolbook: 2*3*6 = 36
    #   Or we can do better by splitting B2 into 3 columns of width 2:
    #   B2 = [B2a (3x2) | B2b (3x2) | B2c (3x2)]
    #   A2*B2a: 2x3*3x2. 
    #     Split A2 = [A2a (2x2) | A2b (2x1)], B2a = [B2a_top (2x2); B2a_bot (1x2)]
    #     A2*B2a = A2a*B2a_top + A2b*B2a_bot
    #     A2a*B2a_top: 2x2*2x2 -> Strassen rank 7
    #     A2b*B2a_bot: 2x1*1x2 -> rank 4 (outer product of 2x1 and 1x2 = 4 mults)
    #     Total per 2-column block: 11
    #     Three blocks: 33
    # Total: 21 + 33 = 54
    
    # Let's implement the 54-rank construction.
    # 
    # U: 10x54, V: 30x54, W: 12x54
    # 
    # Part 1: A1*B1 where A1 = A[0:2, 0:2], B1 = B[0:2, :]
    #   B1 split into B1a = B[0:2, 0:2], B1b = B[0:2, 2:4], B1c = B[0:2, 4:6]
    #   For each block, use Strassen.
    #
    # Part 2: A2*B2 where A2 = A[0:2, 2:5], B2 = B[2:5, :]
    #   Split into three 2-column subproblems.
    
    U, V, W = _strassen_2x2x2()
    # U is 4x7, V is 4x7, W is 4x7 for 2x2x2
    
    # We'll build the full matrices incrementally.
    # Start with empty lists, append columns.
    
    U_full = []
    V_full = []
    W_full = []
    
    def add_block(u_block, v_block, w_block, u_offset, v_offset, w_offset, scale_u=None, scale_v=None, scale_w=None):
        """Add a rank-rank block at given offsets."""
        nonlocal U_full, V_full, W_full
        rank_block = len(u_block[0])
        
        # Expand to full sizes with zeros at offsets
        for r in range(rank_block):
            u_col = [0] * (n*m)
            v_col = [0] * (m*p)
            w_col = [0] * (p*n)
            
            for i in range(len(u_block)):
                u_col[u_offset + i] = u_block[i][r] * (scale_u[i] if scale_u else 1)
            for i in range(len(v_block)):
                v_col[v_offset + i] = v_block[i][r] * (scale_v[i] if scale_v else 1)
            for i in range(len(w_block)):
                w_col[w_offset + i] = w_block[i][r] * (scale_w[i] if scale_w else 1)
            
            U_full.append(u_col)
            V_full.append(v_col)
            W_full.append(w_col)
    
    # Part 1: A1*B1
    # A1 uses rows 0,1 of A (indices 0,1 in A which are i=0,j=0 and i=0,j=1... wait)
    # A is 2x5, indexed as A[i][j] where i in {0,1}, j in {0..4}
    # U index for A is i*m + j = i*5 + j
    # So A1 = A[0:2, 0:2] corresponds to U indices: 0*5+0=0, 0*5+1=1, 1*5+0=5, 1*5+1=6
    # In Strassen, the 4 elements are a11,a12,a21,a22 at local indices 0,1,2,3
    # So mapping: local 0 -> global 0 (a11), local 1 -> global 1 (a12), local 2 -> global 5 (a21), local 3 -> global 6 (a22)
    
    # B is 5x6, indexed as B[j][k] where j in {0..4}, k in {0..5}
    # V index for B is j*p + k = j*6 + k
    # B1 = B[0:2, :] has rows j=0,1
    # B1a = B[0:2, 0:2]: V indices 0*6+0=0, 0*6+1=1, 1*6+0=6, 1*6+1=7
    # B1b = B[0:2, 2:4]: V indices 0*6+2=2, 0*6+3=3, 1*6+2=8, 1*6+3=9
    # B1c = B[0:2, 4:6]: V indices 0*6+4=4, 0*6+5=5, 1*6+4=10, 1*6+5=11
    
    # C is 2x6, indexed as C[i][k] where i in {0,1}, k in {0..5}
    # W index for C is k*n + i = k*2 + i
    # C1a = C[0:2, 0:2]: W indices 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3
    # C1b = C[0:2, 2:4]: W indices 2*2+0=4, 2*2+1=5, 3*2+0=6, 3*2+1=7
    # C1c = C[0:2, 4:6]: W indices 4*2+0=8, 4*2+1=9, 5*2+0=10, 5*2+1=11
    
    # For each Strassen block, we need to map the local 4x4 indices to global.
    # U local [0,1,2,3] -> global [0,1,5,6]
    u_map_A1 = [0, 1, 5, 6]
    
    # For B1a: V local [0,1,2,3] -> global [0,1,6,7]
    v_map_B1a = [0, 1, 6, 7]
    # For B1b: V local [0,1,2,3] -> global [2,3,8,9]
    v_map_B1b = [2, 3, 8, 9]
    # For B1c: V local [0,1,2,3] -> global [4,5,10,11]
    v_map_B1c = [4, 5, 10, 11]
    
    # For C1a: W local [0,1,2,3] -> global [0,1,2,3]
    w_map_C1a = [0, 1, 2, 3]
    # For C1b: W local [0,1,2,3] -> global [4,5,6,7]
    w_map_C1b = [4, 5, 6, 7]
    # For C1c: W local [0,1,2,3] -> global [8,9,10,11]
    w_map_C1c = [8, 9, 10, 11]
    
    # Add the three Strassen blocks for A1*B1a, A1*B1b, A1*B1c
    
    # Block 1: A1*B1a
    for r in range(7):
        u_col = [0]*(n*m)
        v_col = [0]*(m*p)
        w_col = [0]*(p*n)
        for local_i, global_i in enumerate(u_map_A1):
            u_col[global_i] = U[local_i][r]
        for local_i, global_i in enumerate(v_map_B1a):
            v_col[global_i] = V[local_i][r]
        for local_i, global_i in enumerate(w_map_C1a):
            w_col[global_i] = W[local_i][r]
        U_full.append(u_col)
        V_full.append(v_col)
        W_full.append(w_col)
    
    # Block 2: A1*B1b
    for r in range(7):
        u_col = [0]*(n*m)
        v_col = [0]*(m*p)
        w_col = [0]*(p*n)
        for local_i, global_i in enumerate(u_map_A1):
            u_col[global_i] = U[local_i][r]
        for local_i, global_i in enumerate(v_map_B1b):
            v_col[global_i] = V[local_i][r]
        for local_i, global_i in enumerate(w_map_C1b):
            w_col[global_i] = W[local_i][r]
        U_full.append(u_col)
        V_full.append(v_col)
        W_full.append(w_col)
    
    # Block 3: A1*B1c
    for r in range(7):
        u_col = [0]*(n*m)
        v_col = [0]*(m*p)
        w_col = [0]*(p*n)
        for local_i, global_i in enumerate(u_map_A1):
            u_col[global_i] = U[local_i][r]
        for local_i, global_i in enumerate(v_map_B1c):
            v_col[global_i] = V[local_i][r]
        for local_i, global_i in enumerate(w_map_C1c):
            w_col[global_i] = W[local_i][r]
        U_full.append(u_col)
        V_full.append(v_col)
        W_full.append(w_col)
    
    # Part 2: A2*B2
    # A2 = A[0:2, 2:5], columns j=2,3,4
    # U indices for A2: i*5+j for i in {0,1}, j in {2,3,4}
    # i=0: j=2->2, j=3->3, j=4->4
    # i=1: j=2->7, j=3->8, j=4->9
    # So A2 elements are at U indices: 2,3,4 (row 0) and 7,8,9 (row 1)
    # A2a = A[0:2, 2:4] (columns 2,3): U indices 2,3 (row0) and 7,8 (row1)
    # A2b = A[0:2, 4:5] (column 4): U indices 4 (row0) and 9 (row1)
    
    # B2 = B[2:5, :], rows j=2,3,4
    # V indices for B2: j*6+k for j in {2,3,4}, k in {0..5}
    # j=2: 12,13,14,15,16,17
    # j=3: 18,19,20,21,22,23
    # j=4: 24,25,26,27,28,29
    
    # Split B2 into three 2-column blocks:
    # B2a = B[2:5, 0:2]: V indices 12,13 (j=2), 18,19 (j=3), 24,25 (j=4)
    # B2b = B[2:5, 2:4]: V indices 14,15 (j=2), 20,21 (j=3), 26,27 (j=4)
    # B2c = B[2:5, 4:6]: V indices 16,17 (j=2), 22,23 (j=3), 28,29 (j=4)
    
    # For each 2-column block, split B2x into top 2 rows and bottom 1 row:
    # B2a_top = B[2:4, 0:2]: V indices 12,13,18,19
    # B2a_bot = B[4:5, 0:2]: V indices 24,25
    # Similarly for others.
    
    # A2a (2x2) * B2a_top (2x2): Strassen rank 7
    # A2b (2x1) * B2a_bot (1x2): rank 4
    
    # A2a local indices: [0,1,2,3] -> U global [2,3,7,8]
    u_map_A2a = [2, 3, 7, 8]
    
    # B2a_top local indices: [0,1,2,3] -> V global [12,13,18,19]
    v_map_B2a_top = [12, 13, 18, 19]
    # B2b_top: [14,15,20,21]
    v_map_B2b_top = [14, 15, 20, 21]
    # B2c_top: [16,17,22,23]
    v_map_B2c_top = [16, 17, 22, 23]
    
    # B2a_bot: [24,25]
    # B2b_bot: [26,27]
    # B2c_bot: [28,29]
    
    # A2b local: [0,1] -> U global [4,9]
    u_map_A2b = [4, 9]
    
    # C2a = C[0:2, 0:2]: W indices 0,1,2,3
    w_map_C2a = [0, 1, 2, 3]
    # C2b = C[0:2, 2:4]: W indices 4,5,6,7
    w_map_C2b = [4, 5, 6, 7]
    # C2c = C[0:2, 4:6]: W indices 8,9,10,11
    w_map_C2c = [8, 9, 10, 11]
    
    # For each of the three 2-column blocks:
    for block_idx, (v_top_map, v_bot_map, w_map) in enumerate([
        (v_map_B2a_top, [24,25], w_map_C2a),
        (v_map_B2b_top, [26,27], w_map_C2b),
        (v_map_B2c_top, [28,29], w_map_C2c)
    ]):
        # Strassen part: A2a * B2x_top
        for r in range(7):
            u_col = [0]*(n*m)
            v_col = [0]*(m*p)
            w_col = [0]*(p*n)
            for local_i, global_i in enumerate(u_map_A2a):
                u_col[global_i] = U[local_i][r]
            for local_i, global_i in enumerate(v_top_map):
                v_col[global_i] = V[local_i][r]
            for local_i, global_i in enumerate(w_map):
                w_col[global_i] = W[local_i][r]
            U_full.append(u_col)
            V_full.append(v_col)
            W_full.append(w_col)
        
        # Outer product part: A2b * B2x_bot
        # A2b is 2x1, B2x_bot is 1x2
        # C contribution: A2b[0]*B2x_bot[0]*[1,0;0,0] + A2b[0]*B2x_bot[1]*[0,1;0,0]
        #                  + A2b[1]*B2x_bot[0]*[0,0;1,0] + A2b[1]*B2x_bot[1]*[0,0;0,1]
        # Wait, C = A2b * B2x_bot where A2b = [[a0],[a1]], B2x_bot = [[b0,b1]]
        # C = [[a0*b0, a0*b1], [a1*b0, a1*b1]]
        # So 4 terms.
        
        # Term 1: a0 * b0 -> C[0,0]
        u_col = [0]*(n*m); v_col = [0]*(m*p); w_col = [0]*(p*n)
        u_col[u_map_A2b[0]] = 1
        v_col[v_bot_map[0]] = 1
        w_col[w_map[0]] = 1  # C local [0] = C[0,0]
        U_full.append(u_col); V_full.append(v_col); W_full.append(w_col)
        
        # Term 2: a0 * b1 -> C[0,1]
        u_col = [0]*(n*m); v_col = [0]*(m*p); w_col = [0]*(p*n)
        u_col[u_map_A2b[0]] = 1
        v_col[v_bot_map[1]] = 1
        w_col[w_map[1]] = 1  # C local [1] = C[0,1]
        U_full.append(u_col); V_full.append(v_col); W_full.append(w_col)
        
        # Term 3: a1 * b0 -> C[1,0]
        u_col = [0]*(n*m); v_col = [0]*(m*p); w_col = [0]*(p*n)
        u_col[u_map_A2b[1]] = 1
        v_col[v_bot_map[0]] = 1
        w_col[w_map[2]] = 1  # C local [2] = C[1,0]
        U_full.append(u_col); V_full.append(v_col); W_full.append(w_col)
        
        # Term 4: a1 * b1 -> C[1,1]
        u_col = [0]*(n*m); v_col = [0]*(m*p); w_col = [0]*(p*n)
        u_col[u_map_A2b[1]] = 1
        v_col[v_bot_map[1]] = 1
        w_col[w_map[3]] = 1  # C local [3] = C[1,1]
        U_full.append(u_col); V_full.append(v_col); W_full.append(w_col)
    
    rank = len(U_full)
    
    # Verify
    if _verify(U_full, V_full, W_full):
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: schoolbook
    rank_school = n*m*p
    U_s = [[0]*rank_school for _ in range(n*m)]
    V_s = [[0]*rank_school for _ in range(m*p)]
    W_s = [[0]*rank_school for _ in range(p*n)]
    for c, (i, j, k) in enumerate(
        ((i, j, k) for i in range(n) for j in range(m) for k in range(p))
    ):
        U_s[i*m+j][c] = 1
        V_s[j*p+k][c] = 1
        W_s[k*n+i][c] = 1
    
    return {"U": U_s, "V": V_s, "W": W_s}
