import numpy as np
import time
from fractions import Fraction
from itertools import product, combinations_with_replacement

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def to_lists(U, V, W):
        U_list = [[round(float(x), 10) for x in row] for row in U]
        V_list = [[round(float(x), 10) for x in row] for row in V]
        W_list = [[round(float(x), 10) for x in row] for row in W]
        return U_list, V_list, W_list

    # Correct rank-15 construction using block partitioning
    # A is 2x3, B is 3x4, C is 2x4
    # Split A into [A1(2x2) A2(2x1)] and B into [B1(2x4); B2(1x4)]
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 * 2x4 -> use Strassen-like (rank 7 for 2x2x2, but here 2x2x4)
    # Actually let's use a cleaner decomposition
    
    def build_rank_15():
        rank = 15
        U = [[0.0] * rank for _ in range(n * m)]  # 6 rows: A[i,j]
        V = [[0.0] * rank for _ in range(m * p)]  # 12 rows: B[j,k]
        W = [[0.0] * rank for _ in range(p * n)]  # 8 rows: C[k,i] -> index k*n+i
        
        def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
            U[u_idx][col] += u_val
            V[v_idx][col] += v_val
            W[w_idx][col] += w_val

        # Block 1: A[0:2, 0:2] * B[0:2, 0:4] -> C[0:2, 0:4]
        # This is a 2x2 * 2x4 multiplication
        # We can decompose this as two 2x2*2x2 multiplications (columns 0-1 and 2-3)
        # Or use a rank-7 decomposition for 2x2x4
        
        # For 2x2 * 2x4, we can use:
        # Split B into B_left (2x2) and B_right (2x2)
        # C_left = A * B_left (rank 7 via Strassen)
        # C_right = A * B_right (rank 7 via Strassen)
        # Total: 14 for this block
        
        # Strassen for 2x2 * 2x2:
        # M1 = (A11+A22)*(B11+B22) -> contributes to C11, C12, C21, C22
        # M2 = A22*(B21-B11) -> C21, C22
        # M3 = (A11-A22)*(B12+B22) -> C11, C12
        # M4 = A11*(B12-B22) -> C11, C12
        # M5 = (A11+A12)*B22 -> C12, C22
        # M6 = (A21-A11)*(B11+B12) -> C21, C22
        # M7 = (A12-A21)*(B11+B21) -> C11, C21
        
        # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # B_left indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # C_left indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) [i.e., C[k*n+i]]
        
        # Strassen term 1: (A11+A22)*(B11+B22)
        add_term(0, 1, 0, 1, 0, 1, 0)   # C11
        add_term(0, 1, 0, 1, 1, 1, 0)   # C12 (k=0,i=1 -> W[0*2+1]=W[1])
        add_term(0, 1, 0, 1, 2, 1, 0)   # C21 (k=1,i=0 -> W[1*2+0]=W[2])
        add_term(0, 1, 0, 1, 3, 1, 0)   # C22 (k=1,i=1 -> W[1*2+1]=W[3])
        add_term(3, 1, 0, 1, 0, 1, 0)
        add_term(3, 1, 0, 1, 1, 1, 0)
        add_term(3, 1, 0, 1, 2, 1, 0)
        add_term(3, 1, 0, 1, 3, 1, 0)
        add_term(0, 1, 3, 1, 0, 1, 0)
        add_term(0, 1, 3, 1, 1, 1, 0)
        add_term(0, 1, 3, 1, 2, 1, 0)
        add_term(0, 1, 3, 1, 3, 1, 0)
        add_term(3, 1, 3, 1, 0, 1, 0)
        add_term(3, 1, 3, 1, 1, 1, 0)
        add_term(3, 1, 3, 1, 2, 1, 0)
        add_term(3, 1, 3, 1, 3, 1, 0)
        
        # This approach is getting complex. Let me use a known rank-15 decomposition.
        # The border rank of 2x3x4 is 15, and there's a known decomposition.
        # Let me use the approach from the literature.
        
        # Actually, let me just build a correct rank-22 and return it,
        # since the rank-15 construction is error-prone without careful verification.
        return None, None, None, None

    def build_rank_22():
        rank = 22
        U = [[0.0] * rank for _ in range(n * m)]  # 6 rows
        V = [[0.0] * rank for _ in range(m * p)]  # 12 rows
        W = [[0.0] * rank for _ in range(p * n)]  # 8 rows
        
        # Use the standard bilinear decomposition:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # This is rank n*m*p = 24 in general, but we can do better.
        # For 2x3x4, the rank is known to be 15 (border rank) or 17 (tensor rank).
        # Let me construct a valid decomposition.
        
        # Simple outer product decomposition (rank 24):
        # For each (i,j,k), add term with U[i*m+j][t]=1, V[j*p+k][t]=1, W[k*n+i][t]=1
        # But we can share terms.
        
        # Let's use a block approach:
        # Split into A[0:2,0:2]*B[0:2,0:4] + A[0:2,2:3]*B[2:3,0:4]
        
        # Block 1: 2x2 * 2x4
        # Use rank-8 decomposition for 2x2*2x4 (two 2x2*2x2 Strassen blocks = 14, or better)
        # Actually, 2x2*2x4 can be done in rank 8:
        # M1 = A11*(B11+B21) -> C11, C21
        # M2 = A12*(B11+B21) -> C11, C21  (wait, this isn't right)
        
        # Let me just use the naive rank-24 and verify it works, then try to optimize.
        # Actually, for safety, let me construct a known valid decomposition.
        
        # Rank-22: Use the fact that we can share some terms.
        # Let me construct it term by term and verify.
        
        # Term t: U[i,j], V[j,k], W[k,i] for specific (i,j,k)
        # We need: for each (i,k), sum_j A[i,j]*B[j,k] = C[i,k]
        # So the tensor T[i,j,k] = 1 for all (i,j,k) needs to be decomposed.
        
        # Simple decomposition: for each j, do outer product of A[:,j] and B[j,:]
        # That's 3 terms, each of rank n*p = 8, total rank 24.
        
        # Better: use Strassen on sub-blocks.
        # A[0:2, 0:2] * B[0:2, 0:4]: 2x2 * 2x4
        # A[0:2, 2:3] * B[2:3, 0:4]: 2x1 * 1x4 = rank 4 (outer product)
        # Total: rank(2x2*2x4) + 4
        
        # For 2x2 * 2x4:
        # Split B into B[:,0:2] and B[:,2:4]
        # 2x2*2x2 (Strassen, rank 7) + 2x2*2x2 (Strassen, rank 7) = 14
        # Or use a combined decomposition of rank 8 or 9.
        
        # Let me use rank 14 for the 2x2*2x4 part + 4 for the 2x1*1x4 part = 18.
        # Or find a better decomposition.
        
        # For now, let me construct a valid rank-18 decomposition.
        rank = 18
        U = [[0.0] * rank for _ in range(n * m)]  # 6 rows
        V = [[0.0] * rank for _ in range(m * p)]  # 12 rows
        W = [[0.0] * rank for _ in range(p * n)]  # 8 rows
        
        def add_strassen_block(a_rows, b_cols, c_rows, c_cols, start_col):
            """Add a Strassen decomposition for a 2x2 * 2x2 block."""
            # a_rows: (i1, i2), b_cols: (j1, j2), c_rows: (i1, i2), c_cols: (k1, k2)
            # U indices: i*m + j for i in a_rows, j in b_cols
            # V indices: j*p + k for j in b_cols, k in c_cols
            # W indices: k*n + i for k in c_cols, i in c_rows
            
            i1, i2 = a_rows
            j1, j2 = b_cols
            k1, k2 = c_cols
            
            U_i1j1 = i1 * m + j1
            U_i1j2 = i1 * m + j2
            U_i2j1 = i2 * m + j1
            U_i2j2 = i2 * m + j2
            
            V_j1k1 = j1 * p + k1
            V_j1k2 = j1 * p + k2
            V_j2k1 = j2 * p + k1
            V_j2k2 = j2 * p + k2
            
            W_k1i1 = k1 * n + i1
            W_k1i2 = k1 * n + i2
            W_k2i1 = k2 * n + i1
            W_k2i2 = k2 * n + i2
            
            # Strassen:
            # M1 = (A11+A22)*(B11+B22)
            U[U_i1j1][start_col] += 1
            U[U_i2j2][start_col] += 1
            V[V_j1k1][start_col] += 1
            V[V_j2k2][start_col] += 1
            W[W_k1i1][start_col] += 1
            W[W_k1i2][start_col] += 1
            W[W_k2i1][start_col] += 1
            W[W_k2i2][start_col] += 1
            
            # M2 = A22*(B21-B11)
            U[U_i2j2][start_col+1] += 1
            V[V_j2k1][start_col+1] += 1
            V[V_j1k1][start_col+1] += -1
            W[W_k1i2][start_col+1] += 1
            W[W_k2i2][start_col+1] += 1
            
            # M3 = (A11-A22)*(B12+B22)
            U[U_i1j1][start_col+2] += 1
            U[U_i2j2][start_col+2] += -1
            V[V_j1k2][start_col+2] += 1
            V[V_j2k2][start_col+2] += 1
            W[W_k1i1][start_col+2] += 1
            W[W_k2i1][start_col+2] += 1
            
            # M4 = A11*(B12-B22)
            U[U_i1j1][start_col+3] += 1
            V[V_j1k2][start_col+3] += 1
            V[V_j2k2][start_col+3] += -1
            W[W_k1i1][start_col+3] += 1
            W[W_k2i1][start_col+3] += 1
            
            # M5 = (A11+A12)*B22
            U[U_i1j1][start_col+4] += 1
            U[U_i1j2][start_col+4] += 1
            V[V_j2k2][start_col+4] += 1
            W[W_k1i1][start_col+4] += 1
            W[W_k2i1][start_col+4] += 1
            
            # M6 = (A21-A11)*(B11+B12)
            U[U_i2j1][start_col+5] += 1
            U[U_i1j1][start_col+5] += -1
            V[V_j1k1][start_col+5] += 1
            V[V_j1k2][start_col+5] += 1
            W[W_k1i2][start_col+5] += 1
            W[W_k2i2][start_col+5] += 1
            
            # M7 = (A12-A21)*(B11+B21)
            U[U_i1j2][start_col+6] += 1
            U[U_i2j1][start_col+6] += -1
            V[V_j1k1][start_col+6] += 1
            V[V_j2k1][start_col+6] += 1
            W[W_k1i1][start_col+6] += 1
            W[W_k2i1][start_col+6] += 1
            
            return start_col + 7
        
        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        col = add_strassen_block((0,1), (0,1), (0,1), (0,1), 0)
        
        # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        col = add_strassen_block((0,1), (0,1), (0,1), (2,3), col)
        
        # Block 3: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
        # This is 2x1 * 1x4, rank 4
        # U indices: 0*3+2=2 (A[0,2]), 1*3+2=5 (A[1,2])
        # V indices: 2*4+0=8, 2*4+1=9, 2*4+2=10, 2*4+3=11
        # W indices: k*2+i for k in 0..3, i in 0..1
        
        U[2][col] = 1
        V[8][col] = 1
        W[0][col] = 1  # k=0,i=0
        col += 1
        
        U[2][col] = 1
        V[9][col] = 1
        W[1][col] = 1  # k=1,i=0
        col += 1
        
        U[2][col] = 1
        V[10][col] = 1
        W[2][col] = 1  # k=2,i=0
        col += 1
        
        U[2][col] = 1
        V[11][col] = 1
        W[3][col] = 1  # k=3,i=0
        col += 1
        
        U[5][col] = 1
        V[8][col] = 1
        W[4][col] = 1  # k=0,i=1
        col += 1
        
        U[5][col] = 1
        V[9][col] = 1
        W[5][col] = 1  # k=1,i=1
        col += 1
        
        U[5][col] = 1
        V[10][col] = 1
        W[6][col] = 1  # k=2,i=1
        col += 1
        
        U[5][col] = 1
        V[11][col] = 1
        W[7][col] = 1  # k=3,i=1
        col += 1
        
        return U, V, W, rank

    # Build and verify rank-18
    U18, V18, W18, rank18 = build_rank_22()
    if verify(U18, V18, W18, rank18):
        return {"U": U18, "V": V18, "W": W18}

    # Fallback: naive rank-24
    rank = 24
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][t] = 1.0
                V[j*p+k][t] = 1.0
                W[k*n+i][t] = 1.0
                t += 1
    return {"U": U, "V": V, "W": W}

