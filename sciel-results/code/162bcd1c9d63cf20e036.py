import time
import random
import numpy as np
from fractions import Fraction

N, M, P = 4, 4, 9
NM, MP, PN = N * M, M * P, P * N  # 16, 36, 36

# Precompute support set: (u, v, w) with u=i*M+j, v=j*P+k, w=k*N+i
SUPPORT = []
SUPPORT_SET = set()
for i in range(N):
    for j in range(M):
        for k in range(P):
            u = i * M + j
            v = j * P + k
            w = k * N + i
            SUPPORT.append((u, v, w))
            SUPPORT_SET.add((u, v, w))

# For each (u, v) pair, the list of w's that are in support (should be 0 or 1 for matmul)
# Actually for matmul tensor, given (u, v) = (i*M+j, j'*P+k), support exists iff i == k and j == j',
# giving w = k*N + i. So at most one w per (u,v).
# Similarly, at most one u per (v,w), and at most one v per (u,w).
UV_TO_W = {}
VW_TO_U = {}
UW_TO_V = {}
for (u, v, w) in SUPPORT:
    UV_TO_W[(u, v)] = w
    VW_TO_U[(v, w)] = u
    UW_TO_V[(u, w)] = v


def _verify(U, V, W, n, m, p):
    """Verify using float arithmetic (fast)."""
    Ua = np.asarray(U, dtype=np.float64)
    Va = np.asarray(V, dtype=np.float64)
    Wa = np.asarray(W, dtype=np.float64)
    T = np.einsum('ur,vr,wr->uvw', Ua, Va, Wa)
    for (u, v, w) in SUPPORT:
        if abs(T[u, v, w] - 1.0) > 1e-9:
            return False
    # Check non-support: sample + full if small
    # Full check is 20736 - 144 = 20592 entries, fast enough
    mask = np.ones((n * m, m * p, p * n), dtype=bool)
    for (u, v, w) in SUPPORT:
        mask[u, v, w] = False
    if np.any(np.abs(T[mask]) > 1e-9):
        return False
    return True


def _verify_exact(U, V, W):
    """Verify with Fraction arithmetic for final answer."""
    from fractions import Fraction
    rank = len(U[0]) if U and U[0] else 0
    # T[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
    # We only need to check support entries == 1 and a sample of non-support == 0.
    # But for exactness, check all non-support.
    # Use integer scaling: entries are half-integers, so 2x entries are integers.
    U2 = [[2 * x for x in row] for row in U]
    V2 = [[2 * x for x in row] for row in V]
    W2 = [[2 * x for x in row] for row in W]
    # T2 = sum_r U2 * V2 * W2 / 8; we want T2/8 == 1 on support, 0 elsewhere
    # i.e., sum_r U2[u,r]*V2[v,r]*W2[w,r] == 8 on support, 0 elsewhere
    from collections import defaultdict
    T = np.zeros((NM, MP, PN), dtype=np.int64)
    for r in range(rank):
        Ur = np.asarray([U2[i][r] for i in range(NM)], dtype=np.int64)
        Vr = np.asarray([V2[i][r] for i in range(MP)], dtype=np.int64)
        Wr = np.asarray([W2[i][r] for i in range(PN)], dtype=np.int64)
        T += np.einsum('u,v,w->uvw', Ur, Vr, Wr)
    for (u, v, w) in SUPPORT:
        if T[u, v, w] != 8:
            return False
    mask = np.ones((NM, MP, PN), dtype=bool)
    for (u, v, w) in SUPPORT:
        mask[u, v, w] = False
    if np.any(T[mask] != 0):
        return False
    return True


def _build_schoolbook():
    rank = N * M * P
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    c = 0
    for (u, v, w) in SUPPORT:
        U[u][c] = 1
        V[v][c] = 1
        W[w][c] = 1
        c += 1
    return {"U": U, "V": V, "W": W}


def _build_strassen114():
    """Rank-114 Strassen-based: two 4x4x4 Strassen blocks (rank 49 each) + 4x4x1 schoolbook (rank 16).
    4x4x9 = [4x4 * 4x9]. Partition B into [4x4 | 4x5]? Actually B is 4x9 = 4 rows, 9 cols.
    Partition output C (4x9) into C1 (4x4) and C2 (4x5).
    C1 = A * B1 where B1 is 4x4 (first 4 cols of B): use Strassen rank 49.
    C2 = A * B2 where B2 is 4x5 (last 5 cols of B): use schoolbook 4*4*5 = 80? That's too much.
    Better: partition B into B1 (4x4) and B2 (4x5). For C2 = A * B2, we can further split B2 into 4x4 and 4x1.
    C2a = A * B2a (4x4): Strassen rank 49.
    C2b = A * B2b (4x1): schoolbook rank 16.
    Total: 49 + 49 + 16 = 114.
    """
    rank = 114
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]

    # Strassen 4x4x4 coefficients
    # A, B are 4x4. Partition into 2x2 blocks:
    # A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
    # 7 products:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For 4x4 matrices, block indices: row block r in {0,1}, col block c in {0,1}
    # Each block is 2x2. Global row index in 4x4: r*2 + i, col: c*2 + j, i,j in {0,1}.
    # So A[r*2+i, c*2+j], B same, C same.

    # Strassen 2x2x2 (base): 7 rank-1 terms for 2x2 matrices
    # For 2x2: A=[[a11,a12],[a21,a22]], B=[[b11,b12],[b21,b22]]
    # M1=(a11+a22)(b11+b22)
    # M2=(a21+a22)b11
    # M3=a11(b12-b22)
    # M4=a22(b21-b11)
    # M5=(a11+a12)b22
    # M6=(a21-a11)(b11+b12)
    # M7=(a12-a22)(b21+b22)
    # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6

    # For 4x4 using 2x2 blocks, each "block element" is itself a 2x2 matrix.
    # The Strassen products M1..M7 are each 2x2 matrix products, so each has rank 4 in the base.
    # So 7 * 4 = 28 rank-1 terms per 4x4x4? No wait.
    # Actually M1 = (A11+A22)*(B11+B22) is a 2x2 * 2x2 product = 4 scalar multiplications.
    # So each Mi contributes 4 rank-1 terms, total 7*4 = 28. But that's schoolbook for 4x4x4.
    # 
    # The Strassen algorithm for 4x4x4 using 2x2 blocks: 7 block multiplications, each 2x2x2.
    # Each 2x2x2 can be done with 7 rank-1 terms (Strassen base). So 7*7 = 49 rank-1 terms.
    # That's the rank-49 for 4x4x4.

    # Let me construct it properly.
    # 2x2x2 Strassen: for 2x2 matrices, 7 rank-1 terms.
    # Term t=0: U: a11,a22 (coeff 1 each); V: b11,b22 (coeff 1 each); W: c11(1), c22(1)
    #   Actually M1 = (a11+a22)(b11+b22). This is one rank-1 in the "block" sense but
    #   as a scalar tensor it's (a11+a22) * (b11+b22) = a11*b11 + a11*b22 + a22*b11 + a22*b22.
    #   That's 4 scalar products, not 1. So we need to expand.
    #
    # I think the cleaner way: just use the known 49-term decomposition for 4x4x4.
    # Let me hardcode it. The 4x4x4 matmul tensor has rank 49 (Strassen).
    # 
    # Alternative: use the 2x2x2 Strassen recursively. Each 2x2 block product uses 7 terms.
    # For 4x4: 7 block products, each is 2x2x2 with 7 terms = 49 total.
    #
    # Let me define the 2x2x2 Strassen terms explicitly.
    # 2x2x2: indices (i,j,k) with i,j in {0,1} for A cols and B rows, k for C.
    # A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1] -> u indices 0,1,2,3
    # B is 2x2: B[0,0], B[0,1], B[1,0], B[1,1] -> v indices 0,1,2,3
    # C is 2x2: C[0,0], C[0,1], C[1,0], C[1,1] -> w indices 0,1,2,3
    #
    # 7 terms (using Strassen):
    # t0: M1 = (a00+a11)(b00+b11)
    #   U: a00=1, a11=1
    #   V: b00=1, b11=1
    #   W: c00 += 1, c11 += 1 (M1 contributes to C00 and C11)
    #   Actually M1 contributes: C00 += M1, C11 += M1
    # t1: M2 = (a10+a11)b00
    #   U: a10=1, a11=1
    #   V: b00=1
    #   W: c10 += 1 (M2 contributes to C10), c11 -= 1 (M2 contributes -M2 to C11)
    #   Wait: C10 = M2+M4, C11 = M1-M2+M3+M6
    #   So M2: c10 += 1, c11 -= 1
    # t2: M3 = a00(b01-b11)
    #   U: a00=1
    #   V: b01=1, b11=-1
    #   W: c01 += 1, c11 += 1
    # t3: M4 = a11(b10-b00)
    #   U: a11=1
    #   V: b10=1, b00=-1
    #   W: c00 += 1, c10 += 1
    # t4: M5 = (a00+a01)b11
    #   U: a00=1, a01=1
    #   V: b11=1
    #   W: c00 -= 1, c01 += 1
    # t5: M6 = (a10-a00)(b00+b01)
    #   U: a10=1, a00=-1
    #   V: b00=1, b01=1
    #   W: c11 += 1
    # t6: M7 = (a01-a11)(b10+b11)
    #   U: a01=1, a11=-1
    #   V: b10=1, b11=1
    #   W: c00 += 1

    # Let me verify: 
    # C00 = M1 + M4 - M5 + M7
    #   M1: c00 += 1, M4: c00 += 1, M5: c00 -= 1, M7: c00 += 1 -> total 1+1-1+1 = 2? 
    #   That's wrong. Let me recheck Strassen.
    #
    # Standard Strassen 2x2:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # So the W coefficients should be:
    # t0 (M1): W[0,0]+=1, W[1,1]+=1
    # t1 (M2): W[1,0]+=1, W[1,1]+=-1
    # t2 (M3): W[0,1]+=1, W[1,1]+=1
    # t3 (M4): W[0,0]+=1, W[1,0]+=1
    # t4 (M5): W[0,0]+=-1, W[0,1]+=1
    # t5 (M6): W[1,1]+=1
    # t6 (M7): W[0,0]+=1

    # Let me verify C00: M1(+1) + M4(+1) + M5(-1) + M7(+1) = 2. That's wrong, should be 1.
    # 
    # I think I'm confusing myself. Let me just use the explicit 49-term 4x4x4 decomposition
    # by directly computing it.
    #
    # Actually, let me just use a simpler approach: hardcode the 49 terms for 4x4x4
    # by enumerating all (u,v,w) in support and using a known decomposition.
    #
    # For now, let me use a different approach: since we know the schoolbook is 144,
    # and Strassen gives 114, let me just use the schoolbook and then try to reduce.
    # The 114 construction is complex; let me focus on the search algorithm.

    # Simplified: just return schoolbook for now, the search will handle reduction.
    # Actually, let me try a cleaner Strassen construction.
    
    # 4x4x4 Strassen via 2x2 blocks, each 2x2x2 using 7 terms:
    # Block (br, bc) where br,bc in {0,1}: rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # A_block[br,bc] is 2x2: A[2*br+ii, 2*bc+jj] for ii,jj in {0,1}
    # 
    # 7 block products, each is 2x2x2 Strassen (7 terms).
    # Total 49 terms.
    
    # 2x2x2 Strassen terms (local 2x2 indices):
    # For local A (2x2), local B (2x2), local C (2x2):
    # term 0: U = [1,0,1,0] (a00,a01,a10,a11 -> a00=1,a11=1), V = [1,0,1,0] (b00=1,b11=1), W = [1,0,0,1] (c00=1,c11=1)
    # term 1: U = [0,0,1,1] (a10=1,a11=1), V = [1,0,0,0] (b00=1), W = [0,0,1,-1] (c10=1,c11=-1)
    # term 2: U = [1,0,0,0] (a00=1), V = [0,1,-1,0] (b01=1,b10=-1), W = [0,1,0,1] (c01=1,c11=1)
    # term 3: U = [0,0,0,1] (a11=1), V = [-1,0,1,0] (b00=-1,b10=1), W = [1,0,1,0] (c00=1,c10=1)
    # term 4: U = [1,1,0,0] (a00=1,a01=1), V = [0,0,0,1] (b11=1), W = [-1,1,0,0] (c00=-1,c01=1)
    # term 5: U = [-1,0,1,0] (a00=-1,a10=1), V = [1,1,0,0] (b00=1,b01=1), W = [0,0,0,1] (c11=1)
    # term 6: U = [0,1,-1,0] (a01=1,a10=-1), V = [0,0,1,1] (b10=1,b11=1), W = [1,0,0,0] (c00=1)

    S2 = [
        ([1,0,1,0], [1,0,1,0], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,-1,0], [0,1,0,1]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,-1,0], [0,0,1,1], [1,0,0,0]),
    ]

    def add_4x4x4(col_offset, a_row_offset, b_row_offset, c_row_offset):
        """Add a 4x4x4 Strassen decomposition.
        a_row_offset: offset for A rows (0-based), A is 4x4 starting at row a_row_offset
        b_row_offset: offset for B, B is 4x4
        c_row_offset: offset for C, C is 4x4
        """
        col = col_offset
        # 7 block products
        block_defs = [
            # (A11+A22, B11+B22, c00+, c11+) -> M1
            # (A21+A22, B11, c10+, c11-) -> M2
            # (A11, B12-B22, c01+, c11+) -> M3
            # (A22, B21-B11, c00+, c10+) -> M4
            # (A11+A12, B22, c00-, c01+) -> M5
            # (A21-A11, B11+B12, c11+) -> M6
            # (A12-A22, B21+B22, c00+) -> M7
        ]
        # For each block product, we need to add 2x2x2 Strassen (7 terms).
        # Block (br,bc) of A: rows 2*br+ii, cols 2*bc+jj
        # Block (br,bc) of B: rows 2*br+ii, cols 2*bc+jj
        # Block (br,bc) of C: rows 2*br+ii, cols 2*bc+jj
        
        # M1 = (A11+A22)(B11+B22): A blocks (0,0)+(1,1), B blocks (0,0)+(1,1), C: (0,0)+, (1,1)+
        # M2 = (A21+A22)B11: A blocks (1,0)+(1,1), B block (0,0), C: (1,0)+, (1,1)-
        # M3 = A11(B12-B22): A block (0,0), B blocks (0,1)-(1,1), C: (0,1)+, (1,1)+
        # M4 = A22(B21-B11): A block (1,1), B blocks (1,0)-(0,0), C: (0,0)+, (1,0)+
        # M5 = (A11+A12)B22: A blocks (0,0)+(0,1), B block (1,1), C: (0,0)-, (0,1)+
        # M6 = (A21-A11)(B11+B12): A blocks (1,0)-(0,0), B blocks (0,0)+(0,1), C: (1,1)+
        # M7 = (A12-A22)(B21+B22): A blocks (0,1)-(1,1), B blocks (1,0)+(1,1), C: (0,0)+
        
        # For each of the 7 M's, we have a sum/difference of A-blocks and B-blocks.
        # Each block is 2x2, so the product is a 2x2x2 matmul with modified coefficients.
        # We use the 2x2x2 Strassen (7 terms) for each, with the block coefficients applied.
        
        # This is getting complex. Let me just enumerate all 49 terms directly.
        # For term (m, t) where m in 0..6 (which M) and t in 0..6 (which 2x2x2 Strassen term):
        # The U coefficient for A element (2*br+ii, 2*bc+jj) is:
        #   sum over block_defs[m] of (block_coeff * S2[t][0][ii*2+jj])
        # Similarly for V and W.
        
        # Define block contributions for each M:
        # M=0 (M1): A blocks: (0,0)+1, (1,1)+1; B blocks: (0,0)+1, (1,1)+1; C blocks: (0,0)+1, (1,1)+1
        # M=1 (M2): A blocks: (1,0)+1, (1,1)+1; B blocks: (0,0)+1; C blocks: (1,0)+1, (1,1)-1
        # M=2 (M3): A blocks: (0,0)+1; B blocks: (0,1)+1, (1,1)-1; C blocks: (0,1)+1, (1,1)+1
        # M=3 (M4): A blocks: (1,1)+1; B blocks: (1,0)+1, (0,0)-1; C blocks: (0,0)+1, (1,0)+1
        # M=4 (M5): A blocks: (0,0)+1, (0,1)+1; B blocks: (1,1)+1; C blocks: (0,0)-1, (0,1)+1
        # M=5 (M6): A blocks: (1,0)+1, (0,0)-1; B blocks: (0,0)+1, (0,1)+1; C blocks: (1,1)+1
        # M=6 (M7): A blocks: (0,1)+1, (1,1)-1; B blocks: (1,0)+1, (1,1)+1; C blocks: (0,0)+1
        
        M_defs = [
            # (A_blocks, B_blocks, C_blocks) where each is list of ((br,bc), coeff)
            ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
            ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1),((1,1),-1)]),
            ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)]),
            ([((1,1),1)], [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)]),
            ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),-1),((0,1),1)]),
            ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]),
            ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,0),1)]),
        ]
        
        for m in range(7):
            A_blocks, B_blocks, C_blocks = M_defs[m]
            for t in range(7):
                U2, V2, W2 = S2[t]
                # U2, V2, W2 are 4-element arrays for 2x2 matrices (row-major: 00,01,10,11)
                for ii in range(2):
                    for jj in range(2):
                        local_idx = ii * 2 + jj
                        # U contribution
                        u_coeff = 0
                        for (br, bc), coeff in A_blocks:
                            u_coeff += coeff * U2[local_idx]
                        if u_coeff != 0:
                            # Global A index: row = a_row_offset + 2*br + ii, col = 2*bc + jj
                            a_row = a_row_offset + 2 * 0 + ii  # A is 4x4, so col offset is 0
                            a_col = 2 * bc + jj
                            u_idx = a_row * 4 + a_col
                            U[u_idx][col] += u_coeff
                        # V contribution
                        v_coeff = 0
                        for (br, bc), coeff in B_blocks:
                            v_coeff += coeff * V2[local_idx]
                        if v_coeff != 0:
                            b_row = 2 * br + ii
                            b_col = 2 * bc + jj
                            v_idx = b_row * 4 + b_col  # B is 4x4, row offset 0
                            V[v_idx][col] += v_coeff
                        # W contribution
                        w_coeff = 0
                        for (br, bc), coeff in C_blocks:
                            w_coeff += coeff * W2[local_idx]
                        if w_coeff != 0:
                            # C is 4x9, but for this 4x4x4 subproblem, C is 4x4
                            # w index in the 4x4 C: row = 2*br+ii, col = 2*bc+jj
                            # Global w index: w = k*N + i where k is col of C, i is row of C
                            # For 4x4 C: w = c_col * 4 + c_row
                            c_row = 2 * br + ii
                            c_col = 2 * bc + jj
                            w_idx = c_col * 4 + c_row  # + c_row_offset for global
                            w_idx += c_row_offset
                            W[w_idx][col] += w_coeff
                col += 1
        return col
    
    col = 0
    # C1 = A * B1 (B1 is 4x4, first 4 cols of B)
    # A is 4x4 (rows 0-3, cols 0-3), B1 is 4x4 (rows 0-3, cols 0-3), C1 is 4x4 (rows 0-3, cols 0-3)
    # w index for C1: w = k*4 + i, k in 0..3, i in 0..3
    col = add_4x4x4(col, 0, 0, 0)
    
    # C2a = A * B2a (B2a is 4x4, cols 4-7 of B)
    # B2a: rows 0-3, cols 0-3 (local), global cols 4-7
    # C2a is 4x4, global cols 4-7
    # w index for C2a: w = k*4 + i, k in 4..7 -> w = (k-4+4)*4 + i = k*4+i, but k is global col
    # Actually w = k_global * N + i_global = k_global * 4 + i
    # For C2a: k_global in 4..7, i in 0..3
    # In the 4x4x4 subproblem, local k in 0..3, global k = 4 + local_k
    # w_global = (4 + local_k) * 4 + i = 16 + local_k*4 + i
    # But the 4x4x4 function computes w_idx = c_col * 4 + c_row + c_row_offset
    # c_row_offset should be 16 to shift to cols 4-7
    col = add_4x4x4(col, 0, 0, 16)
    
    # C2b = A * B2b (B2b is 4x1, col 8 of B)
    # Schoolbook: 4*4*1 = 16 terms
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 1 + 0][col] = 1  # B2b is 4x1, local index j*1+0 = j
            # global B index: row j, col 8 -> v = j*9 + 8
            # But V is indexed by j*p+k = j*9+k. For B2b, k=0 local, global k=8.
            # So v_global = j*9 + 8
            # The V matrix has MP = 36 rows. v = j*9 + 8.
            V[j * 9 + 8][col] = 1
            # W: w = k_global*N + i = 8*4 + i = 32 + i
            W[8 * 4 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}


def _random_rank1_search(deadline, seed=42):
    """
    Build a decomposition by adding rank-1 terms.
    Each term is chosen to cover uncovered support entries while respecting zero constraints.
    """
    rng = random.Random(seed)
    
    # Track which support entries are covered
    # covered[u,v,w] = sum of contributions from added terms
    covered = np.zeros((NM, MP, PN), dtype=np.float64)
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Find all (u,v) pairs that are in support
    uv_pairs = [(u, v) for (u, v, w) in SUPPORT]
    
    # For each (u,v), the corresponding w is determined
    # A rank-1 term (U_vec, V_vec, W_vec) contributes to T[u,v,w] = U_vec[u]*V_vec[v]*W_vec[w]
    # For the term to be valid, for all (u',v') not in support, U_vec[u']*V_vec[v'] must be 0
    # (or the product with W_vec must be 0)
    
    # Strategy: pick a random (u,v) in support, set U_vec[u] = 1, V_vec[v] = 1.
    # Then for the term to not violate non-support constraints, we need:
    # For all (u',v') not in support: U_vec[u']*V_vec[v'] = 0
    # This means: if U_vec[u'] != 0 then V_vec[v'] must be 0 for all v' where (u',v') not in support.
    # Similarly if V_vec[v'] != 0 then U_vec[u'] must be 0 for all u' where (u',v') not in support.
    
    # For a given u, the v's that are in support with u: v_support(u) = {v : (u,v) in UV_TO_W}
    # For a given v, the u's that are in support with v: u_support(v) = {u : (u,v) in UV_TO_W}
    
    # If U_vec is nonzero only at u, and V_vec is nonzero only at v, and (u,v) is in support,
    # then the term is valid (no non-support violations).
    # But that's just a single (u,v) term, which is schoolbook.
    
    # To get a better decomposition, we need U_vec and V_vec to be nonzero at multiple positions
    # while maintaining the zero constraint.
    
    # For 4x4x9: u in 0..15, v in 0..35.
    # u = i*4+j (i in 0..3, j in 0..3), v = j'*9+k (j' in 0..3, k in 0..8)
    # (u,v) in support iff i == k and j == j', w = k*4+i = i*4+i.
    # So for a given u = i*4+j, the only v in support is v = j*9+i.
    # For a given v = j'*9+k, the only u in support is u = k*4+j'.
    
    # So each u has exactly one v in support, and each v has exactly one u in support.
    # This means: if U_vec is nonzero at u, then V_vec can only be nonzero at v_support(u).
    # And if V_vec is nonzero at v, then U_vec can only be nonzero at u_support(v).
    
    # So the only valid rank-1 terms (with nonzero U and V) are those where
    # U is nonzero only at some u, and V is nonzero only at v_support(u).
    # That gives only 144 possible (u,v) pairs, each with a single w.
    # This is just schoolbook!
    
    # Wait, that can't be right. The tensor must have rank < 144.
    # The issue: W_vec can be nonzero at multiple w's.
    # For a term with U nonzero at u and V nonzero at v (where (u,v) in support, w = w_uv),
    # W_vec[w] contributes to T[u,v,w] = U[u]*V[v]*W[w].
    # For non-support (u',v',w'): T[u',v',w'] = U[u']*V[v']*W[w']
    # If U is nonzero only at u, then T[u',v',w'] = 0 for u' != u. Good.
    # For u' = u: T[u,v',w'] = U[u]*V[v']*W[w']. For this to be 0 when (u,v',w') not in support:
    #   If V is nonzero only at v, then T[u,v',w'] = 0 for v' != v. Good.
    #   For v' = v: T[u,v,w'] = U[u]*V[v]*W[w']. For this to be 0 when w' != w_uv:
    #     W[w'] must be 0 for w' != w_uv.
    # So W is also forced to be nonzero only at w_uv.
    # This confirms: for this tensor, any rank-1 term in a valid decomposition
    # must have U, V, W each nonzero at exactly one position, and that position must be
    # a support entry. So the tensor is a sum of 144 orthogonal rank-1 terms?
    # 
    # No wait. The rank-1 terms don't have to be orthogonal. The sum of rank-1 terms
    # gives the tensor. Individual terms can have U, V, W nonzero at multiple positions,
    # as long as the sum works out.
    #
    # The constraint is on the sum, not on individual terms.
    # For a set of terms, the sum T[u,v,w] = sum_r U_r[u]*V_r[v]*W_r[w] must equal
    # 1 on support and 0 elsewhere.
    #
    # So individual terms CAN have U, V, W nonzero at multiple positions, as long as
    # the interference cancels out.
    
    # Let me try a different approach: start with schoolbook and try to merge terms.
    # Or use a randomized approach where we generate random terms and solve for coefficients.
    
    # Actually, let me just use the known 114-rank Strassen construction and then
    # try to reduce further.
    pass


def construct():
    n, m, p = N, M, P
    deadline = time.time() + 140.0

    # Start with Strassen 114
    res = _build_strassen114()
    if not _verify(res["U"], res["V"], res["W"], n, m, p):
        res = _build_schoolbook()
        if not _verify(res["U"], res["V"], res["W"], n, m, p):
            return res

    U, V, W = res["U"], res["V"], res["W"]
    current_rank = len(U[0])

    # Try greedy removal
    if time.time() < deadline - 20:
        U, V, W = _greedy_remove(U, V, W, n, m, p, deadline - 20)
        if len(U[0]) < current_rank:
            current_rank = len(U[0])

    # Try random column ops + removal
    if time.time() < deadline - 10:
        U2, V2, W2 = _random_column_ops_and_remove(U, V, W, n, m, p, deadline, seed=12345)
        if len(U2[0]) < current_rank:
            U, V, W = U2, V2, W2
            current_rank = len(U2[0])

    # Final verification
    if not _verify(U, V, W, n, m, p):
        res_orig = _build_strassen114()
        if _verify(res_orig["U"], res_orig["V"], res_orig["W"], n, m, p):
            return res_orig
        res_school = _build_schoolbook()
        return res_school

    return {"U": U, "V": V, "W": W}


def _greedy_remove(U, V, W, n, m, p, deadline):
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        rank = len(U[0])
        for c in range(rank):
            if time.time() > deadline:
                break
            U_new = [row[:c] + row[c+1:] for row in U]
            V_new = [row[:c] + row[c+1:] for row in V]
            W_new = [row[:c] + row[c+1:] for row in W]
            if _verify(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                changed = True
                break
    return U, V, W


def _random_column_ops_and_remove(U, V, W, n, m, p, deadline, seed=42):
    rng = random.Random(seed)
    best_U, best_V, best_W = [r[:] for r in U], [r[:] for r in V], [r[:] for r in W]
    best_rank = len(best_U[0])
    U = [r[:] for r in U]
    V = [r[:] for r in V]
    W = [r[:] for r in W]
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        if len(U[0]) <= best_rank:
            break
        rank = len(U[0])
        a = rng.randint(rank)
        b = rng.randint(rank)
        if a == b:
            continue
        alpha = rng.choice([-3, -2, -1, -0.5, 0.5, 1, 2, 3])
        for r in range(len(U)):
            U[r][a] += alpha * U[r][b]
        for r in range(len(V)):
            V[r][a] += alpha * V[r][b]
        for r in range(len(W)):
            W[r][a] += alpha * W[r][b]
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            rk = len(U[0])
            for c in range(rk):
                if time.time() > deadline - 5:
                    break
                U_new = [row[:c] + row[c+1:] for row in U]
                V_new = [row[:c] + row[c+1:] for row in V]
                W_new = [row[:c] + row[c+1:] for row in W]
                if _verify(U_new, V_new, W_new, n, m, p):
                    U, V, W = U_new, V_new, W_new
                    changed = True
                    if len(U[0]) < best_rank:
                        best_rank = len(U[0])
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                    break
        if len(U[0]) > best_rank and trial % 20 == 0:
            U = [row[:] for row in best_U]
            V = [row[:] for row in best_V]
            W = [row[:] for row in best_W]
    return best_U, best_V, best_W
