"""Matrix multiplication 4x5 by 5x6 — Hybrid Strassen/schoolbook decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """
    Build the 2x2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _remove_redundant_columns(U, V, W, tol=1e-9):
    """
    Remove columns that are linearly dependent across U, V, W simultaneously.
    A column r is redundant if the triple (U[:,r], V[:,r], W[:,r]) lies in the
    span of other such triples. We detect this by checking if removing column r
    still preserves the tensor. For efficiency, we use a greedy approach:
    check if column r of U is in the span of other U columns, AND similarly for V and W.
    This is a sufficient (not necessary) condition for redundancy.
    """
    nU, rU = len(U), len(U[0])
    nV, rV = len(V), len(V[0])
    nW, rW = len(W), len(W[0])
    
    if rU != rV or rV != rW:
        return U, V, W
    
    rank = rU
    keep = list(range(rank))
    
    # Greedy: try to remove columns from the end
    for c in range(rank - 1, -1, -1):
        if c not in keep:
            continue
        # Check if column c of U is in span of other kept columns
        others = [k for k in keep if k != c]
        if not others:
            break
        
        # Build matrix of other U columns
        U_others = np.array([U[row][k] for row in range(nU) for k in others], dtype=float).reshape(nU, len(others))
        U_c = np.array([U[row][c] for row in range(nU)], dtype=float)
        
        # Check if U_c is in column space of U_others
        try:
            Q, R = np.linalg.qr(U_others)
            proj = Q @ (Q.T @ U_c)
            if np.linalg.norm(U_c - proj) > tol:
                continue
        except:
            continue
        
        # Similarly check V
        V_others = np.array([V[row][k] for row in range(nV) for k in others], dtype=float).reshape(nV, len(others))
        V_c = np.array([V[row][c] for row in range(nV)], dtype=float)
        try:
            Q, R = np.linalg.qr(V_others)
            proj = Q @ (Q.T @ V_c)
            if np.linalg.norm(V_c - proj) > tol:
                continue
        except:
            continue
        
        # Similarly check W
        W_others = np.array([W[row][k] for row in range(nW) for k in others], dtype=float).reshape(nW, len(others))
        W_c = np.array([W[row][c] for row in range(nW)], dtype=float)
        try:
            Q, R = np.linalg.qr(W_others)
            proj = Q @ (Q.T @ W_c)
            if np.linalg.norm(W_c - proj) > tol:
                continue
        except:
            continue
        
        # Column c is redundant
        keep.remove(c)
    
    # Rebuild with only kept columns
    new_rank = len(keep)
    U_new = [[U[row][k] for k in keep] for row in range(nU)]
    V_new = [[V[row][k] for k in keep] for row in range(nV)]
    W_new = [[W[row][k] for k in keep] for row in range(nW)]
    
    return U_new, V_new, W_new

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one global decomposition.
    Each decomposition is a tuple: (U_sub, V_sub, W_sub, i_start, j_start, k_start, i_size, j_size, k_size)
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, i_s, j_s, k_s, i_sz, j_sz, k_sz in decomps:
        rank_sub = len(U_sub[0])
        
        # Map A indices: i in [0, i_sz), j in [0, j_sz)
        for i in range(i_sz):
            for j in range(j_sz):
                u_row_sub = i * j_sz + j
                u_row_glob = (i_s + i) * m + (j_s + j)
                for c in range(rank_sub):
                    U[u_row_glob][current_rank + c] = U_sub[u_row_sub][c]
        
        # Map B indices: j in [0, j_sz), k in [0, k_sz)
        for j in range(j_sz):
            for k in range(k_sz):
                v_row_sub = j * k_sz + k
                v_row_glob = (j_s + j) * p + (k_s + k)
                for c in range(rank_sub):
                    V[v_row_glob][current_rank + c] = V_sub[v_row_sub][c]
        
        # Map C indices: k in [0, k_sz), i in [0, i_sz)
        # Convention: W[k*n + i]
        for k in range(k_sz):
            for i in range(i_sz):
                w_row_sub = k * i_sz + i
                w_row_glob = (k_s + k) * n + (i_s + i)
                for c in range(rank_sub):
                    W[w_row_glob][current_rank + c] = W_sub[w_row_sub][c]
        
        current_rank += rank_sub
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = 140

    # Strategy: Split the 4x5x6 multiplication into blocks.
    # We can split the inner dimension m=5 into 2 and 3.
    # A (4x5) = [A1 (4x2) | A2 (4x3)]
    # B (5x6) = [B1 (2x6); B2 (3x6)]
    # C = A1*B1 + A2*B2
    
    # Block 1: 4x2 * 2x6 -> 4x6
    # We can further split this. 4=2+2, 6=3+3.
    # A1 (4x2) = [[A11 (2x2)], [A21 (2x2)]]
    # B1 (2x6) = [B11 (2x3) | B12 (2x3)]
    # C1 = A11*B11 + A21*B12 (if we align dimensions correctly? No, B1 is 2x6, so B11 is 2x3, B12 is 2x3)
    # Wait, A1 is 4x2. B1 is 2x6.
    # A1 = [A_top (2x2); A_bot (2x2)]
    # B1 = [B_left (2x3) | B_right (2x3)]
    # C1 = A_top * B_left + A_bot * B_right ? No.
    # C1 = A1 * B1 = [A_top; A_bot] * [B_left | B_right]
    #    = A_top*B_left + A_bot*B_left (for left part)
    #    = A_top*B_right + A_bot*B_right (for right part)
    # This is just standard block multiplication. It doesn't reduce rank unless we use Strassen on the blocks.
    
    # Let's use Strassen on the 2x2x2 blocks where possible.
    # For 4x2 * 2x6:
    # We have 4 rows of A, 2 cols of A (inner), 6 cols of B.
    # We can split the 4 rows into two 2x2 blocks of A.
    # A = [A1 (2x2); A2 (2x2)]
    # B = [B1 (2x3) | B2 (2x3)]
    # C = A1*B1 + A2*B1 (left part, 2x3)
    #     A1*B2 + A2*B2 (right part, 2x3)
    # This is 4 products of 2x2 * 2x3.
    # Rank of 2x2 * 2x3 is at most 2*2*3 = 12 (schoolbook) or potentially less?
    # Actually, 2x2 * 2x2 is rank 7. 2x2 * 2x3 can be seen as 2x2 * 2x2 + 2x2 * 2x1?
    # Or just use schoolbook for 2x2*2x3 which is rank 12.
    # Total rank for this block: 4 * 12 = 48? No, that's if we do them independently.
    # Wait, A1*B1 and A2*B1 share B1.
    # Let's just use the generic schoolbook for the 4x2*2x6 block? Rank 4*2*6 = 48.
    # Or use Strassen 2x2x2 for the 2x2 parts?
    # A1 (2x2) * B1_left (2x2 part of 2x3) -> C1_left_top (2x2)
    # This is getting complicated to optimize manually.
    
    # Let's try a simpler split:
    # Split m=5 into 2 and 3.
    # Block 1: 4x2 * 2x6. Rank 48 (schoolbook).
    # Block 2: 4x3 * 3x6. Rank 72 (schoolbook).
    # Total: 120. Same as baseline.
    
    # We need to use Strassen to beat 120.
    # The 2x2x2 Strassen is rank 7 vs 8 schoolbook.
    # Can we fit 2x2x2 blocks?
    # 4x5x6.
    # If we split 4 into 2+2, 5 into 2+3, 6 into 2+4?
    
    # Let's try to construct a 2x2x2 Strassen block for the top-left 2x2x2 part.
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # This is a valid 2x2x2 sub-tensor.
    # Rank 7.
    
    # What about the rest?
    # This is a "bordered" tensor.
    
    # Alternative: Use the known rank 105 solution from the prompt knowledge.
    # "Replacing the potentially invalid/rank-28 Strassen block for 4x4x2 with a guaranteed valid Schoolbook block (rank 32) ensures the decomposition is correct, yielding a total rank of 105 (49+24+32)"
    
    # Let's try to replicate that structure:
    # 4x5x6 can be viewed as 4x4x2 + 4x1x6? No.
    # 4x5x6 = 4x4x6 + 4x1x6?
    # 4x4x6 can be decomposed?
    # 4x4x4 is rank 49 (Strassen).
    # 4x4x2 is rank 28? (2 * 14? or 4 * 7?)
    # If we use 4x4x4 (rank 49) + 4x1x2 (rank 8) + 4x4x2 (rank 28)?
    # Total 49 + 8 + 28 = 85?
    # But dimensions don't match directly.
    
    # Let's stick to the prompt's suggestion: "Construct a rank-96 decomposition... 2x3x3 Strassen-Winograd block (rank 21) and a 2x3x2 block (rank 14) repeated twice"
    # This implies splitting 4x5x6 into:
    # (2x3x3) + (2x3x2) + (2x3x2) ?
    # 2+2=4 (n)
    # 3+2+2=7? No, m=5.
    # Maybe 3+2=5 (m).
    # 3+2+2=7? No, p=6.
    
    # Let's look at the dimensions:
    # n=4, m=5, p=6.
    # Split n into 2+2.
    # Split m into 3+2.
    # Split p into 3+3.
    
    # Block 1: n=2, m=3, p=3.
    # Block 2: n=2, m=3, p=3.
    # Block 3: n=2, m=2, p=3.
    # Block 4: n=2, m=2, p=3.
    
    # Total rank = 2 * rank(2x3x3) + 2 * rank(2x2x3).
    # If rank(2x3x3) is 18 (schoolbook) or less?
    # If rank(2x2x3) is 12 (schoolbook) or less?
    # 2*18 + 2*12 = 36 + 24 = 60.
    # This is much better than 120.
    
    # Let's verify if we can just use schoolbook for these blocks.
    # Yes, schoolbook is always valid.
    # Rank(2x3x3) = 2*3*3 = 18.
    # Rank(2x2x3) = 2*2*3 = 12.
    # Total = 2*18 + 2*12 = 60.
    
    # Can we do better with Strassen?
    # 2x2x2 is rank 7.
    # 2x3x3 can be split into 2x2x2 + 2x1x3 + 2x2x1 + 2x1x1?
    # This is getting complex. Let's just use the 60 rank schoolbook combination first to ensure correctness and a big jump from 120.
    
    decomps = []
    
    # Block 1: i=0..1, j=0..2, k=0..2 (2x3x3)
    U1, V1, W1 = _build_schoolbook(2, 3, 3)
    decomps.append((U1, V1, W1, 0, 0, 0, 2, 3, 3))
    
    # Block 2: i=0..1, j=0..2, k=3..5 (2x3x3)
    U2, V2, W2 = _build_schoolbook(2, 3, 3)
    decomps.append((U2, V2, W2, 0, 0, 3, 2, 3, 3))
    
    # Block 3: i=2..3, j=0..2, k=0..2 (2x3x3)
    U3, V3, W3 = _build_schoolbook(2, 3, 3)
    decomps.append((U3, V3, W3, 2, 0, 0, 2, 3, 3))
    
    # Block 4: i=2..3, j=0..2, k=3..5 (2x3x3)
    U4, V4, W4 = _build_schoolbook(2, 3, 3)
    decomps.append((U4, V4, W4, 2, 0, 3, 2, 3, 3))
    
    # Wait, m=5. I used j=0..2 (size 3) for all. I need j=3..4 (size 2) for the rest.
    # My split of m was 3+2.
    # So for j=0..2, I need to cover all i and k.
    # For j=3..4, I need to cover all i and k.
    
    # Let's re-plan the blocks based on m split 3+2.
    # Part A: m=3 (j=0,1,2). n=4, p=6.
    # Part B: m=2 (j=3,4). n=4, p=6.
    
    # Part A (4x3x6):
    # Split n into 2+2. Split p into 3+3.
    # A1: 2x3x3 (i=0..1, j=0..2, k=0..2)
    # A2: 2x3x3 (i=0..1, j=0..2, k=3..5)
    # A3: 2x3x3 (i=2..3, j=0..2, k=0..2)
    # A4: 2x3x3 (i=2..3, j=0..2, k=3..5)
    # Rank: 4 * 18 = 72.
    
    # Part B (4x2x6):
    # Split n into 2+2. Split p into 3+3.
    # B1: 2x2x3 (i=0..1, j=3..4, k=0..2)
    # B2: 2x2x3 (i=0..1, j=3..4, k=3..5)
    # B3: 2x2x3 (i=2..3, j=3..4, k=0..2)
    # B4: 2x2x3 (i=2..3, j=3..4, k=3..5)
    # Rank: 4 * 12 = 48.
    
    # Total Rank: 72 + 48 = 120.
    # This is just schoolbook again! Because 4*3*6 + 4*2*6 = 72 + 48 = 120.
    # Splitting doesn't change the total rank if we use schoolbook for all sub-blocks.
    # We MUST use Strassen (or other low-rank) decompositions for the sub-blocks to beat 120.
    
    # Let's use Strassen for the 2x2x2 parts where possible.
    # In Part A (4x3x6), we have 2x3x3 blocks.
    # Can we decompose 2x3x3 with rank < 18?
    # 2x3x3 contains 2x2x2.
    # Let's try to use Strassen 2x2x2 for the 2x2x2 part and schoolbook for the rest.
    # 2x3x3 = 2x2x2 + 2x1x3 + 2x2x1 + 2x1x1.
    # This is messy.
    
    # Let's try the 105 rank solution from the prompt.
    # It mentioned 49 + 24 + 32.
    # 49 is rank of 4x4x4.
    # 24 is rank of 4x2x3? (4*2*3=24 schoolbook).
    # 32 is rank of 4x1x8? No.
    
    # Let's try to construct a 4x4x4 Strassen block.
    # We have n=4, m=5, p=6.
    # We can take a 4x4x4 sub-tensor: i=0..3, j=0..3, k=0..3.
    # Rank 49.
    # Remaining:
    # i=0..3, j=0..3, k=4..5 (4x4x2)
    # i=0..3, j=4, k=0..5 (4x1x6)
    # Rank(4x4x2) <= 32 (schoolbook).
    # Rank(4x1x6) = 24 (schoolbook).
    # Total: 49 + 32 + 24 = 105.
    
    # This matches the prompt's "105" finding!
    # Let's implement this.
    
    decomps = []
    
    # Block 1: 4x4x4 (i=0..3, j=0..3, k=0..3)
    # We need a rank 49 decomposition for 4x4x4.
    # We can build this by combining 16 2x2x2 Strassen blocks?
    # No, 4x4x4 is not just 4 independent 2x2x2s.
    # It is a Kronecker product of 2x2x2 with 2x2x2?
    # The rank of the Kronecker product is the product of ranks: 7*7=49.
    # So we can build 4x4x4 by taking the Kronecker product of two 2x2x2 Strassen decompositions.
    
    def _build_kronecker_strassen_4x4x4():
        U2, V2, W2 = _build_strassen_2x2x2()
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        for t1 in range(7):
            for t2 in range(7):
                col = t1 * 7 + t2
                # U[i1*2+i2, j1*2+j2] = U2[i1,j1,t1] * U2[i2,j2,t2]
                for i1 in range(2):
                    for j1 in range(2):
                        c1 = U2[i1*2+j1][t1]
                        if c1 == 0: continue
                        for i2 in range(2):
                            for j2 in range(2):
                                c2 = U2[i2*2+j2][t2]
                                if c2 == 0: continue
                                u_row = (i1*2+i2)*4 + (j1*2+j2)
                                U[u_row][col] += c1 * c2
                
                # V[j1*2+j2, k1*2+k2] = V2[j1,k1,t1] * V2[j2,k2,t2]
                for j1 in range(2):
                    for k1 in range(2):
                        c1 = V2[j1*2+k1][t1]
                        if c1 == 0: continue
                        for j2 in range(2):
                            for k2 in range(2):
                                c2 = V2[j2*2+k2][t2]
                                if c2 == 0: continue
                                v_row = (j1*2+j2)*4 + (k1*2+k2)
                                V[v_row][col] += c1 * c2
                
                # W[k1*2+k2, i1*2+i2] = W2[k1,i1,t1] * W2[k2,i2,t2]
                for k1 in range(2):
                    for i1 in range(2):
                        c1 = W2[k1*2+i1][t1]
                        if c1 == 0: continue
                        for k2 in range(2):
                            for i2 in range(2):
                                c2 = W2[k2*2+i2][t2]
                                if c2 == 0: continue
                                w_row = (k1*2+k2)*4 + (i1*2+i2)
                                W[w_row][col] += c1 * c2
        
        return U, V, W

    U4, V4, W4 = _build_kronecker_strassen_4x4x4()
    decomps.append((U4, V4, W4, 0, 0, 0, 4, 4, 4))
    
    # Block 2: 4x4x2 (i=0..3, j=0..3, k=4..5)
    # Use schoolbook. Rank 4*4*2 = 32.
    U5, V5, W5 = _build_schoolbook(4, 4, 2)
    decomps.append((U5, V5, W5, 0, 0, 4, 4, 4, 2))
    
    # Block 3: 4x1x6 (i=0..3, j=4, k=0..5)
    # Use schoolbook. Rank 4*1*6 = 24.
    U6, V6, W6 = _build_schoolbook(4, 1, 6)
    decomps.append((U6, V6, W6, 0, 4, 0, 4, 1, 6))
    
    U, V, W = _combine_decompositions(decomps, n, m, p)
    
    # Remove redundant columns if time permits
    if time.time() < deadline - 30:
        U, V, W = _remove_redundant_columns(U, V, W)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
