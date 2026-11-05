import time
import sys
import numpy as np

# Deadline guard
_START = time.time()
_DEADLINE = 150.0
_SAFETY = 2.0  # seconds to spare

def _time_left():
    return _DEADLINE - _START - time.time() - _SAFETY

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Return {'U', 'V', 'W'} for 4x4 * 4x5 matrix multiplication.
    
    Strategy:
      - 4x4x4 part: Strassen decomposition (rank 7) applied to 2x2 blocks,
        with schoolbook for each 2x2x2 sub-multiplication.
        7 outer terms * 4 inner terms = 28 terms for 4x4x4.
      - 4x4x1 part (5th column of B): schoolbook, rank 16.
      Total rank: 28 + 16 = 44.
    
    U: shape (16, 44) - A factors, indexed by (i, j) where A is 4x4
    V: shape (20, 44) - B factors, indexed by (j, k) where B is 4x5
    W: shape (20, 44) - C factors, indexed by (k, i) where C is 5x4
      W[k*n + i, r] = coefficient for C[k][i] from rank-r term
    """
    n, m, p = 4, 4, 5  # A: n x m, B: m x p, C: n x p
    
    # But wait: the problem says "4 by 4 matrix by 4 by 5 matrix"
    # So A is 4x4, B is 4x5, C is 4x5.
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (20, rank) with W[k*n+i, r]
    
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # C is 4x5 (n x p), B is 4x5 (m x p)
    # C[i][k] = sum_j A[i][j] * B[j][k]
    # W index: k*n + i, where k is column of C (0..4), i is row of C (0..3)
    # So W has p*n = 5*4 = 20 rows.
    
    # For the 4x4x4 part (first 4 columns of B):
    # C[i][k] for k=0..3 = sum_{j=0..3} A[i][j] * B[j][k]
    # For the 4x4x1 part (5th column of B, k=4):
    # C[i][4] = sum_{j=0..3} A[i][j] * B[j][4]
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 16 x 44
    V = [[0] * rank for _ in range(m * p)]  # 20 x 44
    W = [[0] * rank for _ in range(p * n)]  # 20 x 44
    
    col = 0
    
    def add_term(u_list, v_list, w_list):
        nonlocal col
        if col >= rank:
            return
        for idx_u, c in u_list:
            if c != 0:
                U[idx_u][col] += c
        for idx_v, c in v_list:
            if c != 0:
                V[idx_v][col] += c
        for idx_w, c in w_list:
            if c != 0:
                W[idx_w][col] += c
        col += 1
    
    # Strassen for 4x4x4 using 2x2 blocks
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]] where each block is 2x2
    # C = A*B, each C block is 2x2
    
    # 7 Strassen products for the outer 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Then:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix product, computed with schoolbook (4 multiplications)
    # So 7 * 4 = 28 terms for the 4x4x4 part.
    
    # For each Strassen term M_s, it's a product of two 2x2 matrices:
    # M_s = P * Q where P and Q are linear combos of 2x2 blocks
    # Each 2x2 multiplication has 4 terms (schoolbook)
    
    # Let's define the Strassen terms with their P and Q block coefficients
    # P block coeffs: [p00, p01, p10, p11] for A00, A01, A10, A11
    # Q block coeffs: [q00, q01, q10, q11] for B00, B01, B10, B11
    
    strassen_terms = [
        # (p_blocks, q_blocks, c_block_contribs)
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [(0, 1), (3, 1)]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [(2, 1), (3, -1)]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [(1, 1), (3, 1)]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [(0, 1), (2, 1)]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [(0, -1), (1, 1)]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [(3, 1)]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [(0, 1)]),
    ]
    
    # For each Strassen term, P = sum of p_blocks[b] * A_b (2x2 each)
    # Q = sum of q_blocks[b] * B_b (2x2 each)
    # M = P * Q (2x2 result)
    # Each element M[r][c] = sum_{s=0..1} P[r][s] * Q[s][c]
    # = sum_{s} (sum_b p_blocks[b] * A_b[r][s]) * (sum_b q_blocks[b] * B_b[s][c])
    # This is a bilinear form. For each (r, s, c) triple, we get one term:
    # (sum_b p_blocks[b] * A_b[r][s]) * (sum_b q_blocks[b] * B_b[s][c])
    # The contribution to C_block[m_idx][r][c] is sign * M[r][c]
    
    # For each Strassen term and each of its 4 schoolbook sub-terms (r, s, c in 0..1):
    # U factor: sum_b p_blocks[b] * A_b[r][s] -> indices in U
    # V factor: sum_b q_blocks[b] * B_b[s][c] -> indices in V
    # W factor: sign * C_block[m_idx][r][c] -> index in W
    
    def get_A_block_indices(b_idx, r, s):
        """Get U index for A_b[r][s]"""
        # A is 4x4, blocks: A00 = A[0:2,0:2], A01 = A[0:2,2:4], etc.
        if b_idx == 0:  # A00
            gi, gj = r, s
        elif b_idx == 1:  # A01
            gi, gj = r, 2 + s
        elif b_idx == 2:  # A10
            gi, gj = 2 + r, s
        elif b_idx == 3:  # A11
            gi, gj = 2 + r, 2 + s
        return gi * m + gj  # U index: i*m + j
    
    def get_B_block_indices(b_idx, s, c):
        """Get V index for B_b[s][c]"""
        # B is 4x5, but for the 4x4 part, we use first 4 columns
        # Blocks: B00 = B[0:2,0:2], B01 = B[0:2,2:4], etc.
        if b_idx == 0:  # B00
            gj, gk = s, c
        elif b_idx == 1:  # B01
            gj, gk = s, 2 + c
        elif b_idx == 2:  # B10
            gj, gk = 2 + s, c
        elif b_idx == 3:  # B11
            gj, gk = 2 + s, 2 + c
        return gj * p + gk  # V index: j*p + k
    
    def get_C_block_indices(b_idx, r, c):
        """Get W index for C_b[r][c], where b_idx is the C block index"""
        # C is 4x5, blocks for first 4 cols: C00 = C[0:2,0:2], etc.
        if b_idx == 0:  # C00
            gi, gk = r, c
        elif b_idx == 1:  # C01
            gi, gk = r, 2 + c
        elif b_idx == 2:  # C10
            gi, gk = 2 + r, c
        elif b_idx == 3:  # C11
            gi, gk = 2 + r, 2 + c
        # W index: k*n + i
        return gk * n + gi
    
    # Process Strassen terms
    for p_blocks, q_blocks, c_contrib in strassen_terms:
        for r in range(2):
            for s in range(2):
                for c in range(2):
                    # U factor: sum_b p_blocks[b] * A_b[r][s]
                    u_list = []
                    for b in range(4):
                        if p_blocks[b] != 0:
                            idx = get_A_block_indices(b, r, s)
                            u_list.append((idx, p_blocks[b]))
                    
                    # V factor: sum_b q_blocks[b] * B_b[s][c]
                    v_list = []
                    for b in range(4):
                        if q_blocks[b] != 0:
                            idx = get_B_block_indices(b, s, c)
                            v_list.append((idx, q_blocks[b]))
                    
                    # W factor: sum over c_block_contribs
                    w_list = []
                    for c_block_idx, sign in c_contrib:
                        idx = get_C_block_indices(c_block_idx, r, c)
                        w_list.append((idx, sign))
                    
                    add_term(u_list, v_list, w_list)
    
    # Part 2: 4x4 * 4x1 (5th column of B, k=4)
    # C[i][4] = sum_{j=0..3} A[i][j] * B[j][4]
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            if col >= rank:
                break
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1  # k=4, so index = 4*4 + i = 16 + i
            col += 1
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
    print(f"Rank: {U.shape[1]}")
    
    # Verify: build the multiplication tensor
    n, m, p = 4, 4, 5
    # T[i, j, k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For random A and B, C = A @ B
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * A[i,j] ... no
    # Actually: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum... )
    # The tensor decomposition means:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j? No.
    # The standard form: C[i,k] = sum_{r} (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r]) * W[k*n+i, r]
    # Wait, that's not right either. Let me think more carefully.
    
    # The tensor T has entries T[i, j, k] = delta (i.e., the multiplication tensor)
    # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # The rank-r decomposition: T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    # where u_r is a vector over (i,j), v_r over (j,k), w_r over (i,k)
    # So T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # And C[i,k] = sum_{j} A[i,j] B[j,k] = sum_j sum_r U[i,j,r] V[j,k,r] W[i,k,r] A[i,j] B[j,k]
    # This should equal sum_r W[i,k,r] (sum_j U[i,j,r] A[i,j]) (sum_j V[j,k,r] B[j,k])
    # For this to work, we need: for each r, the vectors u_r and v_r to be "separable"
    # in a specific way. Actually, the standard decomposition is:
    # C[i,k] = sum_r W[i,k,r] * (sum_j U[i,j,r] A[i,j]) * (sum_k V[j,k,r] B[j,k])
    # No, that's not right. Let me use the correct form:
    # For matrix multiplication, the rank-r decomposition satisfies:
    # For all i, j, k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 (since A[i,j]*B[j,k] contributes to C[i,k])
    # Actually T[i,j,k] is the coefficient in the bilinear form, which is 1 for all valid (i,j,k).
    
    # Let's just verify with random matrices
    np.random.seed(42)
    A = np.random.randint(-5, 5, size=(n, m)).astype(float)
    B = np.random.randint(-5, 5, size=(m, p)).astype(float)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        # For each rank term r:
        # a_vec[i] = sum_j U[i*m+j, r] * A[i, j]
        # b_vec[k] = sum_j V[j*p+k, r] * B[j, k]
        # C[i,k] += W[k*n+i, r] * a_vec[i] * b_vec[k]
        a_vec = np.zeros(n)
        for i in range(n):
            for j in range(m):
                a_vec[i] += U[i*m+j, r] * A[i, j]
        b_vec = np.zeros(p)
        for k in range(p):
            for j in range(m):
                b_vec[k] += V[j*p+k, r] * B[j, k]
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W[k*n+i, r] * a_vec[i] * b_vec[k]
    
    print(f"Max error: {np.max(np.abs(C_computed - C_expected))}")
    if np.max(np.abs(C_computed - C_expected)) < 1e-10:
        print("VERIFICATION PASSED")
    else:
        print("VERIFICATION FAILED")
        print("C_expected:\n", C_expected)
        print("C_computed:\n", C_computed)
