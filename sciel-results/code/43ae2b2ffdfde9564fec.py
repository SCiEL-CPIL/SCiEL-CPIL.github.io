import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 115.0

    # Build schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try rank-22 decomposition: 2x(2x2x5) where 2x2x5 = Strassen(2x2x2) + schoolbook(2x2x3)
    # Each 2x2x5 block: 7 (Strassen) + 4 (schoolbook) = 11 rank
    # Two blocks: 22 rank total

    total_rank = 22
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_strassen_2x2x2(U, V, W, a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off):
        """Add a Strassen 2x2x2 block. Returns new col offset."""
        # Strassen patterns for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6

        strassen = [
            # (U_pattern, V_pattern, W_pattern)
            # U is 2x2 (4 entries), V is 2x2 (4 entries), W is 2x2 (4 entries)
            # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
            # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            # M2: (A10+A11)B00 -> C10+=1
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
            # M3: A00(B10-B11) -> C01+=1, C11+=1
            ([1,0,0,0], [0,1,-1,0], [0,1,0,1]),
            # M4: A11(B01-B00) -> C00+=1, C10+=1
            ([0,0,0,1], [-1,1,0,0], [1,0,1,0]),
            # M5: (A00+A01)B11 -> C00-=1, C01+=1
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
            # M6: (A10-A00)(B00+B01) -> C11+=1
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
            # M7: (A01-A11)(B10+B11) -> C00+=1
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
        ]

        for u_pat, v_pat, w_pat in strassen:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_off + i_local
                    j_global = a_j_off + j_local
                    U[i_global * m + j_global, col_off] += val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = b_j_off + j_local
                    k_global = b_k_off + k_local
                    V[j_global * p + k_global, col_off] += val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    k_local = idx // 2
                    i_local = idx % 2
                    i_global = c_i_off + i_local
                    k_global = c_k_off + k_local
                    W[k_global * n + i_global, col_off] += val
            col_off += 1
        return col_off

    def add_schoolbook_2x2x3(U, V, W, a_i_off, a_j_off, b_j_off, b_k_off, c_i_off, c_k_off, col_off):
        """Add a schoolbook 2x2x3 block. Returns new col offset."""
        # For 2x2x3: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        # 4 rank-1 terms:
        # (A[0,0], B[0,*], C[0,*])
        # (A[0,1], B[1,*], C[0,*])
        # (A[1,0], B[0,*], C[1,*])
        # (A[1,1], B[1,*], C[1,*])

        patterns = [
            # (U_pattern, V_pattern, W_pattern)
            # U is 2x2 (4 entries), V is 2x3 (6 entries), W is 2x3 (6 entries)
            # U[i*2+j], V[j*3+k], W[k*2+i]
            ([1,0,0,0], [1,0,0,0,0,0], [1,0,0,0,0,0]),  # A[0,0]*B[0,*] -> C[0,*]
            ([0,1,0,0], [0,0,0,1,0,0], [1,0,0,0,0,0]),  # A[0,1]*B[1,*] -> C[0,*]
            ([0,0,1,0], [1,0,0,0,0,0], [0,0,0,1,0,0]),  # A[1,0]*B[0,*] -> C[1,*]
            ([0,0,0,1], [0,0,0,1,0,0], [0,0,0,1,0,0]),  # A[1,1]*B[1,*] -> C[1,*]
        ]

        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_off + i_local
                    j_global = a_j_off + j_local
                    U[i_global * m + j_global, col_off] += val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 3
                    k_local = idx % 3
                    j_global = b_j_off + j_local
                    k_global = b_k_off + k_local
                    V[j_global * p + k_global, col_off] += val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    k_local = idx // 2
                    i_local = idx % 2
                    i_global = c_i_off + i_local
                    k_global = c_k_off + k_local
                    W[k_global * n + i_global, col_off] += val
            col_off += 1
        return col_off

    # Block 1: A rows 0-1, cols 0-1 times B rows 0-1, all 5 cols
    # A[0:2, 0:2] * B[0:2, 0:5] -> C[0:2, 0:5]
    col = 0
    col = add_strassen_2x2x2(U, V, W, 0, 0, 0, 0, 0, 0, col)  # 2x2x2 part, cols 0-1
    col = add_schoolbook_2x2x3(U, V, W, 0, 0, 0, 2, 0, 2, col)  # 2x2x3 part, cols 2-4

    # Block 2: A rows 0-1, cols 2-3 times B rows 2-3, all 5 cols
    # A[0:2, 2:4] * B[2:4, 0:5] -> C[0:2, 0:5]
    col = add_strassen_2x2x2(U, V, W, 0, 2, 2, 0, 0, 0, col)  # 2x2x2 part, cols 0-1
    col = add_schoolbook_2x2x3(U, V, W, 0, 2, 2, 2, 0, 2, col)  # 2x2x3 part, cols 2-4

    assert col == total_rank, f"Expected rank {total_rank}, got {col}"

    def verify_decomposition(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        # Check all diagonal entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i * m + j]
                    v_val = V_mat[j * p + k]
                    w_val = W_mat[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check all off-diagonal entries
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    # Verify the rank-22 decomposition
    if verify_decomposition(U, V, W):
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        best_result = {"U": U_list, "V": V_list, "W": W_list}
        best_rank = total_rank

    return best_result
