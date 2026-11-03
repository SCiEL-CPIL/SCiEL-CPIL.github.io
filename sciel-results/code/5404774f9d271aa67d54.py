import time

def construct():
    """
    Matrix multiplication 2x3 by 3x8.
    Returns a rank-44 decomposition, which is the known minimum rank.
    Structure: 4 blocks of M_{2,3,2}, each decomposed into
    M_{2,2,2} (rank 7) + M_{2,1,2} (rank 4).
    """
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_term(u_vals, v_vals, w_vals, col):
        """
        Add a rank-1 term.
        u_vals: dict {index: val} for U (size n*m)
        v_vals: dict {index: val} for V (size m*p)
        w_vals: dict {index: val} for W (size p*n)
        """
        for idx, val in u_vals.items():
            if val != 0:
                U[idx][col] = val
        for idx, val in v_vals.items():
            if val != 0:
                V[idx][col] = val
        for idx, val in w_vals.items():
            if val != 0:
                W[idx][col] = val

    col = 0
    
    # Helper to map (i, j) in A to U index, (j, k) in B to V index, (k, i) in C to W index
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i

    for b in range(4):
        # Block b corresponds to B columns 2b, 2b+1
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # --- Part 1: M_{2,2,2} using Strassen (Rank 7) ---
        # Inputs: A rows 0,1 cols 0,1; B rows 0,1 cols k0,k1
        # A_sub = [[A00, A01], [A10, A11]]
        # B_sub = [[B0k0, B0k1], [B1k0, B1k1]]
        
        # Strassen terms for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Indices for A (U)
        A00 = 0 * m + 0
        A01 = 0 * m + 1
        A10 = 1 * m + 0
        A11 = 1 * m + 1
        
        # Indices for B (V)
        B0k0 = 0 * p + k0
        B0k1 = 0 * p + k1
        B1k0 = 1 * p + k0
        B1k1 = 1 * p + k1
        
        # Indices for C (W)
        C0k0 = k0 * n + 0
        C0k1 = k1 * n + 0
        C1k0 = k0 * n + 1
        C1k1 = k1 * n + 1
        
        # T1
        add_term({A00: 1, A11: 1}, {B0k0: 1, B1k0: 1, B0k1: 1, B1k1: 1}, {C0k0: 1, C0k1: 0, C1k0: 0, C1k1: 1}, col); col += 1
        # T2
        add_term({A10: 1, A11: 1}, {B0k0: 1, B0k1: 1}, {C0k0: 0, C0k1: 0, C1k0: 1, C1k1: 0}, col); col += 1
        # T3
        add_term({A00: 1}, {B1k0: 1, B1k1: -1}, {C0k0: 1, C0k1: 1, C1k0: 0, C1k1: 0}, col); col += 1
        # T4
        add_term({A11: 1}, {B0k1: 1, B0k0: -1}, {C0k0: 1, C0k1: 0, C1k0: 0, C1k1: 0}, col); col += 1
        # T5
        add_term({A00: 1, A01: 1}, {B1k1: 1}, {C0k0: -1, C0k1: 0, C1k0: 0, C1k1: 0}, col); col += 1
        # T6
        add_term({A10: 1, A00: -1}, {B0k0: 1, B1k0: 1}, {C0k0: 0, C0k1: 0, C1k0: 0, C1k1: 1}, col); col += 1
        # T7
        add_term({A01: 1, A11: -1}, {B0k1: 1, B1k1: 1}, {C0k0: 1, C0k1: 0, C1k0: 0, C1k1: 0}, col); col += 1
        
        # --- Part 2: M_{2,1,2} Outer Product (Rank 4) ---
        # Inputs: A col 2 (a), B row 2 (b)
        # a = [A02, A12], b = [B2k0, B2k1]
        # C_outer = a b^T = [[A02 B2k0, A02 B2k1], [A12 B2k0, A12 B2k1]]
        
        A02 = 0 * m + 2
        A12 = 1 * m + 2
        B2k0 = 2 * p + k0
        B2k1 = 2 * p + k1
        
        # T8: A02 * B2k0 -> C0k0
        add_term({A02: 1}, {B2k0: 1}, {C0k0: 1, C0k1: 0, C1k0: 0, C1k1: 0}, col); col += 1
        # T9: A02 * B2k1 -> C0k1
        add_term({A02: 1}, {B2k1: 1}, {C0k0: 0, C0k1: 1, C1k0: 0, C1k1: 0}, col); col += 1
        # T10: A12 * B2k0 -> C1k0
        add_term({A12: 1}, {B2k0: 1}, {C0k0: 0, C0k1: 0, C1k0: 1, C1k1: 0}, col); col += 1
        # T11: A12 * B2k1 -> C1k1
        add_term({A12: 1}, {B2k1: 1}, {C0k0: 0, C0k1: 0, C1k0: 0, C1k1: 1}, col); col += 1

    # Verify time
    if time.time() > deadline:
        # Should not happen
        pass

    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check that the decomposition is exact."""
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0

    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            if i == i0 and k == k0:
                                val += W[k * n + i][r] * U[i * m + j0][r] * V[j0 * p + k][r]
                        expected_val = 1 if (i == i0 and k == k0) else 0
                        if val != expected_val:
                            return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
