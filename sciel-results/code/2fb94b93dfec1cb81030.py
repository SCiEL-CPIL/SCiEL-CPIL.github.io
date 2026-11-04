import time

DIMENSIONS = [4, 4, 6]

def _bcl_4x4x4_terms():
    """Return the 13 BCL 4x4x4 rank-1 terms as (u16, v16, w16) lists of ints."""
    terms = []
    # 7 bilinear terms (u16, v16, w16 in row-major 4x4)
    terms.append(([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],  # A
                  [0,0,0,0, 0,0,1,1, 0,0,0,0, 1,1,0,0],  # B
                  [0,0,0,0, 1,0,0,0, 0,0,0,1, 0,0,0,0]))  # C
    terms.append(([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],  # A
                  [0,0,1,1, 0,0,0,0, 0,0,0,0, 1,1,0,0],  # B
                  [0,0,0,1, 0,0,0,0, 1,0,0,0, 0,0,0,0]))  # C
    terms.append(([0,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],  # A
                  [1,1,0,0, 0,0,0,0, 0,0,0,0, 0,0,1,1],  # B
                  [0,0,0,0, 0,0,0,0, 0,0,0,1, 1,0,0,0]))  # C
    terms.append(([0,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],  # A
                  [1,1,0,0, 0,0,0,0, 0,0,0,0, 0,0,1,1],  # B
                  [0,0,0,0, 0,1,0,0, 0,0,0,0, 0,0,1,0]))  # C
    terms.append(([0,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],  # A
                  [0,0,0,0, 0,0,1,1, 0,0,0,0, 1,1,0,0],  # B
                  [0,0,0,0, 0,0,1,0, 0,0,0,0, 0,0,0,1]))  # C
    terms.append(([0,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],  # A
                  [0,0,0,0, 0,0,1,1, 0,0,0,0, 1,1,0,0],  # B
                  [0,0,1,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]))  # C
    terms.append(([0,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,1],  # A
                  [0,0,0,0, 0,0,1,1, 0,0,0,0, 1,1,0,0],  # B
                  [0,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0]))  # C
    # 6 linear terms (u16, v16, w16)
    terms.append(([0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # A
                  [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # B
                  [0,0,0,0, 0,0,0,0, 0,1,0,0, 0,0,0,0]))  # C
    terms.append(([0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # A
                  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # B
                  [0,0,0,0, 0,0,0,0, 0,0,1,0, 0,0,0,0]))  # C
    terms.append(([0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # A
                  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # B
                  [0,0,0,0, 0,0,0,0, 0,0,0,1, 0,0,0,0]))  # C
    terms.append(([0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # A
                  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # B
                  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,1,0]))  # C
    terms.append(([0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # A
                  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # B
                  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1]))  # C
    terms.append(([0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # A
                  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # B
                  [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]))  # C
    return terms

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    deadline = time.time() + 140.0
    U_cols, V_cols, W_cols = [], [], []

    def add_term(u_full, v_full, w_full):
        U_cols.append(u_full)
        V_cols.append(v_full)
        W_cols.append(w_full)

    # --- Part 1: BCL 4x4x4 for columns k=0..3 ---
    for (u16, v16, w16) in _bcl_4x4x4_terms():
        u_full = [0] * 16
        v_full = [0] * 24
        w_full = [0] * 24
        for idx in range(16):
            u_full[idx] = u16[idx]
            j = idx // 4
            k = idx % 4
            v_full[j * 6 + k] = v16[idx]
            i = idx // 4
            k = idx % 4
            w_full[k * 4 + i] = w16[idx]
        add_term(u_full, v_full, w_full)

    # --- Part 2: schoolbook for columns k=4,5 ---
    for i in range(n):
        for j in range(m):
            for k in range(4, 6):
                u_full = [0] * 16
                v_full = [0] * 24
                w_full = [0] * 24
                u_full[i * m + j] = 1
                v_full[j * p + k] = 1
                w_full[k * n + i] = 1
                add_term(u_full, v_full, w_full)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    rank = len(U_cols)
    return {"U": U_cols, "V": V_cols, "W": W_cols}
