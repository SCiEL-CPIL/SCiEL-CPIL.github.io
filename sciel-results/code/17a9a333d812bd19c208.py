"""Matrix multiplication 4x5x5: find a rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Uses a block-partitioned approach:
      - Split 4x5x5 into 8 sub-blocks of size (2x2x2) or (2x3x2) or (2x2x3) or (2x3x3).
      - Use Strassen (rank 7) for 2x2x2 blocks.
      - Use rank 9 for 2x3x2 and 2x2x3 blocks (split into 2x2x2 + 2x1x2 or similar).
      - Use rank 12 for 2x3x3 blocks (split into 2x2x2 + 2x1x3 + 2x2x1 + 2x1x1, or better).
    Total rank = 7+9+9+12+7+9+9+12 = 74.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 5, 5

    # We'll build U, V, W as lists of lists, accumulating rank-1 terms
    U_terms = []
    V_terms = []
    W_terms = []

    def add_term(u_vec, v_vec, w_vec):
        """Add a single rank-1 term. u_vec is length n*m, v_vec length m*p, w_vec length p*n."""
        U_terms.append(u_vec)
        V_terms.append(v_vec)
        W_terms.append(w_vec)

    def zero_u():
        return [0.0] * (n * m)

    def zero_v():
        return [0.0] * (m * p)

    def zero_w():
        return [0.0] * (p * n)

    def set_u(u, i, j, val):
        u[i * m + j] = val

    def set_v(v, j, k, val):
        v[j * p + k] = val

    def set_w(w, k, i, val):
        w[k * n + i] = val

    # Strassen 2x2x2 decomposition: 7 rank-1 terms
    # For matrices A (2x2), B (2x2), C = A*B (2x2)
    # C[0,0] = M1 - M4 + M6 + M7
    # C[0,1] = M3 + M4
    # C[1,0] = M1 + M2
    # C[1,1] = M5 - M2 + M3 + M7
    # Where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)

    def strassen_2x2x2(i_base, j_base, k_base):
        """Add Strassen terms for a 2x2x2 block starting at global indices (i_base, j_base, k_base)."""
        # M1
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, i_base, j_base, 1); set_u(u, i_base+1, j_base+1, 1)
        set_v(v, j_base, k_base, 1); set_v(v, j_base+1, k_base+1, 1)
        set_w(w, k_base, i_base, 1); set_w(w, k_base+1, i_base+1, 1)
        add_term(u, v, w)

        # M2
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, i_base+1, j_base, 1); set_u(u, i_base+1, j_base+1, 1)
        set_v(v, j_base, k_base, 1)
        set_w(w, k_base, i_base+1, 1); set_w(w, k_base+1, i_base+1, 1)
        add_term(u, v, w)

        # M3
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, i_base, j_base, 1)
        set_v(v, j_base, k_base+1, 1); set_v(v, j_base+1, k_base+1, -1)
        set_w(w, k_base+1, i_base, 1); set_w(w, k_base+1, i_base+1, 1)
        add_term(u, v, w)

        # M4
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, i_base+1, j_base+1, 1)
        set_v(v, j_base+1, k_base, 1); set_v(v, j_base, k_base, -1)
        set_w(w, k_base, i_base, 1); set_w(w, k_base, i_base+1, 1)
        add_term(u, v, w)

        # M5
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, i_base, j_base, 1); set_u(u, i_base, j_base+1, 1)
        set_v(v, j_base+1, k_base+1, 1)
        set_w(w, k_base+1, i_base, 1); set_w(w, k_base+1, i_base+1, 1)
        add_term(u, v, w)

        # M6
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, i_base+1, j_base, 1); set_u(u, i_base, j_base, -1)
        set_v(v, j_base, k_base, 1); set_v(v, j_base, k_base+1, 1)
        set_w(w, k_base, i_base, 1); set_w(w, k_base+1, i_base, 1)
        add_term(u, v, w)

        # M7
        u = zero_u(); v = zero_v(); w = zero_w()
        set_u(u, i_base, j_base+1, 1); set_u(u, i_base+1, j_base+1, -1)
        set_v(v, j_base+1, k_base, 1); set_v(v, j_base+1, k_base+1, 1)
        set_w(w, k_base, i_base+1, 1); set_w(w, k_base+1, i_base+1, 1)
        add_term(u, v, w)

    # Schoolbook 2x1x1: rank 2
    def schoolbook_2x1x1(i_base, j_base, k_base):
        for di in range(2):
            for dj in range(1):
                for dk in range(1):
                    u = zero_u(); v = zero_v(); w = zero_w()
                    set_u(u, i_base + di, j_base + dj, 1)
                    set_v(v, j_base + dj, k_base + dk, 1)
                    set_w(w, k_base + dk, i_base + di, 1)
                    add_term(u, v, w)

    # Schoolbook 1x2x1: rank 2
    def schoolbook_1x2x1(i_base, j_base, k_base):
        for di in range(1):
            for dj in range(2):
                for dk in range(1):
                    u = zero_u(); v = zero_v(); w = zero_w()
                    set_u(u, i_base + di, j_base + dj, 1)
                    set_v(v, j_base + dj, k_base + dk, 1)
                    set_w(w, k_base + dk, i_base + di, 1)
                    add_term(u, v, w)

    # Schoolbook 1x1x2: rank 2
    def schoolbook_1x1x2(i_base, j_base, k_base):
        for di in range(1):
            for dj in range(1):
                for dk in range(2):
                    u = zero_u(); v = zero_v(); w = zero_w()
                    set_u(u, i_base + di, j_base + dj, 1)
                    set_v(v, j_base + dj, k_base + dk, 1)
                    set_w(w, k_base + dk, i_base + di, 1)
                    add_term(u, v, w)

    # Schoolbook 2x2x1: rank 4
    def schoolbook_2x2x1(i_base, j_base, k_base):
        for di in range(2):
            for dj in range(2):
                for dk in range(1):
                    u = zero_u(); v = zero_v(); w = zero_w()
                    set_u(u, i_base + di, j_base + dj, 1)
                    set_v(v, j_base + dj, k_base + dk, 1)
                    set_w(w, k_base + dk, i_base + di, 1)
                    add_term(u, v, w)

    # Schoolbook 2x1x2: rank 4
    def schoolbook_2x1x2(i_base, j_base, k_base):
        for di in range(2):
            for dj in range(1):
                for dk in range(2):
                    u = zero_u(); v = zero_v(); w = zero_w()
                    set_u(u, i_base + di, j_base + dj, 1)
                    set_v(v, j_base + dj, k_base + dk, 1)
                    set_w(w, k_base + dk, i_base + di, 1)
                    add_term(u, v, w)

    # Schoolbook 1x2x2: rank 4
    def schoolbook_1x2x2(i_base, j_base, k_base):
        for di in range(1):
            for dj in range(2):
                for dk in range(2):
                    u = zero_u(); v = zero_v(); w = zero_w()
                    set_u(u, i_base + di, j_base + dj, 1)
                    set_v(v, j_base + dj, k_base + dk, 1)
                    set_w(w, k_base + dk, i_base + di, 1)
                    add_term(u, v, w)

    # For 2x3x2: split j into 2+1, giving 2x2x2 (Strassen, rank 7) + 2x1x2 (schoolbook, rank 4) = 11
    # But we can do better: 2x3x2 can be done with rank 9? Let's use schoolbook: 2*3*2=12
    # Actually, let's just use schoolbook for non-2x2x2 blocks to be safe.
    # 2x3x2: rank 12
    # 2x2x3: rank 12
    # 2x3x3: rank 18
    # Total: 7+12+12+18+7+12+12+18 = 98. Not an improvement.

    # Let me reconsider. The directive said rank 45 is possible. Let me try a different partition.
    # What if we partition 4x5x5 as:
    # i: 2+2, j: 2+2+1, k: 2+2+1
    # Blocks:
    # (2,2,2): rank 7
    # (2,2,2): rank 7
    # (2,2,1): rank 4 (schoolbook)
    # (2,2,2): rank 7
    # (2,2,2): rank 7
    # (2,2,1): rank 4
    # (2,1,2): rank 4
    # (2,1,2): rank 4
    # (2,1,1): rank 2
    # Total: 7+7+4+7+7+4+4+4+2 = 46. That's much better!

    # Let me implement this partition: i = 2+2, j = 2+2+1, k = 2+2+1
    # Block (i_offset, j_offset, k_offset, i_size, j_size, k_size)
    blocks = [
        # (i_off, j_off, k_off, i_sz, j_sz, k_sz)
        (0, 0, 0, 2, 2, 2),  # 2x2x2: Strassen rank 7
        (0, 2, 0, 2, 2, 2),  # 2x2x2: Strassen rank 7
        (0, 4, 0, 2, 1, 2),  # 2x1x2: schoolbook rank 4
        (0, 0, 2, 2, 2, 2),  # 2x2x2: Strassen rank 7
        (0, 2, 2, 2, 2, 2),  # 2x2x2: Strassen rank 7
        (0, 4, 2, 2, 1, 2),  # 2x1x2: schoolbook rank 4
        (2, 0, 0, 2, 2, 2),  # 2x2x2: Strassen rank 7
        (2, 2, 0, 2, 2, 2),  # 2x2x2: Strassen rank 7
        (2, 4, 0, 2, 1, 2),  # 2x1x2: schoolbook rank 4
        (2, 0, 2, 2, 2, 2),  # 2x2x2: Strassen rank 7
        (2, 2, 2, 2, 2, 2),  # 2x2x2: Strassen rank 7
        (2, 4, 2, 2, 1, 2),  # 2x1x2: schoolbook rank 4
    ]

    # Wait, that's 12 blocks, but 4x5x5 = 4*5*5 = 100 elements.
    # 2*2*2*8 + 2*1*2*4 = 32 + 16 = 48. That's not 100.
    # I made an error. 4*5*5 = 100. My partition only covers 48.

    # Let me redo the partition. n=4, m=5, p=5.
    # i: 2+2, j: 2+2+1, k: 2+2+1
    # Number of blocks: 2 * 3 * 3 = 18 blocks.
    # Let me list all 18:
    blocks = []
    for ii in range(2):
        for jj in range(3):
            for kk in range(3):
                i_off = ii * 2
                j_off = jj * 2 if jj < 2 else 4
                k_off = kk * 2 if kk < 2 else 4
                i_sz = 2
                j_sz = 2 if jj < 2 else 1
                k_sz = 2 if kk < 2 else 1
                blocks.append((i_off, j_off, k_off, i_sz, j_sz, k_sz))

    # Now add terms for each block
    for (i_off, j_off, k_off, i_sz, j_sz, k_sz) in blocks:
        if i_sz == 2 and j_sz == 2 and k_sz == 2:
            strassen_2x2x2(i_off, j_off, k_off)
        elif i_sz == 2 and j_sz == 1 and k_sz == 2:
            schoolbook_2x1x2(i_off, j_off, k_off)
        elif i_sz == 2 and j_sz == 2 and k_sz == 1:
            schoolbook_2x2x1(i_off, j_off, k_off)
        elif i_sz == 2 and j_sz == 1 and k_sz == 1:
            schoolbook_2x1x1(i_off, j_off, k_off)
        else:
            # General schoolbook fallback
            for di in range(i_sz):
                for dj in range(j_sz):
                    for dk in range(k_sz):
                        u = zero_u(); v = zero_v(); w = zero_w()
                        set_u(u, i_off + di, j_off + dj, 1)
                        set_v(v, j_off + dj, k_off + dk, 1)
                        set_w(w, k_off + dk, i_off + di, 1)
                        add_term(u, v, w)

    rank = len(U_terms)

    # Convert to numpy arrays
    U = np.array(U_terms, dtype=float).T if rank > 0 else np.zeros((n*m, 0))
    V = np.array(V_terms, dtype=float).T if rank > 0 else np.zeros((m*p, 0))
    W = np.array(W_terms, dtype=float).T if rank > 0 else np.zeros((p*n, 0))

    # Verify correctness
    def verify(U, V, W, rank):
        if rank == 0:
            return False
        T = np.zeros((n, m, p))
        for t in range(rank):
            u = U[:, t].reshape(n, m)
            v = V[:, t].reshape(m, p)
            w = W[:, t].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-10)

    if verify(U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 100
    U_sb = np.zeros((n * m, n * m * p))
    V_sb = np.zeros((m * p, n * m * p))
    W_sb = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
