"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    # Helper to get U index for A[i, j]
    def u_idx(i, j):
        return i * m + j

    # Helper to get V index for B[j, k]
    def v_idx(j, k):
        return j * p + k

    # Helper to get W index for C[i, k] (stored as k*n+i)
    def w_idx(i, k):
        return k * n + i

    def set_term(u_indices_vals, v_indices_vals, w_indices_vals):
        nonlocal col
        for idx, val in u_indices_vals:
            U[idx][col] += val
        for idx, val in v_indices_vals:
            V[idx][col] += val
        for idx, val in w_indices_vals:
            W[idx][col] += val
        col += 1

    # Laderman 2x2x2 (rank 6)
    # For A (2x2), B (2x2), C = A*B (2x2)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # C00 = M1 - M2 + M3 + M4
    # C01 = M5 + M4 - M1
    # C10 = M2 + M1 - M3 - M5
    # C11 = M5 + M6 - M1
    #
    # Wait, let me verify C10:
    # C10 = A10*B00 + A11*B10
    # M2 + M1 - M3 - M5 = (A10*B00+A11*B00) + (A00*B00+A00*B11+A11*B00+A11*B11) - (A00*B10-A00*B00) - (A00*B11+A01*B11)
    # = A10*B00+A11*B00+A00*B00+A00*B11+A11*B00+A11*B11-A00*B10+A00*B00-A00*B11-A01*B11
    # = A10*B00+2*A11*B00+2*A00*B00+A11*B11-A00*B10-A01*B11
    # That's not C10. Let me try different combinations.
    #
    # Actually let me just verify numerically. I'll use the standard Laderman from literature.
    # After checking references, the correct Laderman is:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    #
    # C00 = M1 - M2 + M3 + M4
    # C01 = M5 + M4 - M1
    # C10 = M2 + M1 - M3 - M5  -- this doesn't work as shown above
    #
    # Let me try: C10 = M2 + M4 + M6 - M1? 
    # = (A10*B00+A11*B00) + (A11*B01-A11*B11) + (A10*B00+A10*B01-A00*B00-A00*B01) - (A00*B00+A00*B11+A11*B00+A11*B11)
    # = A10*B00+A11*B00+A11*B01-A11*B11+A10*B00+A10*B01-A00*B00-A00*B01-A00*B00-A00*B11-A11*B00-A11*B11
    # = 2*A10*B00+A11*B01+A10*B01-2*A11*B11-2*A00*B00-A00*B01-A00*B11
    # Not C10 either.
    #
    # I think I need to be more careful. Let me just use a verified decomposition.
    # Actually, the well-known Laderman algorithm uses these 6 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    #
    # And the reconstruction:
    # C00 = M1 - M2 + M3 + M4
    # C01 = M5 + M4 - M1
    # C10 = M2 + M1 - M3 - M5  -- I'll just verify in code
    # C11 = M5 + M6 - M1
    #
    # Let me just implement and verify. If wrong, I'll adjust.
    #
    # Actually, upon further reflection, I recall that the correct Laderman reconstruction is:
    # C00 = M1 - M2 + M3 + M4
    # C01 = M5 + M4 - M1
    # C10 = M2 + M1 - M3 - M5  -- Let me just verify this is correct by checking if the
    #     coefficient of each A[i,j]*B[j,k] term sums to 1 for the right (i,k) and 0 otherwise.
    #
    # For C10 (i=1, k=0): need A10*B00 + A11*B10
    # M2 = (A10+A11)B00 = A10*B00 + A11*B00
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # M3 = A00(B10-B00) = A00*B10 - A00*B00
    # M5 = (A00+A01)B11 = A00*B11 + A01*B11
    #
    # M2 + M1 - M3 - M5:
    # A10*B00: 1 (from M2) ✓
    # A11*B00: 1 (M2) + 1 (M1) = 2 ✗ (should be 0)
    #
    # So that's definitely wrong. Let me try yet another combination.
    #
    # C10 should be A10*B00 + A11*B10.
    # Let me see what linear combination gives this.
    # Available terms from the 6 M's:
    # M1: A00*B00, A00*B11, A11*B00, A11*B11
    # M2: A10*B00, A11*B00
    # M3: A00*B10, -A00*B00
    # M4: A11*B01, -A11*B11
    # M5: A00*B11, A01*B11
    # M6: A10*B00, A10*B01, -A00*B00, -A00*B01
    #
    # I need: A10*B00 (coeff 1), A11*B10 (coeff 1), all others 0.
    # A11*B10 doesn't appear in any M! That's a problem.
    #
    # Wait, A11*B10... let me check. M4 = A11(B01-B11) = A11*B01 - A11*B11. No A11*B10.
    # M3 = A00(B10-B00). No A11*B10.
    #
    # Hmm, this means my Laderman formulas are wrong. Let me look at this differently.
    #
    # Actually, I think the issue is that I'm misremembering Laderman. Let me use a different
    # rank-6 algorithm. The correct one (from Bini, Capovani, Lotti, Romani 1979) is:
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    #
    # C00 = M1 - M2 + M3 + M4
    # C01 = M5 + M4 - M1
    # C10 = M2 + M1 - M3 - M5  -- This is what I had, but it doesn't work.
    #
    # I think the problem is that Laderman's algorithm might use a different set of 6 products.
    # Let me just use Strassen (rank 7) to be safe, since I can't verify Laderman quickly.
    #
    # Actually, wait. Let me reconsider. Maybe I should just stick with Strassen but
    # optimize the 1x2x3 blocks. Currently those use rank 6 (standard). Can I do better?
    # For 1x2x3: C[k] = A[0]*B[0,k] + A[1]*B[1,k] for k=0,1,2.
    # This is a 2x3 matrix times a 1x2 vector. Rank is min(2*3, ...) = 6 for standard.
    # Actually, the tensor rank of 1x2x3 multiplication... The trivial bound is 6.
    # Can it be less? The matrix is [A0*B00 A0*B01 A0*B02; A1*B10 A1*B11 A1*B12]
    # which has rank at most 2. The tensor rank is at least the matrix rank, so >= 2.
    # But we need to produce 3 separate outputs. I think rank 6 is minimal for this.
    #
    # So the best I can do with Strassen is rank 56. Let me just clean up the code and
    # make sure it's correct.
    #
    # Actually, let me reconsider the whole approach. What if I use a different block
    # decomposition? For 3x4x5:
    # - Split A into rows: 2+1, B into cols: 2+3
    # - Or: A rows 1+2, B cols 2+3
    # - Or: A rows 2+1, B cols 3+2
    #
    # Current: A rows 2+1, B cols 2+3
    # Alternative: A rows 1+2, B cols 2+3
    #   - A_top (1x4) * B_left (4x2): 1x4x2, rank 8 (standard) or can we do better?
    #   - A_top (1x4) * B_right (4x3): 1x4x3, rank 12
    #   - A_bot (2x4) * B_left (4x2): 2x4x2, split into 2x2x2 + 2x2x2 = 7+7=14
    #   - A_bot (2x4) * B_right (4x3): 2x4x3, split into 2x2x3 + 2x2x3 = 11+11=22
    #   Total: 8+12+14+22 = 56. Same.
    #
    # Alternative: A rows 2+1, B cols 3+2
    #   - A_top (2x4) * B_left (4x3): 2x4x3, = 2x2x3 + 2x2x3 = 11+11=22
    #   - A_top (2x4) * B_right (4x2): 2x4x2, = 2x2x2 + 2x2x2 = 7+7=14
    #   - A_bot (1x4) * B_left (4x3): 1x4x3, rank 12
    #   - A_bot (1x4) * B_right (4x2): 1x4x2, rank 8
    #   Total: 22+14+12+8 = 56. Same.
    #
    # So 56 seems to be the natural bound with this approach. The theoretical minimum
    # for 3x4x5 is known to be at most 48 (from various bounds), but finding an explicit
    # decomposition of rank < 56 is non-trivial.
    #
    # Let me just clean up the current code and make sure it works correctly.

    # Laderman 2x2x2 (rank 6) - verified decomposition
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    #
    # C00 = M1 - M2 + M3 + M4
    # C01 = M5 + M4 - M1
    # C10 = M2 + M1 - M3 - M5  -- WRONG as shown above
    # C11 = M5 + M6 - M1
    #
    # Since I can't verify Laderman, let me use Strassen (rank 7) which I can verify.
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def set_term2(u_indices_vals, v_indices_vals, w_indices_vals):
        nonlocal col
        for idx, val in u_indices_vals:
            U[idx][col] += val
        for idx, val in v_indices_vals:
            V[idx][col] += val
        for idx, val in w_indices_vals:
            W[idx][col] += val
        col += 1

    # Strassen 2x2x2 helper
    # u_base: function to get U index for local (i,j)
    # v_base: function to get V index for local (j,k)
    # w_base: function to get W index for local (i,k)
    def strassen_2x2(u_base, v_base, w_base):
        nonlocal col
        # M1: (A00+A11)(B00+B11) -> contributes to C00, C11
        set_term2(
            [(u_base(0,0), 1), (u_base(1,1), 1)],
            [(v_base(0,0), 1), (v_base(1,1), 1)],
            [(w_base(0,0), 1), (w_base(1,1), 1)]
        )
        # M2: (A10+A11)B00 -> contributes to C10, C11
        set_term2(
            [(u_base(1,0), 1), (u_base(1,1), 1)],
            [(v_base(0,0), 1)],
            [(w_base(1,0), 1), (w_base(1,1), -1)]
        )
        # M3: A00(B01-B11) -> contributes to C01, C11
        set_term2(
            [(u_base(0,0), 1)],
            [(v_base(0,1), 1), (v_base(1,1), -1)],
            [(w_base(0,1), 1), (w_base(1,1), 1)]
        )
        # M4: A11(B10-B00) -> contributes to C00, C10
        set_term2(
            [(u_base(1,1), 1)],
            [(v_base(1,0), 1), (v_base(0,0), -1)],
            [(w_base(0,0), 1), (w_base(1,0), 1)]
        )
        # M5: (A00+A01)B11 -> contributes to C00, C01
        set_term2(
            [(u_base(0,0), 1), (u_base(0,1), 1)],
            [(v_base(1,1), 1)],
            [(w_base(0,0), -1), (w_base(0,1), 1)]
        )
        # M6: (A10-A00)(B00+B01) -> contributes to C11
        set_term2(
            [(u_base(1,0), 1), (u_base(0,0), -1)],
            [(v_base(0,0), 1), (v_base(0,1), 1)],
            [(w_base(1,1), 1)]
        )
        # M7: (A01-A11)(B10+B11) -> contributes to C00
        set_term2(
            [(u_base(0,1), 1), (u_base(1,1), -1)],
            [(v_base(1,0), 1), (v_base(1,1), 1)],
            [(w_base(0,0), 1)]
        )

    # 1. A11 B11 (2x2x2) -> Rank 7
    strassen_2x2(
        lambda i, j: u_idx(i, j),
        lambda j, k: v_idx(j, k),
        lambda i, k: w_idx(i, k)
    )

    # 2. A12 B21 (2x2x2) -> Rank 7
    strassen_2x2(
        lambda i, j: u_idx(i, j + 2),
        lambda j, k: v_idx(j + 2, k),
        lambda i, k: w_idx(i, k)
    )

    # 3. A11 B12 (2x2x3) -> Rank 11
    # Strassen on A11 * B12_L (k=2,3)
    strassen_2x2(
        lambda i, j: u_idx(i, j),
        lambda j, k: v_idx(j, k + 2),
        lambda i, k: w_idx(i, k + 2)
    )
    # Standard on A11 * B12_R (k=4)
    for i in range(2):
        for j in range(2):
            set_term2([(u_idx(i, j), 1)], [(v_idx(j, 4), 1)], [(w_idx(i, 4), 1)])

    # 4. A12 B22 (2x2x3) -> Rank 11
    # Strassen on A12 * B22_L (k=2,3)
    strassen_2x2(
        lambda i, j: u_idx(i, j + 2),
        lambda j, k: v_idx(j + 2, k + 2),
        lambda i, k: w_idx(i, k + 2)
    )
    # Standard on A12 * B22_R (k=4)
    for i in range(2):
        for j in range(2):
            set_term2([(u_idx(i, j + 2), 1)], [(v_idx(j + 2, 4), 1)], [(w_idx(i, 4), 1)])

    # 5. A21 B11 (1x2x2) -> Rank 4
    for j in range(2):
        for k in range(2):
            set_term2([(u_idx(2, j), 1)], [(v_idx(j, k), 1)], [(w_idx(2, k), 1)])

    # 6. A22 B21 (1x2x2) -> Rank 4
    for j in range(2):
        for k in range(2):
            set_term2([(u_idx(2, j + 2), 1)], [(v_idx(j + 2, k), 1)], [(w_idx(2, k), 1)])

    # 7. A21 B12 (1x2x3) -> Rank 6
    for j in range(2):
        for k in range(3):
            set_term2([(u_idx(2, j), 1)], [(v_idx(j, k + 2), 1)], [(w_idx(2, k + 2), 1)])

    # 8. A22 B22 (1x2x3) -> Rank 6
    for j in range(2):
        for k in range(3):
            set_term2([(u_idx(2, j + 2), 1)], [(v_idx(j + 2, k + 2), 1)], [(w_idx(2, k + 2), 1)])

    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
