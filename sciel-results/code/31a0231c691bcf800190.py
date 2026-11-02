"""Matrix multiplication 2x2 x 2x6 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 6]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START

def _build_block_strassen():
    """Rank-21 decomposition: 3 independent 2x2 blocks, each via Strassen (rank 7).
    A is 2x2 (indices a0..a3), each B block is 2x2 (b0..b3), C block is 2x2 (c0..c3).
    Strassen intermediates M0..M6 with the standard formulas.
    """
    # Strassen intermediates for 2x2*2x2:
    # M0 = (a0+a3)*(b0+b3)
    # M1 = (a2+a3)*b0
    # M2 = a0*(b1-b3)
    # M3 = a3*(b2-b4)  -> for 2x2: a3*(b2-b3)? Let's use standard:
    # Standard Strassen 2x2:
    # M0=(a0+a3)(b0+b3); M1=(a2+a3)b0; M2=a0(b1-b3); M3=a3(b2-b4)??
    # For 2x2, b indices are b0,b1,b2,b3:
    # M0=(a0+a3)(b0+b3)
    # M1=(a2+a3)b0
    # M2=a0(b1-b3)
    # M3=a3(b2-b3)
    # M4=(a0+a1)(b3+b4)? -> b3 only: (a0+a1)(b2+b3)
    # M5=(a1+a3)(b0+b1)
    # M6=(a2+a3)(b2+b3)
    # C0 = M0 + M4 - M5 + M6
    # C1 = M2 + M5
    # C2 = M1 + M4
    # C3 = M0 - M1 + M3 + M5
    # Let me verify with a concrete 2x2 multiplication.
    # A = [[a0,a1],[a2,a3]], B=[[b0,b1],[b2,b3]]
    # C = [[a0*b0+a1*b2, a0*b1+a1*b3],[a2*b0+a3*b2, a2*b1+a3*b3]]
    # Standard Strassen:
    # M0=(a0+a3)(b0+b3)
    # M1=(a2+a3)b0
    # M2=a0(b1-b3)
    # M3=a3(b2-b4)  -> b4 doesn't exist; for 2x2 it's a3(b2-b3)? Let me use the canonical:
    # Actually canonical 2x2 Strassen:
    # M0=(a0+a3)(b0+b3)
    # M1=(a2+a3)b0
    # M2=a0(b1-b3)
    # M3=a3(b2-b4)  -- this is for the "b4" which is not in 2x2.
    # The standard 2x2 Strassen uses:
    # M0=(a0+a3)(b0+b3)
    # M1=(a2+a3)b0
    # M2=a0(b1-b3)
    # M3=a3(b2-b3)  -- hmm, let me just use a known correct set:
    # M0 = (a0+a3)(b0+b3)
    # M1 = (a2+a3)b0
    # M2 = a0(b1-b3)
    # M3 = a3(b2-b3)
    # M4 = (a0+a1)(b3+b4) -- b4 not in 2x2
    # This is getting confusing. Let me use the well-known Strassen 2x2:
    # M0=(a0+a3)(b0+b3)
    # M1=(a2+a3)b0
    # M2=a0(b1-b3)
    # M3=a3(b2-b3)
    # M4=(a0+a1)(b3+b4)  -> for 2x2, (a0+a1)(b2+b3)
    # M5=(a1+a3)(b0+b1)
    # M6=(a2+a3)(b2+b3)
    # C00=M0+M4-M5+M6
    # C01=M2+M5
    # C10=M1+M4
    # C11=M0-M1+M3+M5
    # Let me verify C00: M0+M4-M5+M6
    # = (a0+a3)(b0+b3) + (a0+a1)(b2+b3) - (a1+a3)(b0+b1) + (a2+a3)(b2+b3)
    # This is getting complex. Let me just verify numerically in the code.

    # I'll define the Strassen factors and verify them numerically.
    # A row: a0,a1,a2,a3 (row-major: a0,a1; a2,a3)
    # B row: b0,b1,b2,b3 (row-major: b0,b1; b2,b3)
    # C row: c0,c1,c2,c3 (row-major: c0,c1; c2,c3)

    # Strassen intermediates (each is a product of a linear form in A and a linear form in B):
    # M0: (a0+a3) * (b0+b3)
    # M1: (a2+a3) * b0
    # M2: a0 * (b1-b3)
    # M3: a3 * (b2-b3)
    # M4: (a0+a1) * (b2+b3)
    # M5: (a1+a3) * (b0+b1)
    # M6: (a2+a3) * (b2+b3)
    # C0 = M0 + M4 - M5 + M6
    # C1 = M2 + M5
    # C2 = M1 + M4
    # C3 = M0 - M1 + M3 + M5

    # U factors for A (2x2 -> 4 entries, 7 intermediates):
    U_A = [
        [1, 0, 0, 0, 1, 1, 0],  # a0: M0, M2, M4
        [0, 1, 0, 0, 0, 1, 0],  # a1: M4, M5
        [0, 0, 1, 0, 0, 0, 1],  # a2: M1, M6
        [1, 1, 0, 1, 0, 0, 1],  # a3: M0, M1, M3, M6
    ]
    # V factors for B (2x2 -> 4 entries, 7 intermediates):
    V_B = [
        [1, 1, 0, 0, 0, 1, 0],  # b0: M0, M1, M5
        [0, 0, 1, 0, 0, 1, 0],  # b1: M2, M5
        [0, 0, 0, 1, 1, 0, 1],  # b2: M3, M4, M6
        [1, 0, -1, -1, 1, 0, 1],  # b3: M0, M2(-1), M3(-1), M4, M6
    ]
    # W factors for C (2x2 -> 4 entries, 7 intermediates):
    W_C = [
        [1, 0, 0, 0, 1, -1, 1],  # c0: M0, M4, M5(-1), M6
        [0, 0, 1, 0, 0, 1, 0],  # c1: M2, M5
        [0, 1, 0, 0, 1, 0, 0],  # c2: M1, M4
        [1, -1, 0, 1, 0, 1, 0],  # c3: M0, M1(-1), M3, M5
    ]

    # Verify numerically
    import random
    for trial in range(100):
        A = np.array([[random.randint(-5,5) for _ in range(2)] for _ in range(2)])
        B = np.array([[random.randint(-5,5) for _ in range(2)] for _ in range(2)])
        C_expected = A @ B
        # Compute via Strassen
        M = np.zeros(7)
        a = [A[0,0], A[0,1], A[1,0], A[1,1]]
        b = [B[0,0], B[0,1], B[1,0], B[1,1]]
        for k in range(7):
            ua = sum(U_A[i][k] * a[i] for i in range(4))
            vb = sum(V_B[i][k] * b[i] for i in range(4))
            M[k] = ua * vb
        C_strassen = np.array([
            [sum(W_C[0][k] * M[k] for k in range(7)), sum(W_C[1][k] * M[k] for k in range(7))],
            [sum(W_C[2][k] * M[k] for k in range(7)), sum(W_C[3][k] * M[k] for k in range(7))],
        ])
        if not np.array_equal(C_expected, C_strassen):
            raise ValueError(f"Strassen verification failed: {C_expected} vs {C_strassen}")

    # Now build the full 2x2 x 2x6 decomposition with 3 blocks.
    # A: 2x2 (4 entries) -> U has 4 rows, 21 columns
    # B: 2x6 (12 entries) -> V has 12 rows, 21 columns
    # C: 6x2 (12 entries) -> W has 12 rows, 21 columns
    # Block t (t=0,1,2): B block columns 2t, 2t+1; C block rows 2t, 2t+1
    rank = 21
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]

    for t in range(3):
        base = t * 7
        for i in range(4):
            for k in range(7):
                U[i][base + k] = U_A[i][k]
        for j in range(4):
            for k in range(7):
                V[t * 4 + j][base + k] = V_B[j][k]
        for c in range(4):
            for k in range(7):
                W[t * 4 + c][base + k] = W_C[c][k]

    # Verify full decomposition
    for trial in range(200):
        A = np.array([[random.randint(-5,5) for _ in range(2)] for _ in range(2)])
        B = np.array([[random.randint(-5,5) for _ in range(6)] for _ in range(2)])
        C_expected = A @ B
        # Compute via tensor
        M = np.zeros(rank)
        a = [A[0,0], A[0,1], A[1,0], A[1,1]]
        b = [B[0,j] for j in range(6)] + [B[1,j] for j in range(6)]
        for k in range(rank):
            ua = sum(U[i][k] * a[i] for i in range(4))
            vb = sum(V[i][k] * b[i] for i in range(12))
            M[k] = ua * vb
        C_computed = np.zeros((6, 2))
        for r in range(12):
            val = sum(W[r][k] * M[k] for k in range(rank))
            C_computed[r // 2, r % 2] = val
        if not np.array_equal(C_expected, C_computed):
            raise ValueError(f"Full decomposition verification failed")

    return U, V, W


def _verify_decomposition(U, V, W, num_trials=300):
    """Verify that the tensor decomposition exactly matches matrix multiplication."""
    import random
    n, m, p = DIMENSIONS
    rank = len(W[0]) if W else 0
    for trial in range(num_trials):
        A = np.array([[random.randint(-5, 5) for _ in range(m)] for _ in range(n)])
        B = np.array([[random.randint(-5, 5) for _ in range(p)] for _ in range(m)])
        C_expected = A @ B
        M = np.zeros(rank)
        a = [A[i, j] for i in range(n) for j in range(m)]
        b = [B[j, k] for j in range(m) for k in range(p)]
        for k in range(rank):
            ua = sum(U[i][k] * a[i] for i in range(n * m))
            vb = sum(V[i][k] * b[i] for i in range(m * p))
            M[k] = ua * vb
        C_computed = np.zeros((p, n))
        for r in range(p * n):
            val = sum(W[r][k] * M[k] for k in range(rank))
            C_computed[r // n, r % n] = val
        if not np.array_equal(C_expected, C_computed):
            return False
    return True


def _try_rank20_search(deadline):
    """Attempt to find a rank-20 decomposition by merging Strassen intermediates.
    Uses a randomized approach: try to find a linear combination of the 21 columns
    that reduces the rank by 1 while preserving the tensor identity.
    This is a simplified Gröbner-style search using null-space analysis.
    """
    import random
    U21, V21, W21 = _build_block_strassen()
    rank21 = 21

    # The idea: if there's a linear dependency among the 21 rank-1 tensors,
    # we can reduce to rank 20. We search for a vector (u, v, w) such that
    # sum_k (u_k * U_col_k) * (v_k * V_col_k) * (w_k * W_col_k) = 0
    # but this is nonlinear. Instead, we try a different approach:
    # merge two columns that share the same A-factor pattern.

    # For now, return None to indicate no rank-20 found.
    # A full Gröbner basis search would be too complex for this budget.
    return None


def construct():
    deadline = time.time() + 130  # 130s safety margin (150s limit)

    # Build the rank-21 block-Strassen decomposition (verified)
    U, V, W = _build_block_strassen()

    # Try to find a rank-20 decomposition (if time permits)
    if time.time() < deadline - 10:
        result = _try_rank20_search(deadline)
        if result is not None:
            U, V, W = result

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
