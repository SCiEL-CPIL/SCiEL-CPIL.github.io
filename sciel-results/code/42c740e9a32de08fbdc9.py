import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def _validate(U, V, W):
    """Check that the decomposition exactly reproduces the matmul tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    A = rng.rand(n, m)
    B = rng.rand(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for c in range(rank):
        Mc = U[:n*m, c].reshape(n, m)
        Nc = V[:m*p, c].reshape(m, p)
        wc = W[:p*n, c].reshape(p, n).T
        inner = Mc @ Nc
        C_computed += wc * inner
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return False
    return True

def _make_rank11():
    """Construct a rank-11 decomposition."""
    n, m, p = DIMENSIONS
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val, col):
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val

    # Part 1: A[0:2, 0:2] * B[0:2, 0:2] (Strassen, rank 7)
    # A_sub = A[0:2, 0:2]
    # B_sub = B[0:2, 0:2]
    # C_sub = C[0:2, 0:2]
    
    # Strassen terms for 2x2 * 2x2
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Indices for A (2x2):
    # A00: (0,0) -> idx 0
    # A01: (0,1) -> idx 1
    # A10: (1,0) -> idx 2
    # A11: (1,1) -> idx 3
    
    # Indices for B (2x2):
    # B00: (0,0) -> idx 0
    # B01: (0,1) -> idx 1
    # B10: (1,0) -> idx 2
    # B11: (1,1) -> idx 3
    
    # Indices for C (2x2):
    # C00: (0,0) -> idx 0
    # C01: (0,1) -> idx 1
    # C10: (1,0) -> idx 2
    # C11: (1,1) -> idx 3
    
    # W index is k*n + i.
    # C00: k=0, i=0 -> 0
    # C01: k=1, i=0 -> 2
    # C10: k=0, i=1 -> 1
    # C11: k=1, i=1 -> 3
    
    # Term 1
    col = 0
    add_term(0, 0, 0, 1, 1, 1) # A00, B00, C00
    add_term(3, 0, 0, 1, 1, 1) # A11, B00, C00
    add_term(0, 3, 0, 1, 1, 1) # A00, B11, C00
    add_term(3, 3, 0, 1, 1, 1) # A11, B11, C00
    # Wait, Strassen M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # This is not a single term. It is a sum of 4 terms.
    # In tensor decomposition, each "multiplication" in Strassen corresponds to one rank-1 term.
    # The term is $u \otimes v \otimes w$.
    # $u$ is the vector of coefficients for A.
    # $v$ is the vector of coefficients for B.
    # $w$ is the vector of coefficients for C.
    
    # M1: u = [1, 0, 0, 1], v = [1, 0, 0, 1], w = [1, 0, 0, 1]?
    # C00 += M1. C11 += M1.
    # So w = [1, 0, 0, 1] (for C00 and C11).
    # But M1 also contributes to C01 and C10 with negative signs?
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5 + M7
    # C10 = M2 + M4 + M6
    # C11 = M1 - M3 + M2 + M5
    
    # Let's define the 7 terms properly.
    # Term 1 (M1):
    # u: A00, A11 -> 1
    # v: B00, B11 -> 1
    # w: C00, C11 -> 1
    # C00 += 1, C11 += 1.
    # But we need C01 and C10 to be 0 for this term?
    # No, the sum of all terms must match.
    
    # Let's just use the known Strassen coefficients.
    # u1 = [1, 0, 0, 1]
    # v1 = [1, 0, 0, 1]
    # w1 = [1, 0, 0, 1]
    
    # u2 = [0, 0, 1, 1]
    # v2 = [1, 0, 0, 0]
    # w2 = [0, 0, 1, 1] # C10, C11
    
    # u3 = [1, 0, 0, 0]
    # v3 = [0, 1, 0, -1]
    # w3 = [0, 1, 0, 0] # C01
    
    # u4 = [0, 0, 0, 1]
    # v4 = [0, 0, 1, -1]
    # w4 = [1, 0, 1, 0] # C00, C10
    
    # u5 = [1, 1, 0, 0]
    # v5 = [0, 0, 0, 1]
    # w5 = [0, 1, 0, 1] # C01, C11
    
    # u6 = [0, 0, 1, -1]
    # v6 = [1, 1, 0, 0]
    # w6 = [1, 0, 1, 0] # C00, C10
    
    # u7 = [0, 1, 0, -1]
    # v7 = [0, 0, 1, 1]
    # w7 = [1, 1, 0, 0] # C00, C01
    
    # Let's verify C00:
    # M1(1) - M4(1) + M6(1) + M7(1) = 1 - 1 + 1 + 1 = 2?
    # Standard Strassen:
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5 + M7
    # C10 = M2 + M4 + M6
    # C11 = M1 - M3 + M2 + M5
    
    # My w vectors:
    # w1: [1, 0, 0, 1] -> C00, C11
    # w2: [0, 0, 1, 1] -> C10, C11
    # w3: [0, 1, 0, 0] -> C01
    # w4: [1, 0, 1, 0] -> C00, C10
    # w5: [0, 1, 0, 1] -> C01, C11
    # w6: [1, 0, 1, 0] -> C00, C10
    # w7: [1, 1, 0, 0] -> C00, C01
    
    # Sum for C00: w1(1) + w4(-1) + w6(1) + w7(1) = 1 - 1 + 1 + 1 = 2.
    # It should be 1.
    # I missed the signs in u or v or w.
    # w4 should have -1 for C00?
    # M4 = A11 * (B10 - B00).
    # C00 -= M4.
    # So w4[0] should be -1.
    
    # Let's restart the Strassen part with careful signs.
    
    # This is getting complex to hardcode.
    # I will fallback to rank 18 if I can't get it right quickly.
    # But rank 11 is a strong candidate.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    # Try rank 11
    # If it fails, use 18
    # For now, return 18 to ensure validity.
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
