import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def _strassen_decomp():
    """Rank-42 Strassen decomposition for 2x2 * 2x12 matrix multiplication."""
    n, m, p = 2, 2, 12
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42

    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    for t in range(num_blocks):
        base = t * 7
        k0 = 2 * t
        k1 = 2 * t + 1

        # M1 = (A00+A11)*(B00+B11)
        c = base + 0
        U[0, c] = 1.0
        U[3, c] = 1.0
        V[0 * p + k0, c] = 1.0
        V[1 * p + k1, c] = 1.0
        W[k0 * n + 0, c] = 1.0
        W[k1 * n + 1, c] = 1.0

        # M2 = (A10+A11)*B00
        c = base + 1
        U[2, c] = 1.0
        U[3, c] = 1.0
        V[0 * p + k0, c] = 1.0
        W[k0 * n + 1, c] = 1.0
        W[k1 * n + 1, c] = -1.0

        # M3 = A00*(B01-B11)
        c = base + 2
        U[0, c] = 1.0
        V[0 * p + k1, c] = 1.0
        V[1 * p + k1, c] = -1.0
        W[k1 * n + 0, c] = 1.0
        W[k1 * n + 1, c] = 1.0

        # M4 = A11*(B10-B00)
        c = base + 3
        U[3, c] = 1.0
        V[1 * p + k0, c] = 1.0
        V[0 * p + k0, c] = -1.0
        W[k0 * n + 0, c] = 1.0
        W[k0 * n + 1, c] = 1.0

        # M5 = (A00+A01)*B11
        c = base + 4
        U[0, c] = 1.0
        U[1, c] = 1.0
        V[1 * p + k1, c] = 1.0
        W[k0 * n + 0, c] = -1.0
        W[k1 * n + 0, c] = 1.0

        # M6 = (A10-A00)*(B00+B01)
        c = base + 5
        U[2, c] = 1.0
        U[0, c] = -1.0
        V[0 * p + k0, c] = 1.0
        V[0 * p + k1, c] = 1.0
        W[k1 * n + 1, c] = 1.0

        # M7 = (A01-A11)*(B10+B11)
        c = base + 6
        U[1, c] = 1.0
        U[3, c] = -1.0
        V[1 * p + k0, c] = 1.0
        V[1 * p + k1, c] = 1.0
        W[k0 * n + 0, c] = 1.0

    return U, V, W


def _verify_decomp(U, V, W, tol=1e-9):
    """Verify that sum_r U[:,r] outer V[:,r] outer W[:,r] equals the matrix mult tensor."""
    n, m, p = 2, 2, 12
    rank = U.shape[1]
    
    # Full verification for safety
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # Target: C_{i, k} = sum_j A_{i, j} B_{j, k}
    # a = i*n + j  (A row-major)
    # b = j*p + k  (B row-major)
    # c = k*n + i  (C column-major? Code uses c = k*n + i in construction? 
    # In _strassen_decomp: W[k0 * n + 0] -> c = k*n + row. So C is col-major 2x12.
    # So c = k*n + i.
    
    for a in range(n * m):
        i_a = a // n
        j_a = a % n
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            # Target contribution to C[i_a, k_b] from A[i_a, j_a] B[j_b, k_b]
            # Non-zero only if j_a == j_b
            if j_a == j_b:
                # Target C index: c = k_b * n + i_a
                c_target = k_b * n + i_a
                target_val = 1.0
            else:
                # For this a,b, all c should be 0? No, we sum over r.
                # It's easier to compute the full tensor for a few random c or all c.
                # Given small size, we can check all c for this a,b.
                pass
                
    # More efficient: Iterate over all a, b, c? 4*24*24 = 2304 points. Rank 42. 2304*42 = 96k ops.
    for a in range(n * m):
        i_a = a // n
        j_a = a % n
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                # c = k_c * n + i_c
                k_c = c // n
                i_c = c % n
                
                # Target: 1 if i_a == i_c and k_b == k_c and j_a == j_b
                target = 1.0 if (i_a == i_c and k_b == k_c and j_a == j_b) else 0.0
                
                comp = 0.0
                for r in range(rank):
                    comp += U[a, r] * V[b, r] * W[c, r]
                
                if abs(comp - target) > tol:
                    return False
    return True


def construct():
    """
    Rank-42 Strassen decomposition for 2x2 * 2x12 matrix multiplication.
    
    The 2x12 matrix B is split into six 2x2 column-pairs. Each pair is
    multiplied by the 2x2 matrix A using Strassen's 7-product algorithm.
    Total rank = 6 * 7 = 42.
    
    This is the Kronecker product decomposition: T(2,2,12) = T(2,2,2) ⊗ T(1,1,6)
    with rank = 7 * 6 = 42.
    """
    U, V, W = _strassen_decomp()
    
    # Verify correctness
    assert _verify_decomp(U, V, W), "Decomposition verification failed"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
