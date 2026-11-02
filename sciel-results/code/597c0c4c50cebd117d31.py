import time
import random
import numpy as np

DIMENSIONS = [2, 2, 3]
TIME_LIMIT = 140.0  # seconds, with 10s safety margin

# EVOLVE-BLOCK-START

def _build_rank11():
    """Build the known rank-11 decomposition."""
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val):
        U[i * m + j][r] = val
    def set_v(j, k, r, val):
        V[j * p + k][r] = val
    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 1: (A00 + A11) * (B00 + B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    # Term 2: (A10 + A11) * B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    # Term 3: A00 * (B01 - B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    # Term 4: A11 * (B10 - B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    # Term 5: (A00 + A01) * B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

    # Term 6: (A10 - A00) * (B00 + B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 7: (A01 - A11) * (B10 + B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 8: A00 * B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 9: A01 * B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 10: A10 * B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 11: A11 * B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}


def _verify_decomposition_fast(U, V, W, n=2, m=2, p=3, trials=10):
    """Verify that the decomposition correctly computes C = A*B for random inputs using NumPy."""
    rank = len(U[0])
    
    # Reshape U, V, W to 3D arrays
    # U: (n, m, rank)
    U_3d = np.array(U).reshape(n, m, rank)
    # V: (m, p, rank)
    V_3d = np.array(V).reshape(m, p, rank)
    # W: (p, n, rank) -> we need (n, p, rank) for element-wise multiply with M
    # W index is k*n+i. So W_3d_raw is (p, n, rank).
    W_3d_raw = np.array(W).reshape(p, n, rank)
    W_3d = np.transpose(W_3d_raw, (1, 0, 2)) # Now (n, p, rank)

    for _ in range(trials):
        A = np.random.randint(-10, 11, (n, m))
        B = np.random.randint(-10, 11, (m, p))
        C_expected = A @ B
        
        # Compute M_r = U_r * V_r for each r
        # U_3d: (n, m, rank)
        # V_3d: (m, p, rank)
        # We want M[i, k, r] = sum_j U[i, j, r] * V[j, k, r]
        # This is a batched matmul.
        # np.einsum('ijr,jkr->ikr', U_3d, V_3d)
        M = np.einsum('ijr,jkr->ikr', U_3d, V_3d)
        
        # C[i, k] = sum_r W[i, k, r] * M[i, k, r]
        C_computed = np.einsum('ikr,ikr->ik', W_3d, M)
        
        if not np.array_equal(C_expected, C_computed):
            return False
    return True


def _check_rank10_over_gf(mod, max_attempts=50000, deadline=None):
    """
    Search for a rank-10 decomposition over a finite field.
    """
    n, m, p = DIMENSIONS
    rank = 10
    
    rng = random.Random(12345)
    attempts = 0
    
    while (deadline is None or time.time() < deadline) and attempts < max_attempts:
        attempts += 1
        
        # Generate random U and V over the field
        U = [[rng.randint(0, mod-1) for _ in range(rank)] for _ in range(n*m)]
        V = [[rng.randint(0, mod-1) for _ in range(rank)] for _ in range(m*p)]
        
        # Compute the "partial product" tensor: C[i,k,r] = sum_j U[i,j,r] * V[j,k,r]
        C = [[[0]*rank for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    total = 0
                    for j in range(m):
                        total = (total + U[i*m+j][r] * V[j*p+k][r]) % mod
                    C[i][k][r] = total
        
        # For each (i,k), we need to find r such that C[i,k,r] != 0
        # and set W[k*n+i, r] = inv(C[i,k,r])
        W = [[0]*rank for _ in range(p*n)]
        valid = True
        
        for i in range(n):
            for k in range(p):
                non_zero_rs = [r for r in range(rank) if C[i][k][r] != 0]
                if not non_zero_rs:
                    valid = False
                    break
                # Pick one non-zero r
                r = rng.choice(non_zero_rs)
                W[k*n + i][r] = pow(C[i][k][r], mod-2, mod)
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify: for all (i,k), sum_r W[k*n+i, r] * C[i,k,r] should equal 1
        for i in range(n):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total = (total + W[k*n+i][r] * C[i][k][r]) % mod
                if total != 1:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Convert to integers (they're already in [0, mod-1])
            return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Get the known rank-11 solution
    sol11 = _build_rank11()
    
    # Verify it works
    if not _verify_decomposition_fast(sol11["U"], sol11["V"], sol11["W"], trials=5):
        # If verification fails (shouldn't happen), return as-is
        return sol11
    
    # Try to find rank-10 over small finite fields
    # Limit search time to 20 seconds to ensure we return quickly if not found
    search_deadline = start_time + 20.0
    if time.time() < search_deadline:
        for mod in [7, 11, 13]:
            if time.time() > search_deadline - 2:
                break
            sol10 = _check_rank10_over_gf(mod, max_attempts=10000, deadline=search_deadline - 2)
            if sol10 is not None:
                # Verify over integers
                if _verify_decomposition_fast(sol10["U"], sol10["V"], sol10["W"], trials=5):
                    return sol10
    
    return sol11

# EVOLVE-BLOCK-END
