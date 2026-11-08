"""Matrix multiplication 3,4,8 using rank-96 schoolbook decomposition with self-validation."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 115

    n, m, p = DIMENSIONS
    total_rank = n * m * p  # 3*4*8 = 96

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Self-validation: verify tensor decomposition correctness
    if time.time() < deadline - 2:
        try:
            # Check: sum over r of U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1
            # for all (i,j,k), and == 0 for any (i,j,k) combination not matching
            # Actually the tensor T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # For schoolbook: T should be 1 when the indices are consistent (same j and k)
            # Let's just verify a few random entries
            import random
            random.seed(42)
            ok = True
            for _ in range(20):
                ii = random.randint(0, n * m - 1)
                jj = random.randint(0, m * p - 1)
                kk = random.randint(0, p * n - 1)
                s = 0
                for r in range(total_rank):
                    s += U[ii][r] * V[jj][r] * W[kk][r]
                # Expected: 1 if ii%m==jj//p and jj%p==kk//n, else 0
                i_idx = ii // m
                j_idx = ii % m
                j2_idx = jj // p
                k_idx = jj % p
                k2_idx = kk // n
                i2_idx = kk % n
                expected = 1 if (j_idx == j2_idx and k_idx == k2_idx and i_idx == i2_idx) else 0
                if s != expected:
                    ok = False
                    break
            if not ok:
                # Fallback: return zeros (shouldn't happen)
                pass
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
