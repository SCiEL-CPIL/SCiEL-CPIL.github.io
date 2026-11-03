"""
Tensor decomposition for 3x4 by 4x4 matrix multiplication.
Schoolbook baseline (rank 48) is trivially valid.
We attempt a randomised search for a lower-rank exact integer/half-integer decomposition.
"""
import time
import numpy as np
from fractions import Fraction
import itertools

DIMENSIONS = [3, 4, 4]
n, m, p = DIMENSIONS

# Tensor T[i,j,k] = 1
# We want U (n*m x r), V (m*p x r), W (p*n x r) such that
# U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r equals 1.

def construct():
    t0 = time.time()
    deadline = t0 + 120.0  # safety margin from 150s limit

    # Start with schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=int)
    V_sb = np.zeros((m * p, rank_sb), dtype=int)
    W_sb = np.zeros((p * n, rank_sb), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_sb[i * m + j, c] = 1
                V_sb[j * p + k, c] = 1
                W_sb[k * n + i, c] = 1

    # Build tensor T as a 3D array
    T = np.ones((n, m, p), dtype=int)

    # Verify schoolbook
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        Uf = U.astype(np.float64)
        Vf = V.astype(np.float64)
        Wf = W.astype(np.float64)
        # Compute reconstruction
        recon = np.zeros((n, m, p), dtype=np.float64)
        for r_idx in range(r):
            # U[:, r_idx] is (n*m,), V[:, r_idx] is (m*p,), W[:, r_idx] is (p*n,)
            # Need to reshape U to (n, m), V to (m, p), W to (p, n)
            Ur = Uf[:, r_idx].reshape(n, m)
            Vr = Vf[:, r_idx].reshape(m, p)
            Wr = Wf[:, r_idx].reshape(p, n)
            # Sum over r: Ur[i,j] * Vr[j,k] * Wr[k,i]
            # This is a tensor contraction
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        recon[i, j, k] += Ur[i, j] * Vr[j, k] * Wr[k, i]
        # Check if recon equals T
        return np.allclose(recon, T, atol=tol)

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Try to find a lower rank decomposition
    # We will search for rank r from min_rank to best_rank - 1
    # A lower bound is max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # But tensor rank is usually much higher.
    # Known lower bound for 3x4x4 is not trivial.
    # Let's try random integer matrices for small ranks.

    # We can also try a greedy or ALS approach, but exactness is hard.
    # Given time constraints, random search for small integer values is feasible.

    # Let's try ranks from 20 up to 47
    for r in range(16, best_rank):
        if time.time() > deadline - 5:
            break

        # Number of random trials per rank
        # More trials for smaller r
        trials = 100 if r < 30 else 50

        for trial in range(trials):
            if time.time() > deadline - 2:
                break

            # Random integer matrices with entries in {-2, -1, 0, 1, 2}
            # Or {-1, 0, 1}
            scale = 2
            U_rand = np.random.randint(-scale, scale + 1, (n * m, r))
            V_rand = np.random.randint(-scale, scale + 1, (m * p, r))
            W_rand = np.random.randint(-scale, scale + 1, (p * n, r))

            # Verify
            if verify(U_rand, V_rand, W_rand):
                best_rank = r
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                break  # Found a decomposition for this rank, move to lower

            # Also try with some entries scaled up to catch cases where 2*2*2=8 etc
            # But verify uses float64 so exactness is checked.

        # If we found a lower rank, we can break early if we want, but we can keep searching for even lower
        # Actually, the loop goes from 16 upwards, so if we find r=30, we still check 16..29 in previous iterations?
        # No, the loop is `for r in range(16, best_rank)`.
        # If best_rank was 48, and we find r=30, we break out of the trial loop, but the outer loop continues?
        # No, `break` only breaks the trial loop. The outer loop increments r.
        # Wait, if we find r=30, best_rank becomes 30.
        # The outer loop condition `range(16, best_rank)` is evaluated once at the start of the loop?
        # No, `range` is evaluated once. So if I set `for r in range(16, 48)`, it will check all.
        # If I find r=30, I should probably stop checking higher r? No, I'm checking from low to high.
        # So if I find r=30, I have a valid 30. I still want to check if 29 works.
        # But the loop is `for r in range(16, best_rank)`.
        # If I update best_rank to 30, the range is fixed at 16..47.
        # I should just break out of everything if I find the theoretical minimum?
        # Or just continue searching for lower ranks.

    # Convert best found to lists for output
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
