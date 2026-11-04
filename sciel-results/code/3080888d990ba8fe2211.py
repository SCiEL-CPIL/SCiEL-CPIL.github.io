import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin from 150s limit

    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 84

    # Build the target tensor T[a, b, c]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Trivial decomposition
    U_trivial = np.zeros((n*m, trivial_rank), dtype=np.float64)
    V_trivial = np.zeros((m*p, trivial_rank), dtype=np.float64)
    W_trivial = np.zeros((p*n, trivial_rank), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, col] = 1.0
                V_trivial[b, col] = 1.0
                W_trivial[c, col] = 1.0
                col += 1

    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = trivial_rank

    def verify_decomposition(U, V, W, rank):
        """Verify that sum_r U[:,r] outer V[:,r] outer W[:,r] == T"""
        recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(rank):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(recon, T, atol=1e-9)

    def round_to_half_int(matrix):
        """Round entries to nearest half-integer"""
        return np.round(matrix * 2) / 2

    def try_rank_decomposition(target_rank, time_limit=2.0, seed=42):
        """
        Attempt to find a decomposition of the given rank using gradient descent.
        Returns (U, V, W) if successful, else None.
        """
        if time.time() > deadline - 5:
            return None

        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1

        lr = 0.05
        decay = 0.999

        start_t = time.time()
        for step in range(500):
            if time.time() > start_t + time_limit:
                break

            # Compute reconstruction
            recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
            for r in range(target_rank):
                recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

            err = T - recon
            err_norm = np.linalg.norm(err)

            if err_norm < 1e-8:
                # Try to round and verify
                U_r = round_to_half_int(U)
                V_r = round_to_half_int(V)
                W_r = round_to_half_int(W)
                if verify_decomposition(U_r, V_r, W_r, target_rank):
                    return U_r, V_r, W_r
                # If rounding fails, keep going but maybe adjust
                # Or return float if allowed? No, must be half-int.
                # So we need to find a solution that is close to half-integers.
                # We can add a penalty term to encourage half-integer values.
                pass

            # Compute gradients
            grad_U = np.einsum('abc,br,cr->ar', err, V, W)
            grad_V = np.einsum('abc,ar,cr->br', err, U, W)
            grad_W = np.einsum('abc,ar,br->cr', err, U, V)

            # Penalty to encourage half-integers
            # Penalty = sum (2*x - round(2*x))^2
            # Grad penalty = 4 * (2*x - round(2*x))
            # This is non-smooth, so we approximate or just rely on the fact that
            # if the solution is exact, it might naturally land on half-integers
            # for these small dimensions.
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            lr *= decay

        # Final check with rounding
        U_r = round_to_half_int(U)
        V_r = round_to_half_int(V)
        W_r = round_to_half_int(W)
        if verify_decomposition(U_r, V_r, W_r, target_rank):
            return U_r, V_r, W_r
        
        return None

    # Try to find a lower rank
    # Start from lower ranks to maximize score
    candidate_ranks = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
            
        # Try a few seeds
        found = False
        for seed in [42, 123, 999]:
            if time.time() > deadline - 10:
                break
            result = try_rank_decomposition(r, time_limit=1.5, seed=seed)
            if result is not None:
                U_new, V_new, W_new = result
                if verify_decomposition(U_new, V_new, W_new, r):
                    best_U = U_new
                    best_V = V_new
                    best_W = W_new
                    best_rank = r
                    found = True
                    break
        
        if found:
            # If we found a rank, we can stop or try to go lower?
            # The list is sorted ascending, so we are already trying lower ranks first.
            # If we find one, it's the lowest in our candidate list that worked.
            # We could continue to check if even lower ones work, but we started at 30.
            # So if 30 worked, we are done.
            # If 40 worked, we should have checked 30, 35 first.
            # So we can break if we find one? No, we want the MINIMAL rank.
            # Since we iterate 30, 35, 40..., if 30 fails and 35 succeeds, 35 is the best found so far.
            # We should continue to see if 30 was a fluke and maybe 32 works?
            # But we only test discrete ranks.
            # Let's just break if we find a valid decomposition for the current rank, 
            # because we are iterating from low to high.
            # Wait, if 30 fails, 35 fails, 40 succeeds. We return 40.
            # This is the best in our set.
            break

    # If we didn't find any lower rank, we return the trivial one
    U_out = best_U[:, :best_rank].tolist()
    V_out = best_V[:, :best_rank].tolist()
    W_out = best_W[:, :best_rank].tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
