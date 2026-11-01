import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Schoolbook fallback (rank 28)
    def schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m + k][col] = 1
                    V[k*p + j][col] = 1
                    W[j*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    fallback = schoolbook()

    def verify(U_mat, V_mat, W_mat):
        """Check if sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c] == 1 for all i,k,j"""
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-8)

    def is_half_integer(mat):
        return np.allclose(mat * 2, np.round(mat * 2), atol=1e-8)

    def to_half_int_list(mat):
        doubled = np.round(mat * 2)
        return doubled.astype(int).tolist()

    # Laderman block for 2x2x2 (rank 7)
    def laderman_block():
        U_b = np.zeros((4, 7))
        V_b = np.zeros((4, 7))
        W_b = np.zeros((4, 7))
        U_b[0,0]=1; U_b[1,0]=-1; U_b[2,0]=-1; U_b[3,0]=1
        V_b[0,0]=1; V_b[1,0]=-1; V_b[2,0]=-1; V_b[3,0]=1
        W_b[0,0]=1; W_b[1,0]=1; W_b[2,0]=1; W_b[3,0]=1
        U_b[0,1]=1; U_b[1,1]=-1
        V_b[1,1]=1; V_b[3,1]=-1
        W_b[1,1]=1; W_b[3,1]=1
        U_b[0,2]=1; U_b[2,2]=-1
        V_b[2,2]=1; V_b[0,2]=-1
        W_b[2,2]=1; W_b[3,2]=1
        U_b[0,3]=1; V_b[0,3]=1; W_b[0,3]=1
        U_b[1,4]=1; V_b[1,4]=1; W_b[1,4]=1
        U_b[2,5]=1; V_b[2,5]=1; W_b[2,5]=1
        U_b[3,6]=1; V_b[3,6]=1; W_b[3,6]=1
        return U_b, V_b, W_b

    def schoolbook_2x2x1():
        U_b = np.zeros((4, 2))
        V_b = np.zeros((2, 2))
        W_b = np.zeros((2, 2))
        U_b[0,0]=1; V_b[0,0]=1; W_b[0,0]=1
        U_b[1,1]=1; V_b[1,1]=1; W_b[1,1]=1
        return U_b, V_b, W_b

    def build_laderman_decomposition():
        """Rank-23 block decomposition (3x Laderman + 1x schoolbook)."""
        rank = 23
        U_full = np.zeros((n*m, rank))
        V_full = np.zeros((m*p, rank))
        W_full = np.zeros((p*n, rank))
        U_l, V_l, W_l = laderman_block()
        U_s, V_s, W_s = schoolbook_2x2x1()
        for block_idx in range(3):
            col_offset = block_idx * 7
            j_start = block_idx * 2
            for c_local in range(7):
                c_global = col_offset + c_local
                for idx in range(4):
                    U_full[idx, c_global] = U_l[idx, c_local]
                for k in range(2):
                    for j_local in range(2):
                        j_global = j_start + j_local
                        V_full[k*p + j_global, c_global] = V_l[k*2 + j_local, c_local]
                for j_local in range(2):
                    for i in range(2):
                        j_global = j_start + j_local
                        W_full[j_global*n + i, c_global] = W_l[j_local*2 + i, c_local]
        col_offset = 21
        for c_local in range(2):
            c_global = col_offset + c_local
            for idx in range(4):
                U_full[idx, c_global] = U_s[idx, c_local]
            for k in range(2):
                V_full[k*p + 6, c_global] = V_s[k, c_local]
            for i in range(2):
                W_full[6*n + i, c_global] = W_s[i, c_local]
        return U_full, V_full, W_full

    # Build the known rank-23 solution as starting point
    U0, V0, W0 = build_laderman_decomposition()
    best_rank = 23
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    # Verify the starting point
    if not verify(U0, V0, W0):
        return fallback

    def compute_error(U_mat, V_mat, W_mat):
        """Compute the L-infinity error of the tensor reconstruction."""
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.max(np.abs(T - 1.0))

    def try_rank_reduction(U_mat, V_mat, W_mat, target_rank, rng, max_iters=2000):
        """
        Attempt to reduce the rank from current to target_rank by:
        1. Selecting target_rank columns from the current factorization
        2. Refining via random half-integer perturbations
        Returns (U, V, W) if success, else None.
        """
        current_rank = U_mat.shape[1]
        if target_rank >= current_rank:
            return None

        # Try to find a subset of columns that spans the tensor
        # Strategy: use SVD-like column selection on the flattened tensor
        # The tensor T[i,k,j] = sum_c U[i,k,c] V[k,j,c] W[j,i,c]
        # We need target_rank columns whose combined contribution equals T.

        # Approach: random column subsets + refinement
        n_tries = 50
        for attempt in range(n_tries):
            if time.time() > deadline:
                return None

            # Select random subset of columns
            cols = rng.choice(current_rank, size=target_rank, replace=False)
            U_sub = U_mat[:, cols].copy()
            V_sub = V_mat[:, cols].copy()
            W_sub = W_mat[:, cols].copy()

            # Check if this subset already works (unlikely for random)
            err = compute_error(U_sub, V_sub, W_sub)
            if err < 1e-8:
                return U_sub, V_sub, W_sub

            # Refine: random half-integer mutations
            best_err = err
            best_U_sub = U_sub.copy()
            best_V_sub = V_sub.copy()
            best_W_sub = W_sub.copy()

            for it in range(max_iters):
                if time.time() > deadline:
                    break

                # Pick a random factor and a random entry to perturb
                which = rng.randint(3)
                if which == 0:
                    row = rng.randint(U_sub.shape[0])
                    col = rng.randint(U_sub.shape[1])
                    old = U_sub[row, col]
                    U_sub[row, col] = old + rng.choice([-1, -0.5, 0.5, 1])
                    err = compute_error(U_sub, V_sub, W_sub)
                    if err < best_err:
                        best_err = err
                        best_U_sub = U_sub.copy()
                        best_V_sub = V_sub.copy()
                        best_W_sub = W_sub.copy()
                        if best_err < 1e-8:
                            U_sub = best_U_sub.copy()
                            V_sub = best_V_sub.copy()
                            W_sub = best_W_sub.copy()
                            return U_sub, V_sub, W_sub
                    else:
                        U_sub[row, col] = old
                elif which == 1:
                    row = rng.randint(V_sub.shape[0])
                    col = rng.randint(V_sub.shape[1])
                    old = V_sub[row, col]
                    V_sub[row, col] = old + rng.choice([-1, -0.5, 0.5, 1])
                    err = compute_error(U_sub, V_sub, W_sub)
                    if err < best_err:
                        best_err = err
                        best_U_sub = U_sub.copy()
                        best_V_sub = V_sub.copy()
                        best_W_sub = W_sub.copy()
                        if best_err < 1e-8:
                            U_sub = best_U_sub.copy()
                            V_sub = best_V_sub.copy()
                            W_sub = best_W_sub.copy()
                            return U_sub, V_sub, W_sub
                    else:
                        V_sub[row, col] = old
                else:
                    row = rng.randint(W_sub.shape[0])
                    col = rng.randint(W_sub.shape[1])
                    old = W_sub[row, col]
                    W_sub[row, col] = old + rng.choice([-1, -0.5, 0.5, 1])
                    err = compute_error(U_sub, V_sub, W_sub)
                    if err < best_err:
                        best_err = err
                        best_U_sub = U_sub.copy()
                        best_V_sub = V_sub.copy()
                        best_W_sub = W_sub.copy()
                        if best_err < 1e-8:
                            U_sub = best_U_sub.copy()
                            V_sub = best_V_sub.copy()
                            W_sub = best_W_sub.copy()
                            return U_sub, V_sub, W_sub
                    else:
                        W_sub[row, col] = old

            # Accept if improved
            if best_err < err:
                U_sub = best_U_sub
                V_sub = best_V_sub
                W_sub = best_W_sub

        return None

    # Try to reduce rank from 23 downward
    rng = np.random.default_rng(42)
    for target_rank in range(22, 14, -1):
        if time.time() > deadline:
            break
        result = try_rank_reduction(best_U, best_V, best_W, target_rank, rng, max_iters=500)
        if result is not None:
            U_t, V_t, W_t = result
            if verify(U_t, V_t, W_t) and is_half_integer(U_t) and is_half_integer(V_t) and is_half_integer(W_t):
                best_rank = target_rank
                best_U, best_V, best_W = U_t, V_t, W_t
                print(f"Reduced to rank {target_rank}")
                # Try to go even lower
                continue
        # If we can't reduce, break (diminishing returns)
        if target_rank <= 20:
            break

    # Also try a completely different approach: random initialization from scratch
    # for ranks below our current best
    if time.time() < deadline - 20:
        for target_rank in range(min(best_rank - 1, 22), 14, -1):
            if time.time() > deadline:
                break

            # Random initialization with half-integer entries
            n_tries = 30
            success = False
            for attempt in range(n_tries):
                if time.time() > deadline:
                    break

                U_rand = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, target_rank))
                V_rand = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, target_rank))
                W_rand = rng.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, target_rank))

                err = compute_error(U_rand, V_rand, W_rand)
                if err < 1e-8:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                    success = True
                    break

                # Local refinement
                best_err = err
                best_U_r = U_rand.copy()
                best_V_r = V_rand.copy()
                best_W_r = W_rand.copy()

                for it in range(300):
                    if time.time() > deadline:
                        break
                    which = rng.randint(3)
                    if which == 0:
                        row = rng.randint(U_rand.shape[0])
                        col = rng.randint(U_rand.shape[1])
                        old = U_rand[row, col]
                        U_rand[row, col] = old + rng.choice([-1, -0.5, 0.5, 1])
                        err = compute_error(U_rand, V_rand, W_rand)
                        if err < best_err:
                            best_err = err
                            best_U_r = U_rand.copy()
                            best_V_r = V_rand.copy()
                            best_W_r = W_rand.copy()
                            if best_err < 1e-8:
                                best_rank = target_rank
                                best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                                success = True
                                break
                        else:
                            U_rand[row, col] = old
                    elif which == 1:
                        row = rng.randint(V_rand.shape[0])
                        col = rng.randint(V_rand.shape[1])
                        old = V_rand[row, col]
                        V_rand[row, col] = old + rng.choice([-1, -0.5, 0.5, 1])
                        err = compute_error(U_rand, V_rand, W_rand)
                        if err < best_err:
                            best_err = err
                            best_U_r = U_rand.copy()
                            best_V_r = V_rand.copy()
                            best_W_r = W_rand.copy()
                            if best_err < 1e-8:
                                best_rank = target_rank
                                best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                                success = True
                                break
                        else:
                            V_rand[row, col] = old
                    else:
                        row = rng.randint(W_rand.shape[0])
                        col = rng.randint(W_rand.shape[1])
                        old = W_rand[row, col]
                        W_rand[row, col] = old + rng.choice([-1, -0.5, 0.5, 1])
                        err = compute_error(U_rand, V_rand, W_rand)
                        if err < best_err:
                            best_err = err
                            best_U_r = U_rand.copy()
                            best_V_r = V_rand.copy()
                            best_W_r = W_rand.copy()
                            if best_err < 1e-8:
                                best_rank = target_rank
                                best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                                success = True
                                break
                        else:
                            W_rand[row, col] = old

                    # Accept improvement
                    if best_err < err:
                        U_rand = best_U_r
                        V_rand = best_V_r
                        W_rand = best_W_r
                        err = best_err

                if success:
                    break

            if success:
                print(f"Random search found rank {target_rank}")

    # Final verification and output
    if verify(best_U, best_V, best_W) and is_half_integer(best_U) and is_half_integer(best_V) and is_half_integer(best_W):
        return {
            "U": to_half_int_list(best_U),
            "V": to_half_int_list(best_V),
            "W": to_half_int_list(best_W)
        }

    return fallback

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
