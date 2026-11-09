<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Baseline rank-64
    U0, V0, W0 = _build_rank64()
    if not _verify_decomposition(U0, V0, W0, n, m, p):
        raise RuntimeError("Baseline rank-64 failed verification")

    best_rank = 64
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    # Try to reduce rank via ALS from various initializations
    # The all-ones tensor rank is exactly 64 (it's the outer product of three
    # one-hot-like structures), so we expect rank 64 to be minimal, but we
    # attempt reductions to be safe.
    np.random.seed(42)
    for target_rank in range(63, 40, -1):
        if time.time() > deadline - 20:
            break
        # Multiple random initializations
        for trial in range(3):
            if time.time() > deadline - 20:
                break
            # Initialize by mixing baseline columns
            perm = np.random.permutation(64)
            U_init = U0[:, perm[:target_rank]] * np.random.randn(target_rank) * 0.1 + U0[:, perm[:target_rank]]
            V_init = V0[:, perm[:target_rank]] * np.random.randn(target_rank) * 0.1 + V0[:, perm[:target_rank]]
            W_init = W0[:, perm[:target_rank]] * np.random.randn(target_rank) * 0.1 + W0[:, perm[:target_rank]]
            U, V, W, res = _als_reduce(U_init, V_init, W_init, target_rank, n, m, p,
                                        max_iter=150, tol=1e-7,
                                        time_budget=deadline - 25)
            if res < 1e-6:
                snapped = _try_snap_and_verify(U, V, W, n, m, p)
                if snapped is not None:
                    Us, Vs, Ws = snapped
                    best_rank = target_rank
                    best_result = {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
                    break
    return best_result
=======
def construct():
    n, m, p = DIMENSIONS
    
    # The multiplication tensor for 2x4 by 4x8 is the all-ones tensor of shape (2, 4, 8).
    # T_{i,j,k} = 1 for all i, j, k.
    # This tensor has rank 1, decomposed as U * V * W where U, V, W are vectors of ones.
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    # Verify the decomposition exactly
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank-64 if verification fails (should not happen)
    U0, V0, W0 = _build_rank64()
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
>>>>>>> REPLACE
