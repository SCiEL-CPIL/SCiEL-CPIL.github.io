<<<<<<< SEARCH
    # Phase 2: Border-rank heuristic via perturbation + ALS
    if time.time() < deadline - 20:
        result = _try_sparse_rank35(U, V, W, time_limit=deadline - time.time() - 10)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_fast(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()

    # Phase 3: Try removing pairs (if time permits)
    if time.time() < deadline - 5:
        rank = U.shape[1]
        for i in range(rank):
            if time.time() > deadline - 5:
                break
            for j in range(i+1, rank):
                if time.time() > deadline - 5:
                    break
                mask = [k for k in range(rank) if k != i and k != j]
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                if _verify_fast(U_new, V_new, W_new):
                    U, V, W = U_new, V_new, W_new
                    rank = U.shape[1]
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            if time.time() > deadline - 5:
                break
            if U.shape[1] < rank:
                break
=======
    # Phase 2: Border-rank heuristic via perturbation + ALS
    if time.time() < deadline - 20:
        remaining_time = deadline - time.time() - 10
        if remaining_time > 5:
            result = _try_sparse_rank35(U, V, W, time_limit=remaining_time)
            if result is not None:
                U_r, V_r, W_r = result
                if _verify_fast(U_r, V_r, W_r):
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()

    # Phase 3: Try removing pairs (if time permits) using vectorized approach
    if time.time() < deadline - 5:
        rank = U.shape[1]
        T_full = np.einsum('ir,jr,kr->ijk', U, V, W)
        found_pair = False
        for i in range(rank):
            if time.time() > deadline - 5:
                break
            for j in range(i+1, rank):
                if time.time() > deadline - 5:
                    break
                # Check if removing columns i and j still gives T=1
                u_i = U[:, i].reshape(2, 4)
                v_i = V[:, i].reshape(4, 5)
                w_i = W[:, i].reshape(5, 2)
                term_i = np.einsum('ij,jk,ki->ijk', u_i, v_i, w_i)
                
                u_j = U[:, j].reshape(2, 4)
                v_j = V[:, j].reshape(4, 5)
                w_j = W[:, j].reshape(5, 2)
                term_j = np.einsum('ij,jk,ki->ijk', u_j, v_j, w_j)
                
                T_without = T_full - term_i - term_j
                if np.allclose(T_without, 1.0, atol=1e-8):
                    mask = [k for k in range(rank) if k != i and k != j]
                    U = U[:, mask]
                    V = V[:, mask]
                    W = W[:, mask]
                    rank = U.shape[1]
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    found_pair = True
                    break
            if found_pair or time.time() > deadline - 5:
                break
>>>>>>> REPLACE
