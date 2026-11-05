    def construct():
        n, m, p = DIMENSIONS
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        best_rank = n * m * p
        best_result = (U_sb, V_sb, W_sb)

        # Search sub-problem ranks
        for r_sub in range(max(m, p), m * p + 1):
            if _time_left() < 10.0:
                break
            result_sub = _try_rank_sub(r_sub, m, p, max_iters=50, seed=42, patterns=3)
            if result_sub is not None:
                U_sub, V_sub, W_sub = result_sub
                U_full, V_full, W_full = _tile_3x(U_sub, V_sub, W_sub, n, m, p)
                if _verify(U_full, V_full, W_full, n, m, p):
                    full_rank = 3 * r_sub
                    if full_rank < best_rank:
                        best_rank = full_rank
                        best_result = (U_full, V_full, W_full)
                    break

        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    