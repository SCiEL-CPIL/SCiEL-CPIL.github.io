    def try_find_rank_16_block_v2():
        # Try random initializations
        best_err = np.inf
        best_x = None
        for _ in range(5):
            U_rand = np.random.randn(9, 16)
            V_rand = np.random.randn(6, 16)
            W_rand = np.random.randn(6, 16)
            x0 = np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()])
            try:
                result = least_squares(residuals_16, x0, method='lm', max_nfev=500)
                if np.max(np.abs(result.fun)) < best_err:
                    best_err = np.max(np.abs(result.fun))
                    best_x = result.x
            except:
                pass
        if best_x is not None and best_err < 1e-8:
             # Check rounding
             pass
        return None
