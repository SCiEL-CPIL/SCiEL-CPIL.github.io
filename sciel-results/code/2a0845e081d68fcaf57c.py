        p_res = calc_penalty(res.x.reshape((n, 2)))
        if p_res < 1e-12:
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
