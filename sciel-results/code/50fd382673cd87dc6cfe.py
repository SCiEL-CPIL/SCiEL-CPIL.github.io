        sol = solve_ivp(
            f, 
            (t0, t1), 
            y0_arr, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13,
            dense_output=False
        )
