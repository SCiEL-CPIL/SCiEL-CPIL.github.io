<<<<<<< SEARCH
            # Now, get policy for M_res
            # We can either interpolate c_opts_V from V_grid to M_res, or re-solve.
            # Interpolation is faster and likely accurate enough given N_V=1000.
            # c(m) is smooth.
            policy[t, z, :] = np.interp(M_res, V_grid, c_opts_V)
=======
            # Now, get policy for M_res
            # Interpolate c_opts_V from V_grid to M_res
            policy[t, z, :] = np.interp(M_res, V_grid, c_opts_V)
>>>>>>> REPLACE

