    U_best, V_best, W_best, rank_best = rank37_decomposition()
    err = verify(U_best, V_best, W_best, rank_best)

    if err > 1e-8:
        # ... rank55 ...
        U_best, V_best, W_best, rank_best = rank55_decomposition()

    final_err = verify(U_best, V_best, W_best, rank_best)
    if final_err > 1e-8:
        # ... schoolbook ...
