<<<<<<< SEARCH
    # Build the rank-28 decomposition
    U28, V28, W28 = build_rank_28()
    if check_decomposition(U28, V28, W28, T):
        # Try to reduce to rank 27
        result_27 = try_rank_27(U28, V28, W28, T)
        if result_27 is not None:
            U27, V27, W27 = result_27
            if check_decomposition(U27, V27, W27, T):
                return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
        
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
=======
    # Build the rank-28 decomposition
    U28, V28, W28 = build_rank_28()
    if check_decomposition(U28, V28, W28, T):
        # Try to reduce to rank 27
        result_27 = try_rank_27(U28, V28, W28, T)
        if result_27 is not None:
            U27, V27, W27 = result_27
            if check_decomposition(U27, V27, W27, T):
                return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
        
        # Try to find rank-26 by removing two terms and checking if residual is rank <= 2
        if time_left() > 10:
            result_26 = try_rank_26(U28, V28, W28, T)
            if result_26 is not None:
                U26, V26, W26 = result_26
                if check_decomposition(U26, V26, W26, T):
                    return {"U": U26.tolist(), "V": V26.tolist(), "W": W26.tolist()}
        
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
>>>>>>> REPLACE

Now I need to add the `try_rank_26` function. Let me add it after the `try_rank_27` function:

