<<<<<<< SEARCH
    # Return the best result found
    U, V, W = best_result
    return {"U": U, "V": W}  # BUG: should be W not W... let me fix
=======
    # Return the best result found
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
