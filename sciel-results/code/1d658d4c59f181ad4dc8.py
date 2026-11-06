<<<<<<< SEARCH
    else:
        # Take top 50 by value density as candidates
        densities = fv / cf
        top_k = 50
        top_local = np.argsort(densities)[-top_k:]
        candidates = feasible_indices[top_local]
        cf = fw[top_local]
        cv = fv[top_local]
=======
    else:
        # Take top 50 by value density as candidates
        densities = fv / fw
        top_k = 50
        top_local = np.argsort(densities)[-top_k:]
        candidates = feasible_indices[top_local]
        cf = fw[top_local]
        cv = fv[top_local]
>>>>>>> REPLACE
