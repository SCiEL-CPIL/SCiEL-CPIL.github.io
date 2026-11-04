<<<<<<< SEARCH
    # Final Polish: High-precision grid search on outer parameters
    best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
    best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
    
    # Coarse-to-fine polish
    best_ocx, best_ocy, best_otheta, best_side = polish_outer_hexagon(
        centers, angles, best_ocx, best_ocy, best_otheta, max_iters=300
    )
    
    # Verify final validity
    if not is_valid(centers, angles):
        # Fallback to initial lattice if optimization resulted in invalid configuration
        centers = [[c[0], c[1]] for c in centers_init]
        angles = [30.0] * 12
        ocx, ocy, otheta = 0.0, 0.0, 0.0
        best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
        # Polish the fallback too
        best_ocx, best_ocy, best_otheta, best_side = polish_outer_hexagon(
            centers, angles, best_ocx, best_ocy, best_otheta, max_iters=200
        )
=======
    # Final Polish: High-precision grid search on outer parameters
    best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
    best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
    
    # Coarse-to-fine polish
    best_ocx, best_ocy, best_otheta, best_side = polish_outer_hexagon(
        centers, angles, best_ocx, best_ocy, best_otheta, max_iters=400
    )
    
    # Verify final validity
    if not is_valid(centers, angles):
        # Fallback to initial lattice if optimization resulted in invalid configuration
        centers = [[c[0], c[1]] for c in centers_init]
        angles = [30.0] * 12
        ocx, ocy, otheta = 0.0, 0.0, 0.0
        best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
        # Polish the fallback too
        best_ocx, best_ocy, best_otheta, best_side = polish_outer_hexagon(
            centers, angles, best_ocx, best_ocy, best_otheta, max_iters=300
        )
>>>>>>> REPLACE
