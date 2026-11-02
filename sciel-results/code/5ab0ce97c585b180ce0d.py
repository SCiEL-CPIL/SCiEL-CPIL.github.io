<<<<<<< SEARCH
    # Phase 10: One more spectral polish if time permits
    if not _check_deadline():
        best_x, best_val = _spectral_coord_descent(best_x, rng, max_iters=2000, initial_step=0.03)
=======
    # Phase 10: Another PSD polish if time permits
    if not _check_deadline():
        best_x, best_val = _psd_projected_gradient(best_x, rng, max_iters=3000, lr=0.003)
>>>>>>> REPLACE

