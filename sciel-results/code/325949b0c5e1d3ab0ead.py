<<<<<<< SEARCH
    # Phase 7: Another best-fit pass after gradient polish
    if time.time() < deadline - 5:
        bf_after_grad, bf_ag_obj = _best_fit_local_search(
            best_pos, n_iter=3000,
            seed=555, deadline=deadline - 3
        )
        if bf_ag_obj > best_obj:
            best_pos = bf_after_grad
            best_obj = bf_ag_obj
=======
    # Phase 7: Another best-fit pass after gradient polish
    if time.time() < deadline - 6:
        bf_after_grad, bf_ag_obj = _best_fit_local_search(
            best_pos, n_iter=3000,
            seed=555, deadline=deadline - 4
        )
        if bf_ag_obj > best_obj:
            best_pos = bf_after_grad
            best_obj = bf_ag_obj
>>>>>>> REPLACE

