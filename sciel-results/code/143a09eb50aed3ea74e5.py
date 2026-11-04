<<<<<<< SEARCH
    # Phase 2: Gradient polish on best
    if best_centers is not None:
        if time.time() < deadline - 30:
            centers_grad = gradient_polish(best_centers, max_iter=400)
            centers_grad = best_fit_local_search(centers_grad, max_sweeps=25)
            evaluate_and_update(centers_grad)

    # Phase 3: High-precision polish with tangent solving
    if best_centers is not None:
        if time.time() < deadline - 15:
            centers_hp = high_precision_polish(best_centers, max_iter=150)
            evaluate_and_update(centers_hp)

        # Multiple gradient + high-precision cycles
        if time.time() < deadline - 8:
            for _ in range(4):
                if time.time() > deadline - 5:
                    break
                centers_g = gradient_polish(best_centers, max_iter=300)
                centers_g = high_precision_polish(centers_g, max_iter=80)
                if not evaluate_and_update(centers_g):
                    break

    # Final: one more high-precision pass
    if best_centers is not None and time.time() < deadline - 3:
        centers_final = high_precision_polish(best_centers, max_iter=50)
        evaluate_and_update(centers_final)
=======
    # Phase 2: Gradient polish on best
    if best_centers is not None:
        if time.time() < deadline - 30:
            centers_grad = gradient_polish(best_centers, max_iter=400)
            centers_grad = best_fit_local_search(centers_grad, max_sweeps=25)
            evaluate_and_update(centers_grad)

    # Phase 3: High-precision polish with tangent solving
    if best_centers is not None:
        if time.time() < deadline - 15:
            centers_hp = high_precision_polish(best_centers, max_iter=150)
            evaluate_and_update(centers_hp)

        # Multiple gradient + high-precision cycles
        if time.time() < deadline - 8:
            for _ in range(4):
                if time.time() > deadline - 5:
                    break
                centers_g = gradient_polish(best_centers, max_iter=300)
                centers_g = high_precision_polish(centers_g, max_iter=80)
                if not evaluate_and_update(centers_g):
                    break

    # Phase 4: Grow radii post-hoc
    if best_centers is not None and time.time() < deadline - 8:
        centers_grown, radii_grown = grow_radii(best_centers, max_iter=100)
        result_grown = np.column_stack([centers_grown, radii_grown])
        if is_valid(result_grown, tol=1e-6):
            score_grown = np.sum(radii_grown)
            if score_grown > best_score:
                best_score = score_grown
                best_centers = centers_grown.copy()
                best_result = result_grown

    # Final: one more high-precision pass
    if best_centers is not None and time.time() < deadline - 3:
        centers_final = high_precision_polish(best_centers, max_iter=50)
        evaluate_and_update(centers_final)
        # One more grow_radii
        if time.time() < deadline - 1:
            centers_grown2, radii_grown2 = grow_radii(best_centers, max_iter=50)
            result_grown2 = np.column_stack([centers_grown2, radii_grown2])
            if is_valid(result_grown2, tol=1e-6):
                score_grown2 = np.sum(radii_grown2)
                if score_grown2 > best_score:
                    best_score = score_grown2
                    best_centers = centers_grown2.copy()
                    best_result = result_grown2
>>>>>>> REPLACE
