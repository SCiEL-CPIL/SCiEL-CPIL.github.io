<<<<<<< SEARCH
            polished = gradient_polish(current_centers, max_iters=100, use_smooth=True)

            if polished is not None:
                p_score = np.sum(polished[:, 2])
                if p_score > best_score:
                    best_score = p_score
                    best_result = polished.copy()
            else:
                polished = gradient_polish(current_centers, max_iters=80, use_smooth=False)
                if polished is not None:
                    p_score = np.sum(polished[:, 2])
                    if p_score > best_score:
                        best_score = p_score
                        best_result = polished.copy()
                elif is_valid(np.column_stack([ls_centers, ls_radii]), tol=1e-6) and ls_score > best_score:
                    best_score = ls_score
                    best_result = np.column_stack([ls_centers, ls_radii]).copy()
=======
            polished = gradient_polish(current_centers, max_iters=150, use_smooth=True)

            if polished is not None:
                p_score = np.sum(polished[:, 2])
                if p_score > best_score:
                    best_score = p_score
                    best_result = polished.copy()
            else:
                polished = gradient_polish(current_centers, max_iters=100, use_smooth=False)
                if polished is not None:
                    p_score = np.sum(polished[:, 2])
                    if p_score > best_score:
                        best_score = p_score
                        best_result = polished.copy()
                elif is_valid(np.column_stack([ls_centers, ls_radii]), tol=1e-6) and ls_score > best_score:
                    best_score = ls_score
                    best_result = np.column_stack([ls_centers, ls_radii]).copy()
>>>>>>> REPLACE

