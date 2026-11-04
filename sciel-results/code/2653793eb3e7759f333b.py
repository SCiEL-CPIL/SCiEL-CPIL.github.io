<<<<<<< SEARCH
    # Final high-precision polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 10:
        polished = _polish(best_result, best_W, best_H, max_iter=8000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], best_W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], best_H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(best_W, best_H) / 2)
        polished, valid_final = _separate_circles(polished, best_W, best_H, max_iter=2000)
        if valid_final:
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished.copy()

    # Second polish pass if time permits
    if best_result is not None and time.time() - start_time < DEADLINE - 5:
        polished2 = _polish(best_result, best_W, best_H, max_iter=4000)
        polished2[:, 0] = np.clip(polished2[:, 0], polished2[:, 2], best_W - polished2[:, 2])
        polished2[:, 1] = np.clip(polished2[:, 1], polished2[:, 2], best_H - polished2[:, 2])
        polished2[:, 2] = np.clip(polished2[:, 2], 0.001, min(best_W, best_H) / 2)
        polished2, valid_final2 = _separate_circles(polished2, best_W, best_H, max_iter=2000)
        if valid_final2:
            s = np.sum(polished2[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished2.copy()
=======
    # Final high-precision polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 15:
        polished = _polish(best_result, best_W, best_H, max_iter=10000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], best_W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], best_H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(best_W, best_H) / 2)
        polished, valid_final = _separate_circles(polished, best_W, best_H, max_iter=3000)
        if valid_final:
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished.copy()

    # Second polish pass if time permits
    if best_result is not None and time.time() - start_time < DEADLINE - 8:
        polished2 = _polish(best_result, best_W, best_H, max_iter=6000)
        polished2[:, 0] = np.clip(polished2[:, 0], polished2[:, 2], best_W - polished2[:, 2])
        polished2[:, 1] = np.clip(polished2[:, 1], polished2[:, 2], best_H - polished2[:, 2])
        polished2[:, 2] = np.clip(polished2[:, 2], 0.001, min(best_W, best_H) / 2)
        polished2, valid_final2 = _separate_circles(polished2, best_W, best_H, max_iter=3000)
        if valid_final2:
            s = np.sum(polished2[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished2.copy()

    # Third polish pass if time permits
    if best_result is not None and time.time() - start_time < DEADLINE - 3:
        polished3 = _polish(best_result, best_W, best_H, max_iter=3000)
        polished3[:, 0] = np.clip(polished3[:, 0], polished3[:, 2], best_W - polished3[:, 2])
        polished3[:, 1] = np.clip(polished3[:, 1], polished3[:, 2], best_H - polished3[:, 2])
        polished3[:, 2] = np.clip(polished3[:, 2], 0.001, min(best_W, best_H) / 2)
        polished3, valid_final3 = _separate_circles(polished3, best_W, best_H, max_iter=2000)
        if valid_final3:
            s = np.sum(polished3[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished3.copy()
>>>>>>> REPLACE
