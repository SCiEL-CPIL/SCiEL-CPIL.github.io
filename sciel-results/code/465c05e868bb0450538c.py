<<<<<<< SEARCH
    # Final greedy polish
    if best_overall is not None:
        signs = best_overall.copy()
        ap_sums = compute_ap_sums(signs)
        hist = build_hist(ap_sums)
        current_max = get_max_from_hist(hist)
        
        improved = True
        # Run polish for a bit longer if time permits
        while improved and time.time() < deadline - 0.5:
            improved = False
            for i in range(n):
                if time.time() > deadline - 0.5:
                    break
                old_sign = signs[i]
                signs[i] = -old_sign
                for ap_id in pos_to_aps[i]:
                    old_abs = abs(ap_sums[ap_id])
                    hist[old_abs] -= 1
                    ap_sums[ap_id] -= 2 * old_sign
                    new_abs = abs(ap_sums[ap_id])
                    hist[new_abs] += 1
                new_max = get_max_from_hist(hist)
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                else:
                    signs[i] = old_sign
                    for ap_id in pos_to_aps[i]:
                        old_abs = abs(ap_sums[ap_id])
                        hist[old_abs] -= 1
                        ap_sums[ap_id] += 2 * old_sign
                        new_abs = abs(ap_sums[ap_id])
                        hist[new_abs] += 1
        
        if current_max < best_overall_max:
            best_overall_max = current_max
            best_overall = signs.copy()
=======
    # Final greedy polish
    if best_overall is not None:
        signs = best_overall.copy()
        ap_sums = compute_ap_sums(signs)
        current_max = int(np.max(np.abs(ap_sums)))
        
        improved = True
        # Run polish for a bit longer if time permits
        while improved and time.time() < deadline - 0.5:
            improved = False
            for i in range(n):
                if time.time() > deadline - 0.5:
                    break
                old_sign = int(signs[i])
                signs[i] = -old_sign
                affected_aps = pos_to_aps[i]
                if len(affected_aps) > 0:
                    ap_sums[affected_aps] -= 2 * old_sign
                new_max = int(np.max(np.abs(ap_sums)))
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                else:
                    signs[i] = old_sign
                    if len(affected_aps) > 0:
                        ap_sums[affected_aps] += 2 * old_sign
        
        if current_max < best_overall_max:
            best_overall_max = current_max
            best_overall = signs.copy()
>>>>>>> REPLACE
