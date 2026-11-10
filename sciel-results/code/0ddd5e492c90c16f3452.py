<<<<<<< SEARCH
    # Phase 3: Final local refinement
    if time.time() < deadline:
        for pos in range(12):
            if time.time() > deadline:
                break
            old_val = best_poly[pos]
            for new_val in range(-64, 65):
                if new_val == old_val or new_val == 0:
                    continue
                best_poly[pos] = new_val
                new_score = evaluate(best_poly)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_poly[pos] = old_val
            best_poly[pos] = old_val

    # Phase 4: Second refinement pass
    if time.time() < deadline:
        for pos in range(12):
            if time.time() > deadline:
                break
            old_val = best_poly[pos]
            for new_val in range(-64, 65):
                if new_val == old_val or new_val == 0:
                    continue
                best_poly[pos] = new_val
                new_score = evaluate(best_poly)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_poly[pos] = old_val
            best_poly[pos] = old_val

    return best_poly
=======
    # Phase 3: Final local refinement - exhaustive over all values
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for pos in range(12):
                if time.time() > deadline - 5:
                    break
                old_val = best_poly[pos]
                for new_val in range(-64, 65):
                    if new_val == old_val or new_val == 0:
                        continue
                    best_poly[pos] = new_val
                    new_score = evaluate(best_poly)
                    if new_score < best_score:
                        best_score = new_score
                        improved = True
                    else:
                        best_poly[pos] = old_val
                best_poly[pos] = old_val

    # Phase 4: Multi-start local refinement from best_poly
    if time.time() < deadline - 3:
        # Try perturbing pairs of coefficients
        for pos1 in range(12):
            if time.time() > deadline - 3:
                break
            for pos2 in range(pos1 + 1, 12):
                if time.time() > deadline - 3:
                    break
                old1, old2 = best_poly[pos1], best_poly[pos2]
                # Try a limited set of coupled changes
                for d1 in [-2, -1, 1, 2]:
                    for d2 in [-2, -1, 1, 2]:
                        if time.time() > deadline - 3:
                            break
                        nv1, nv2 = old1 + d1, old2 + d2
                        if nv1 == 0 or nv2 == 0:
                            continue
                        if not (-64 <= nv1 <= 64 and -64 <= nv2 <= 64):
                            continue
                        best_poly[pos1] = nv1
                        best_poly[pos2] = nv2
                        new_score = evaluate(best_poly)
                        if new_score < best_score:
                            best_score = new_score
                        else:
                            best_poly[pos1] = old1
                            best_poly[pos2] = old2
                best_poly[pos1] = old1
                best_poly[pos2] = old2

    # Phase 5: Final single-pass exhaustive refinement
    if time.time() < deadline - 2:
        for pos in range(12):
            if time.time() > deadline - 2:
                break
            old_val = best_poly[pos]
            for new_val in range(-64, 65):
                if new_val == old_val or new_val == 0:
                    continue
                best_poly[pos] = new_val
                new_score = evaluate(best_poly)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_poly[pos] = old_val
            best_poly[pos] = old_val

    return best_poly
>>>>>>> REPLACE
