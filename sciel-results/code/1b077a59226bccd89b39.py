<<<<<<< SEARCH
    # Final refinement
    if time.time() < deadline:
        for pos in range(12):
            if time.time() > deadline:
                break
            old_val = best_poly[pos]
            for new_val in range(-32, 33):
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
    # Final refinement: try all values in [-64,64] excluding 0 for each position
    if time.time() < deadline:
        for pos in range(12):
            if time.time() > deadline:
                break
            old_val = best_poly[pos]
            # Try values that might create cancellations first
            sq = np.convolve(best_poly, best_poly)
            for d in range(23):
                other = d - pos
                if 0 <= other <= 11 and other != pos:
                    other_sum = 0
                    for i in range(12):
                        j = d - i
                        if 0 <= j <= 11 and i != pos and j != pos:
                            other_sum += best_poly[i] * best_poly[j]
                    if best_poly[other] != 0:
                        target = -other_sum / (2 * best_poly[other])
                        if abs(target) <= 64:
                            tv = int(round(target))
                            for dv in [0, 1, -1, 2, -2]:
                                nv = tv + dv
                                if -64 <= nv <= 64 and nv != 0 and nv != old_val:
                                    best_poly[pos] = nv
                                    new_score = evaluate(best_poly)
                                    if new_score < best_score:
                                        best_score = new_score
            # Then sweep remaining range
            for new_val in range(-64, 65):
                if time.time() > deadline:
                    break
                if new_val == old_val or new_val == 0:
                    continue
                best_poly[pos] = new_val
                new_score = evaluate(best_poly)
                if new_score < best_score:
                    best_score = new_score
            best_poly[pos] = old_val
    
    return best_poly
>>>>>>> REPLACE
