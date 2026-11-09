<<<<<<< SEARCH
    # Time-guarded polish: try to move misplaced boxes into better positions.
    cur_score = score_of(placement)
    while time.time() < DEADLINE - 5:
        improved = False
        for i in range(len(boxes)):
            if time.time() >= DEADLINE - 5:
                break
            bi, x, y, z = placement[i]
            w, h, d = boxes[i]
            # current wasted volume at current pos
            cur_waste = (W - (x + w)) * (H - (y + h)) * (D - (z + d))
            # try to find a better position in same bin
            bboxes = [(bx, by, bz, bw, bh, bd)
                      for j in range(len(boxes)) if placement[j][0] == bi and j != i]
            pos = _best_position(bboxes, W, H, D, w, h, d)
            if pos is not None:
                nw, nh, nz = pos
                new_waste = (W - (nw + w)) * (H - (nh + h)) * (D - (nz + d))
                if new_waste < cur_waste - 1e-9:
                    placement[i] = (bi, nw, nh, nz)
                    improved = True
        if not improved:
            break
=======
    # Time-guarded polish: try to move misplaced boxes into better positions.
    cur_score = score_of(placement)
    while time.time() < DEADLINE - 5:
        improved = False
        for i in range(len(boxes)):
            if time.time() >= DEADLINE - 5:
                break
            bi, x, y, z = placement[i]
            w, h, d = boxes[i]
            # try to find a better position in same bin
            bboxes = [(bx, by, bz, bw, bh, bd)
                      for j in range(len(boxes)) if placement[j][0] == bi and j != i]
            pos = _find_best_position(bboxes, W, H, D, w, h, d)
            if pos is not None:
                nw, nh, nz = pos
                # Better if priority (z, y, x) is lower
                if (nz, nh, nw) < (z, y, x):
                    placement[i] = (bi, nw, nh, nz)
                    improved = True
        if not improved:
            break
>>>>>>> REPLACE
