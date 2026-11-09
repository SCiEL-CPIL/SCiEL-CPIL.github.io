<<<<<<< SEARCH
    def greedy_place(ordering):
        placements = [None] * n
        bins = []  # list of lists of (x, y, z, w, h, d)

        def find_position_in_bin(w, h, d, bin_boxes):
            # Candidate positions: 0 and edges of existing boxes
            x_cands = sorted({0} | {bx for bx, _, _, _, _, _ in bin_boxes} | {bx + bw for bx, _, _, bw, _, _ in bin_boxes})
            y_cands = sorted({0} | {by for _, by, _, _, _, _ in bin_boxes} | {by + bh for _, by, _, _, bh, _ in bin_boxes})
            z_cands = sorted({0} | {bz for _, _, bz, _, _, _ in bin_boxes} | {bz + bd for _, _, bz, _, _, bd in bin_boxes})

            x_cands = [x for x in x_cands if 0 <= x and x + w <= W]
            y_cands = [y for y in y_cands if 0 <= y and y + h <= H]
            z_cands = [z for z in z_cands if 0 <= z and z + d <= D]

            if not x_cands or not y_cands or not z_cands:
                return None

            best_pos = None
            best_cost = float('inf')

            for z in z_cands:
                for y in y_cands:
                    for x in x_cands:
                        # Check overlap
                        overlaps = False
                        for bx, by, bz, bw, bh, bd in bin_boxes:
                            if x < bx + bw and bx < x + w and \
                               y < by + bh and by < y + h and \
                               z < bz + bd and bz < z + d:
                                overlaps = True
                                break
                        if overlaps:
                            continue
                        # Cost: prefer lower z, then lower y, then lower x
                        cost = z * (H + 1) * (W + 1) + y * (W + 1) + x
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (x, y, z)
                            if best_pos == (0, 0, 0):
                                return best_pos

            return best_pos

        for box_idx in ordering:
            w, h, d = boxes[box_idx]

            # Try to place in existing bins, prefer bins with more volume used
            placed = False
            bin_indices = list(range(len(bins)))
            # Sort bins by volume used (descending) to try fullest bins first
            bin_indices.sort(key=lambda bi: sum(bw * bh * bd for _, _, _, bw, bh, bd in bins[bi]), reverse=True)

            for bin_i in bin_indices:
                pos = find_position_in_bin(w, h, d, bins[bin_i])
                if pos is not None:
                    x, y, z = pos
                    bins[bin_i].append((x, y, z, w, h, d))
                    placements[box_idx] = (bin_i, x, y, z, w, h, d)
                    placed = True
                    break

            if not placed:
                new_bin_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements[box_idx] = (new_bin_idx, 0, 0, 0, w, h, d)

        return placements
=======
    def greedy_place(ordering):
        placements = [None] * n
        bins = []  # list of lists of (x, y, z, w, h, d)

        def find_position_in_bin(w, h, d, bin_boxes):
            # Candidate positions: 0 and edges of existing boxes
            x_cands = sorted({0} | {bx for bx, _, _, _, _, _ in bin_boxes} | {bx + bw for bx, _, _, bw, _, _ in bin_boxes})
            y_cands = sorted({0} | {by for _, by, _, _, _, _ in bin_boxes} | {by + bh for _, by, _, _, bh, _ in bin_boxes})
            z_cands = sorted({0} | {bz for _, _, bz, _, _, _ in bin_boxes} | {bz + bd for _, _, bz, _, _, bd in bin_boxes})

            x_cands = [x for x in x_cands if 0 <= x and x + w <= W]
            y_cands = [y for y in y_cands if 0 <= y and y + h <= H]
            z_cands = [z for z in z_cands if 0 <= z and z + d <= D]

            if not x_cands or not y_cands or not z_cands:
                return None

            best_pos = None
            best_cost = float('inf')

            for z in z_cands:
                for y in y_cands:
                    for x in x_cands:
                        # Check overlap
                        overlaps = False
                        for bx, by, bz, bw, bh, bd in bin_boxes:
                            if x < bx + bw and bx < x + w and \
                               y < by + bh and by < y + h and \
                               z < bz + bd and bz < z + d:
                                overlaps = True
                                break
                        if overlaps:
                            continue
                        # Cost: prefer lower z, then lower y, then lower x
                        cost = z * (H + 1) * (W + 1) + y * (W + 1) + x
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (x, y, z)
                            if best_pos == (0, 0, 0):
                                return best_pos

            return best_pos

        for box_idx in ordering:
            w, h, d = boxes[box_idx]

            # Try to place in existing bins, prefer bins with more volume used
            placed = False
            bin_indices = list(range(len(bins)))
            # Sort bins by volume used (descending) to try fullest bins first
            bin_indices.sort(key=lambda bi: sum(bw * bh * bd for _, _, _, bw, bh, bd in bins[bi]), reverse=True)

            for bin_i in bin_indices:
                pos = find_position_in_bin(w, h, d, bins[bin_i])
                if pos is not None:
                    x, y, z = pos
                    bins[bin_i].append((x, y, z, w, h, d))
                    placements[box_idx] = (bin_i, x, y, z, w, h, d)
                    placed = True
                    break

            if not placed:
                new_bin_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements[box_idx] = (new_bin_idx, 0, 0, 0, w, h, d)

        return placements

    def local_search_improve(placements, max_iters=200):
        """Try to improve by moving boxes between bins to reduce B or improve last-bin fill."""
        if time.time() > deadline:
            return placements

        n_bins = max(p[0] for p in placements) + 1 if placements else 0
        bin_contents = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            bin_contents[b].append((i, x, y, z, w, h, d))

        def compute_current_score():
            bin_labels = set(p[0] for p in placements)
            B = len(bin_labels)
            max_bin = max(p[0] for p in placements)
            last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == max_bin)
            occupied_frac = last_bin_vol / container_vol if container_vol > 0 else 0.0
            return B + 0.1 * (1 - occupied_frac)

        best_score = compute_current_score()
        current_placements = [p[:] for p in placements]

        for it in range(max_iters):
            if time.time() > deadline:
                break

            improved = False
            # Try moving each box to a different bin
            for box_i in range(n):
                b_cur, x, y, z, w, h, d = current_placements[box_i]

                # Remove from current bin
                cur_bin_boxes = [item for item in bin_contents[b_cur] if item[0] != box_i]

                # Try placing in each other bin
                for b_target in sorted(bin_contents.keys()):
                    if b_target == b_cur:
                        continue

                    target_boxes = [item for item in bin_contents[b_target] if item[0] != box_i]
                    target_dims = [(tx, ty, tz, tw, th, td) for (_, tx, ty, tz, tw, th, td) in target_boxes]

                    pos = find_position_in_bin(w, h, d, target_dims)
                    if pos is not None:
                        # Tentatively move
                        new_placements = [p[:] for p in current_placements]
                        new_placements[box_i] = (b_target, pos[0], pos[1], pos[2], w, h, d)

                        if not validate_placement(new_placements):
                            continue

                        new_score = compute_current_score_for(new_placements)
                        if new_score < best_score - 1e-9:
                            best_score = new_score
                            current_placements = new_placements
                            improved = True
                            break

                if improved:
                    break

            if not improved:
                break

        return current_placements

    def compute_current_score_for(placements_list):
        bin_labels = set(p[0] for p in placements_list)
        B = len(bin_labels)
        max_bin = max(p[0] for p in placements_list)
        last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements_list if p[0] == max_bin)
        occupied_frac = last_bin_vol / container_vol if container_vol > 0 else 0.0
        return B + 0.1 * (1 - occupied_frac)
>>>>>>> REPLACE

