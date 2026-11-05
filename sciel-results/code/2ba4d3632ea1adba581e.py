<<<<<<< SEARCH
    # Phase 4: Relabel bins to be consecutive
    if bins:
        occupied_bins = sorted([bi for bi in range(len(bins)) if bins[bi]])
        label_map = {old: new for new, old in enumerate(occupied_bins)}
        
        for i in range(n):
            old_bin = result[i][0]
            if old_bin in label_map:
                new_bin = label_map[old_bin]
                if new_bin != old_bin:
                    result[i] = (new_bin, result[i][1], result[i][2], result[i][3], 
                                 result[i][4], result[i][5], result[i][6])
    
    return result
=======
    # Phase 4: Final consolidation pass
    if time.time() < deadline - 1.0:
        for src_bin in range(len(bins)-1, 0, -1):
            if time.time() > deadline - 1.0:
                break
            if not bins[src_bin]:
                continue
            boxes_to_move = list(bins[src_bin])
            boxes_to_move.sort(key=lambda p: p[4]*p[5]*p[6], reverse=True)
            can_move_all = True
            saved_states = []
            for p in boxes_to_move:
                bi, x, y, z, w, h, d = p
                box = (w, h, d)
                box_idx = None
                for i in range(n):
                    if result[i] and result[i][0] == src_bin and result[i][1] == x \
                       and result[i][2] == y and result[i][3] == z \
                       and result[i][4] == w and result[i][5] == h and result[i][6] == d:
                        box_idx = i
                        break
                if box_idx is None:
                    can_move_all = False
                    break
                _update_grid(bin_grids[src_bin], box, (x, y, z), add=False)
                bin_volumes[src_bin] -= w*h*d
                saved_states.append((box_idx, x, y, z, w, h, d))
                placed = False
                for dst_bin in range(len(bins)):
                    if dst_bin == src_bin:
                        continue
                    pos = _find_position(box, bins[dst_bin], bin_grids[dst_bin], W, H, D)
                    if pos is not None:
                        nx, ny, nz = pos
                        bins[dst_bin].append((dst_bin, nx, ny, nz, w, h, d))
                        _update_grid(bin_grids[dst_bin], box, (nx, ny, nz))
                        bin_volumes[dst_bin] += w*h*d
                        result[box_idx] = (dst_bin, nx, ny, nz, w, h, d)
                        placed = True
                        break
                if not placed:
                    for bi2, xi, yi, zi, wi, hi, di in saved_states:
                        _update_grid(bin_grids[src_bin], (wi, hi, di), (xi, yi, zi), add=True)
                        bin_volumes[src_bin] += wi*hi*di
                    can_move_all = False
                    break
            if can_move_all:
                bins[src_bin] = []
                bin_volumes[src_bin] = 0
    
    # Phase 5: Relabel bins to be consecutive
    if bins:
        occupied_bins = sorted([bi for bi in range(len(bins)) if bins[bi]])
        label_map = {old: new for new, old in enumerate(occupied_bins)}
        
        for i in range(n):
            old_bin = result[i][0]
            if old_bin in label_map:
                new_bin = label_map[old_bin]
                if new_bin != old_bin:
                    result[i] = (new_bin, result[i][1], result[i][2], result[i][3], 
                                 result[i][4], result[i][5], result[i][6])
    
    return result
>>>>>>> REPLACE
