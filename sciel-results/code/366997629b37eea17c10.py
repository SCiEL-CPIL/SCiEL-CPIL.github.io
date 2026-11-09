import time
import math
import random
import numpy as np
from collections import defaultdict

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s kill

    W, H, D = container
    n = len(boxes)

    if n == 0:
        return []

    box_volumes = [w * h * d for w, h, d in boxes]
    total_volume = sum(box_volumes)
    container_vol = W * H * D

    if container_vol == 0:
        return [(i, 0, 0, 0, w, h, d) for i, (w, h, d) in enumerate(boxes)]

    # Sort indices by volume descending (Best Fit Decreasing)
    vol_sorted = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)

    # We will run multiple random restarts or perturbations if time permits, 
    # but for 40 boxes, a single good run with strong local search is often sufficient.
    # Let's stick to one robust run to ensure we finish in time.
    
    def compute_score(placements):
        if len(placements) != n:
            return float('inf')
        bin_labels = set(p[0] for p in placements)
        B = len(bin_labels)
        if B == 0:
            return 0
        max_bin = max(p[0] for p in placements)
        last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == max_bin)
        occupied_frac = last_bin_vol / container_vol if container_vol > 0 else 0.0
        return B + 0.1 * (1 - occupied_frac)

    def renumber_bins(placements):
        if len(placements) == 0:
            return placements
        mapping = {}
        next_label = 0
        new_placements = []
        for p in placements:
            old = p[0]
            if old not in mapping:
                mapping[old] = next_label
                next_label += 1
            new_placements.append((mapping[old], p[1], p[2], p[3], p[4], p[5], p[6]))
        return new_placements

    def validate_placement(placements):
        if len(placements) != n:
            return False
        bins = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            bins[b].append((x, y, z, w, h, d))
        for bin_idx, box_list in bins.items():
            for i in range(len(box_list)):
                for j in range(i + 1, len(box_list)):
                    x1, y1, z1, w1, h1, d1 = box_list[i]
                    x2, y2, z2, w2, h2, d2 = box_list[j]
                    if (x1 < x2 + w2 and x2 < x1 + w1 and
                        y1 < y2 + h2 and y2 < y1 + h1 and
                        z1 < z2 + d2 and z2 < z1 + d1):
                        return False
        return True

    class Bin:
        def __init__(self):
            self.boxes = []
            self.volume = 0

        def add_box(self, x, y, z, w, h, d):
            self.boxes.append((x, y, z, w, h, d))
            self.volume += w * h * d

        def remove_box(self, box_tuple):
            self.boxes.remove(box_tuple)
            self.volume -= box_tuple[3] * box_tuple[4] * box_tuple[5]

        def get_candidate_positions(self, w, h, d):
            """Generate candidate positions from box faces and origin."""
            x_cands = {0}
            y_cands = {0}
            z_cands = {0}
            for bx, by, bz, bw, bh, bd in self.boxes:
                x_cands.add(bx)
                x_cands.add(bx + bw)
                y_cands.add(by)
                y_cands.add(by + bh)
                z_cands.add(bz)
                z_cands.add(bz + bd)
            
            # Filter to valid ranges
            x_cands = {x for x in x_cands if 0 <= x <= W - w}
            y_cands = {y for y in y_cands if 0 <= y <= H - h}
            z_cands = {z for z in z_cands if 0 <= z <= D - d}
            
            # Sort to prefer compact positions (lower coordinates)
            x_cands = sorted(x_cands)
            y_cands = sorted(y_cands)
            z_cands = sorted(z_cands)
            
            # Limit candidates to avoid explosion
            # 10 per axis is usually enough for 40 boxes
            x_cands = x_cands[:10]
            y_cands = y_cands[:10]
            z_cands = z_cands[:10]
            
            return x_cands, y_cands, z_cands

        def try_place(self, w, h, d):
            """Try to place a box. Returns (x,y,z) or None.
            Uses Best Fit: minimize new bounding box volume."""
            if not self.boxes:
                if w <= W and h <= H and d <= D:
                    return (0, 0, 0)
                return None

            x_cands, y_cands, z_cands = self.get_candidate_positions(w, h, d)
            
            best_pos = None
            best_metric = float('inf')
            
            # Pre-calculate current bounding box maxes for speed
            max_x = max(b[0] + b[3] for b in self.boxes)
            max_y = max(b[1] + b[4] for b in self.boxes)
            max_z = max(b[2] + b[5] for b in self.boxes)
            
            for cx in x_cands:
                for cy in y_cands:
                    for cz in z_cands:
                        if cx + w > W or cy + h > H or cz + d > D:
                            continue
                        
                        # Check overlap
                        overlap = False
                        for bx, by, bz, bw, bh, bd in self.boxes:
                            if (cx < bx + bw and bx < cx + w and
                                cy < by + bh and by < cy + h and
                                cz < bz + bd and bz < cz + d):
                                overlap = True
                                break
                        if overlap:
                            continue

                        # Metric: prefer positions that minimize new bounding box volume
                        new_max_x = max(max_x, cx + w)
                        new_max_y = max(max_y, cy + h)
                        new_max_z = max(max_z, cz + d)
                        
                        # Prefer smaller new bounding box, then closer to origin
                        metric = new_max_x * new_max_y * new_max_z + 0.001 * (cx + cy + cz)
                        
                        if metric < best_metric:
                            best_metric = metric
                            best_pos = (cx, cy, cz)

            return best_pos

    def greedy_pack(ordering):
        bins = []
        placements = [None] * n

        for box_idx in ordering:
            w, h, d = boxes[box_idx]
            placed = False

            best_bin_idx = -1
            best_pos = None
            best_bin_metric = float('inf')

            for i, bin_obj in enumerate(bins):
                pos = bin_obj.try_place(w, h, d)
                if pos is not None:
                    # We prefer the bin where the box fits "best".
                    # The try_place already minimizes bounding box expansion.
                    # We can just pick the first one or the one with highest volume.
                    # Let's pick the bin with the highest current volume (Fullest Bin)
                    if best_bin_idx == -1 or bins[i].volume > bins[best_bin_idx].volume:
                        best_bin_idx = i
                        best_pos = pos

            if best_bin_idx != -1:
                x, y, z = best_pos
                bins[best_bin_idx].add_box(x, y, z, w, h, d)
                placements[box_idx] = (best_bin_idx, x, y, z, w, h, d)
            else:
                new_bin = Bin()
                new_bin.add_box(0, 0, 0, w, h, d)
                bins.append(new_bin)
                placements[box_idx] = (len(bins) - 1, 0, 0, 0, w, h, d)

        return placements, bins

    def local_search(placements, bins):
        """
        1. Try to empty the last bin by moving its boxes to others.
        2. If B doesn't change, try to fill the last bin by moving boxes from others into it.
        """
        if time.time() > deadline - 10:
            return placements, bins

        # Rebuild bins dict from placements to be safe, though we pass bins
        # We need to map box index to its position in the bin's list for removal
        # It's easier to work with a dictionary of bin -> list of (box_idx, coords)
        
        # Let's refactor to use a standard structure for local search
        bin_dict = defaultdict(list)
        for i, p in enumerate(placements):
            bin_dict[p[0]].append(i)
            
        # Re-create Bin objects from bin_dict for efficient placement checks
        # Actually, we can just use the existing `bins` list if we keep them in sync.
        # But `bins` is a list of Bin objects. `placements` is a list of tuples.
        # It's safer to rebuild a helper structure.
        
        # Let's just use the `bins` list directly, assuming it's in sync.
        # We need to know which box indices are in which bin.
        # The `bins` objects contain tuples (x,y,z,w,h,d) but not the original index.
        # This is a problem. We need to track indices.
        
        # Let's modify the greedy pack to return a structure that tracks indices.
        # Or, simpler: just use `placements` and `bin_dict` for local search, 
        # and write a helper to check if a box can be placed in a bin given its current contents.
        
        def can_place_in_bin(bin_idx, box_idx, target_pos):
            w, h, d = boxes[box_idx]
            tx, ty, tz = target_pos
            if tx + w > W or ty + h > H or tz + d > D:
                return False
            # Check overlap with all boxes in that bin
            for other_idx in bin_dict[bin_idx]:
                if other_idx == box_idx:
                    continue
                op = placements[other_idx]
                ox, oy, oz, ow, oh, od = op[1], op[2], op[3], op[4], op[5], op[6]
                if (tx < ox + ow and ox < tx + w and
                    ty < oy + oh and oy < ty + h and
                    tz < oz + od and oz < tz + d):
                    return False
            return True

        def get_candidates_for_bin(bin_idx, box_idx):
            w, h, d = boxes[box_idx]
            x_cands = {0}
            y_cands = {0}
            z_cands = {0}
            for other_idx in bin_dict[bin_idx]:
                if other_idx == box_idx:
                    continue
                op = placements[other_idx]
                ox, oy, oz, ow, oh, od = op[1], op[2], op[3], op[4], op[5], op[6]
                x_cands.add(ox)
                x_cands.add(ox + ow)
                y_cands.add(oy)
                y_cands.add(oy + oh)
                z_cands.add(oz)
                z_cands.add(oz + od)
            
            x_cands = sorted({x for x in x_cands if 0 <= x <= W - w})[:10]
            y_cands = sorted({y for y in y_cands if 0 <= y <= H - h})[:10]
            z_cands = sorted({z for z in z_cands if 0 <= z <= D - d})[:10]
            return x_cands, y_cands, z_cands

        # Phase 1: Empty the last bin
        max_bin = max(bin_dict.keys())
        while max_bin in bin_dict and bin_dict[max_bin]:
            if time.time() > deadline - 5:
                break
            
            moved_any = False
            # Try to move the largest box in the last bin first
            boxes_in_last = sorted(bin_dict[max_bin], key=lambda i: box_volumes[i], reverse=True)
            
            for box_idx in boxes_in_last[:]:
                if box_idx not in bin_dict[max_bin]:
                    continue
                
                w, h, d = boxes[box_idx]
                moved = False
                for other_bin_idx in bin_dict.keys():
                    if other_bin_idx == max_bin:
                        continue
                    if other_bin_idx not in bin_dict or not bin_dict[other_bin_idx]:
                        continue
                    
                    x_cands, y_cands, z_cands = get_candidates_for_bin(other_bin_idx, box_idx)
                    for cx in x_cands:
                        if moved: break
                        for cy in y_cands:
                            if moved: break
                            for cz in z_cands:
                                if can_place_in_bin(other_bin_idx, box_idx, (cx, cy, cz)):
                                    # Move it
                                    placements[box_idx] = (other_bin_idx, cx, cy, cz, w, h, d)
                                    bin_dict[max_bin].remove(box_idx)
                                    if not bin_dict[max_bin]:
                                        del bin_dict[max_bin]
                                    bin_dict[other_bin_idx].append(box_idx)
                                    moved = True
                                    moved_any = True
                                    break
                if moved:
                    # If we moved a box, the set of bins might have changed
                    if max_bin not in bin_dict:
                        break
                    # Recalculate max_bin if necessary? 
                    # If we emptied max_bin, the new max_bin is the next highest.
                    # But we are in a while loop, so it will re-evaluate.
                    # However, if we didn't empty it, we continue trying to move other boxes.
                    break # Restart the loop to re-sort and check new state

            if not moved_any:
                break

        # Phase 2: Fill the last bin (if B didn't decrease or we want to optimize fill)
        if bin_dict:
            max_bin = max(bin_dict.keys())
            # Try to move boxes from other bins INTO max_bin
            # This increases fill of max_bin, reducing the penalty term.
            # We should try to move boxes that fit well.
            
            # To avoid infinite loops or swapping, we only move if the source bin
            # does not become empty (unless we are okay with B decreasing, which is good).
            # Actually, if source bin becomes empty, B decreases, which is great.
            # So we allow moves that empty other bins.
            
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                # Iterate through all boxes in other bins
                for bin_idx in sorted(bin_dict.keys()):
                    if bin_idx == max_bin:
                        continue
                    if not bin_dict[bin_idx]:
                        continue
                    
                    # Try to move each box in this bin to max_bin
                    for box_idx in bin_dict[bin_idx][:]:
                        if box_idx not in bin_dict[bin_idx]:
                            continue
                        
                        w, h, d = boxes[box_idx]
                        x_cands, y_cands, z_cands = get_candidates_for_bin(max_bin, box_idx)
                        
                        for cx in x_cands:
                            if improved: break
                            for cy in y_cands:
                                if improved: break
                                for cz in z_cands:
                                    if can_place_in_bin(max_bin, box_idx, (cx, cy, cz)):
                                        # Move it
                                        placements[box_idx] = (max_bin, cx, cy, cz, w, h, d)
                                        bin_dict[bin_idx].remove(box_idx)
                                        if not bin_dict[bin_idx]:
                                            del bin_dict[bin_idx]
                                            # If we emptied a bin, B decreased. 
                                            # The new max_bin might be different.
                                            # We should restart the outer loop.
                                            improved = True
                                            break
                                        bin_dict[max_bin].append(box_idx)
                                        # If we didn't empty the source bin, B is same.
                                        # Fill of max_bin increased. This is an improvement.
                                        # We can continue trying to move more boxes.
                                        improved = True
                                        break
                        if improved:
                            break
                    if improved:
                        # If B decreased, max_bin changed, restart
                        break
            
            # Final renumbering is done outside

        return placements

    best_result = None
    best_score = float('inf')

    # Run the greedy pack
    placements, bins = greedy_pack(vol_sorted)
    
    if not validate_placement(placements):
        # Fallback
        return [(i, 0, 0, 0, w, h, d) for i, (w, h, d) in enumerate(boxes)]

    # Run local search
    if time.time() < deadline - 10:
        improved_placements = local_search(placements, bins)
        if validate_placement(improved_placements):
            placements = improved_placements

    placements = renumber_bins(placements)
    score = compute_score(placements)

    return placements
