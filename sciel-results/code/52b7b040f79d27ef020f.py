import time
import math
import random
from collections import defaultdict
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    W, H, D = container
    n = len(boxes)

    if n == 0:
        return []

    box_volumes = [w * h * d for w, h, d in boxes]
    total_volume = sum(box_volumes)
    container_vol = W * H * D

    # Sort indices by various heuristics
    vol_sorted = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    max_dim_sorted = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    sum_dim_sorted = sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True)
    min_dim_sorted = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)
    aspect_sorted = sorted(range(n), key=lambda i: max(boxes[i]) / max(min(boxes[i]), 1), reverse=True)

    orderings = [vol_sorted, max_dim_sorted, sum_dim_sorted, min_dim_sorted, aspect_sorted]
    for _ in range(5):
        random_order = list(range(n))
        random.shuffle(random_order)
        orderings.append(random_order)

    def compute_score(placements):
        if len(placements) != n:
            return float('inf')
        bin_labels = set(p[0] for p in placements)
        B = len(bin_labels)
        max_bin = max(p[0] for p in placements)
        last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == max_bin)
        occupied_frac = last_bin_vol / container_vol if container_vol > 0 else 0.0
        return B + 0.1 * (1 - occupied_frac)

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

    class MEP:
        """Maximal Empty Prism representation for a bin."""
        def __init__(self, x1, y1, z1, x2, y2, z2):
            self.x1 = x1
            self.y1 = y1
            self.z1 = z1
            self.x2 = x2
            self.y2 = y2
            self.z2 = z2

        def volume(self):
            return (self.x2 - self.x1) * (self.y2 - self.y1) * (self.z2 - self.z1)

        def contains(self, x, y, z, w, h, d):
            return (x >= self.x1 and y >= self.y1 and z >= self.z1 and
                    x + w <= self.x2 and y + h <= self.y2 and z + d <= self.z2)

        def __repr__(self):
            return f"MEP({self.x1},{self.y1},{self.z1},{self.x2},{self.y2},{self.z2})"

    def split_mep(mep, x, y, z, w, h, d):
        """Split an MEP around a placed box, returning up to 6 sub-MEPs."""
        new_meps = []
        
        # Split along x-axis (left and right of box)
        if mep.x1 < x:
            new_meps.append(MEP(mep.x1, mep.y1, mep.z1, x, mep.y2, mep.z2))
        if x + w < mep.x2:
            new_meps.append(MEP(x + w, mep.y1, mep.z1, mep.x2, mep.y2, mep.z2))
        
        # Split along y-axis (below and above box)
        if mep.y1 < y:
            new_meps.append(MEP(mep.x1, mep.y1, mep.z1, mep.x2, y, mep.z2))
        if y + h < mep.y2:
            new_meps.append(MEP(mep.x1, y + h, mep.z1, mep.x2, mep.y2, mep.z2))
        
        # Split along z-axis (in front of and behind box)
        if mep.z1 < z:
            new_meps.append(MEP(mep.x1, mep.y1, mep.z1, mep.x2, mep.y2, z))
        if z + d < mep.z2:
            new_meps.append(MEP(mep.x1, mep.y1, z + d, mep.x2, mep.y2, mep.z2))
        
        return new_meps

    def remove_contained_meps(meps, box_x, box_y, box_z, box_w, box_h, box_d):
        """Remove MEPs that are fully contained within the placed box."""
        result = []
        for m in meps:
            if (m.x1 >= box_x and m.y1 >= box_y and m.z1 >= box_z and
                m.x2 <= box_x + box_w and m.y2 <= box_y + box_h and m.z2 <= box_z + box_d):
                continue
            result.append(m)
        return result

    def merge_overlapping_meps(meps):
        """Merge overlapping MEPs to keep the set minimal."""
        if len(meps) <= 1:
            return meps
        
        merged = []
        used = [False] * len(meps)
        
        for i in range(len(meps)):
            if used[i]:
                continue
            
            current = meps[i]
            changed = True
            
            while changed:
                changed = False
                for j in range(len(meps)):
                    if j == i or used[j]:
                        continue
                    
                    other = meps[j]
                    
                    # Check if one contains the other
                    if (current.x1 <= other.x1 and current.y1 <= other.y1 and current.z1 <= other.z1 and
                        current.x2 >= other.x2 and current.y2 >= other.y2 and current.z2 >= other.z2):
                        used[j] = True
                        changed = True
                    elif (other.x1 <= current.x1 and other.y1 <= current.y1 and other.z1 <= current.z1 and
                          other.x2 >= current.x2 and other.y2 >= current.y2 and other.z2 >= current.z2):
                        current = other
                        used[j] = True
                        changed = True
            
            merged.append(current)
        
        return merged

    def greedy_place_mep(ordering):
        """MEP-based greedy packing."""
        placements = [None] * n
        bins = []  # Each bin is a list of MEPs
        bin_volumes = []

        for box_idx in ordering:
            w, h, d = boxes[box_idx]
            placed = False

            if bins:
                # Try bins with most remaining free space first (to pack tightly)
                # Actually, try bins with least remaining free space to fill them up
                bin_indices = sorted(range(len(bins)), key=lambda bi: bin_volumes[bi], reverse=True)
            else:
                bin_indices = []

            for bin_i in bin_indices:
                meps = bins[bin_i]
                
                # Find the best MEP to place the box in
                best_mep_idx = None
                best_remaining_vol = float('inf')
                best_pos = None

                for mep_idx, mep in enumerate(meps):
                    if not mep.contains(0, 0, 0, w, h, d):
                        # Check if box can fit in this MEP
                        if w > mep.x2 - mep.x1 or h > mep.y2 - mep.y1 or d > mep.z2 - mep.z1:
                            continue

                    # Try to place box at the "bottom-left-front" of this MEP
                    x = mep.x1
                    y = mep.y1
                    z = mep.z1

                    # Verify no overlap with existing boxes (should be guaranteed by MEP structure)
                    # But let's be safe
                    valid = True
                    for (bx, by, bz, bw, bh, bd) in [p for p in placements if p is not None and p[0] == bin_i]:
                        if (x < bx + bw and bx < x + w and
                            y < by + bh and by < y + h and
                            z < bz + bd and bz < z + d):
                            valid = False
                            break
                    
                    if not valid:
                        continue

                    # Calculate remaining volume after placing box here
                    new_meps = split_mep(mep, x, y, z, w, h, d)
                    # Remove MEPs contained in the box
                    new_meps = remove_contained_meps(new_meps, x, y, z, w, h, d)
                    remaining_vol = sum(m.volume() for m in new_meps)

                    if remaining_vol < best_remaining_vol:
                        best_remaining_vol = remaining_vol
                        best_mep_idx = mep_idx
                        best_pos = (x, y, z)

                if best_mep_idx is not None:
                    x, y, z = best_pos
                    mep = bins[bin_i][best_mep_idx]
                    
                    # Update MEPs for this bin
                    new_meps = split_mep(mep, x, y, z, w, h, d)
                    new_meps = remove_contained_meps(new_meps, x, y, z, w, h, d)
                    
                    # Remove the old MEP and add new ones
                    bins[bin_i].pop(best_mep_idx)
                    bins[bin_i].extend(new_meps)
                    
                    # Merge overlapping MEPs to keep set small
                    bins[bin_i] = merge_overlapping_meps(bins[bin_i])
                    
                    bin_volumes[bin_i] += w * h * d
                    placements[box_idx] = (bin_i, x, y, z, w, h, d)
                    placed = True
                    break

            if not placed:
                new_bin_idx = len(bins)
                # Create new bin with one MEP that is the entire container
                bins.append([MEP(0, 0, 0, W, H, D)])
                bin_volumes.append(w * h * d)
                placements[box_idx] = (new_bin_idx, 0, 0, 0, w, h, d)

        return placements

    def greedy_place_corner(ordering):
        """Corner-based greedy: always try to place at (0,0,0) first, then along edges."""
        placements = [None] * n
        bins = []
        bin_volumes = []
        bin_boxes = []  # Store boxes per bin for overlap checking

        def find_position_corner(w, h, d, bin_boxes):
            if not bin_boxes:
                return (0, 0, 0)

            # Generate candidate positions
            candidates = [(0, 0, 0)]
            
            # Add positions along edges
            for bx, by, bz, bw, bh, bd in bin_boxes:
                # Along x-axis
                if bx + bw + w <= W:
                    candidates.append((bx + bw, 0, 0))
                # Along y-axis
                if by + bh + h <= H:
                    candidates.append((0, by + bh, 0))
                # Along z-axis
                if bz + bd + d <= D:
                    candidates.append((0, 0, bz + bd))
                # Corners
                if bx + bw + w <= W and by + bh + h <= H:
                    candidates.append((bx + bw, by + bh, 0))
                if bx + bw + w <= W and bz + bd + d <= D:
                    candidates.append((bx + bw, 0, bz + bd))
                if by + bh + h <= H and bz + bd + d <= D:
                    candidates.append((0, by + bh, bz + bd))
                # All corners
                if bx + bw + w <= W and by + bh + h <= H and bz + bd + d <= D:
                    candidates.append((bx + bw, by + bh, bz + bd))

            # Remove duplicates and sort by priority (z, y, x)
            unique_candidates = []
            seen = set()
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    unique_candidates.append(c)
            
            unique_candidates.sort(key=lambda p: (p[2], p[1], p[0]))

            for x, y, z in unique_candidates:
                if x + w > W or y + h > H or z + d > D:
                    continue
                
                # Check overlap
                overlaps = False
                for bx, by, bz, bw, bh, bd in bin_boxes:
                    if (x < bx + bw and bx < x + w and
                        y < by + bh and by < y + h and
                        z < bz + bd and bz < z + d):
                        overlaps = True
                        break
                
                if not overlaps:
                    return (x, y, z)

            return None

        for box_idx in ordering:
            w, h, d = boxes[box_idx]
            placed = False

            if bins:
                bin_indices = sorted(range(len(bins)), key=lambda bi: bin_volumes[bi], reverse=True)
            else:
                bin_indices = []

            for bin_i in bin_indices:
                pos = find_position_corner(w, h, d, bin_boxes[bin_i])
                if pos is not None:
                    x, y, z = pos
                    bin_boxes[bin_i].append((x, y, z, w, h, d))
                    bin_volumes[bin_i] += w * h * d
                    placements[box_idx] = (bin_i, x, y, z, w, h, d)
                    placed = True
                    break

            if not placed:
                new_bin_idx = len(bins)
                bins.append([])
                bin_volumes.append(w * h * d)
                bin_boxes.append([(0, 0, 0, w, h, d)])
                placements[box_idx] = (new_bin_idx, 0, 0, 0, w, h, d)

        return placements

    def local_improve(placements):
        """Try to improve placement by moving boxes between bins."""
        if time.time() > deadline:
            return placements

        n = len(placements)
        bins = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            bins[b].append(i)

        bin_list = list(bins.keys())
        bin_volumes = {b: sum(boxes[i][0] * boxes[i][1] * boxes[i][2] for i in bins[b]) for b in bin_list}

        improved = True
        max_iterations = 50
        iteration = 0

        while improved and iteration < max_iterations and time.time() < deadline:
            improved = False
            iteration += 1

            for box_idx in range(n):
                b, x, y, z, w, h, d = placements[box_idx]
                w, h, d = boxes[box_idx]

                # Try to move this box to another bin
                for target_bin in bin_list:
                    if target_bin == b:
                        continue

                    # Try to place box in target bin
                    target_boxes = [(placements[i][1], placements[i][2], placements[i][3],
                                    boxes[i][0], boxes[i][1], boxes[i][2])
                                   for i in bins[target_bin] if i != box_idx]

                    # Find position
                    candidates = [(0, 0, 0)]
                    for bx, by, bz, bw, bh, bd in target_boxes:
                        if bx + bw + w <= W:
                            candidates.append((bx + bw, 0, 0))
                        if by + bh + h <= H:
                            candidates.append((0, by + bh, 0))
                        if bz + bd + d <= D:
                            candidates.append((0, 0, bz + bd))
                        if bx + bw + w <= W and by + bh + h <= H:
                            candidates.append((bx + bw, by + bh, 0))
                        if bx + bw + w <= W and bz + bd + d <= D:
                            candidates.append((bx + bw, 0, bz + bd))
                        if by + bh + h <= H and bz + bd + d <= D:
                            candidates.append((0, by + bh, bz + bd))
                        if bx + bw + w <= W and by + bh + h <= H and bz + bd + d <= D:
                            candidates.append((bx + bw, by + bh, bz + bd))

                    unique_candidates = []
                    seen = set()
                    for c in candidates:
                        if c not in seen:
                            seen.add(c)
                            unique_candidates.append(c)
                    unique_candidates.sort(key=lambda p: (p[2], p[1], p[0]))

                    for tx, ty, tz in unique_candidates:
                        if tx + w > W or ty + h > H or tz + d > D:
                            continue

                        overlaps = False
                        for bx, by, bz, bw, bh, bd in target_boxes:
                            if (tx < bx + bw and bx < tx + w and
                                ty < by + bh and by < ty + h and
                                tz < bz + bd and bz < tz + d):
                                overlaps = True
                                break

                        if not overlaps:
                            # Check if this move improves the score
                            old_score = compute_score(placements)

                            # Temporarily move
                            old_placement = placements[box_idx]
                            placements[box_idx] = (target_bin, tx, ty, tz, w, h, d)
                            bins[b].remove(box_idx)
                            bins[target_bin].append(box_idx)
                            bin_volumes[b] -= w * h * d
                            bin_volumes[target_bin] += w * h * d

                            new_score = compute_score(placements)

                            if new_score < old_score - 1e-9:
                                improved = True
                            else:
                                # Revert
                                placements[box_idx] = old_placement
                                bins[b].append(box_idx)
                                bins[target_bin].remove(box_idx)
                                bin_volumes[b] += w * h * d
                                bin_volumes[target_bin] -= w * h * d

                            break

                    if improved:
                        break

        return placements

    best_result = None
    best_score = float('inf')

    strategies = [greedy_place_mep, greedy_place_corner]

    for strategy in strategies:
        for order in orderings:
            if time.time() > deadline:
                break

            try:
                placements = strategy(order)

                if not validate_placement(placements):
                    continue

                # Apply local improvement
                placements = local_improve(placements)

                if not validate_placement(placements):
                    continue

                score = compute_score(placements)

                if score < best_score:
                    best_score = score
                    best_result = placements[:]
            except Exception:
                continue

    if best_result is not None:
        return best_result

    # Fallback: one box per bin
    return [(i, 0, 0, 0, w, h, d) for i, (w, h, d) in enumerate(boxes)]
