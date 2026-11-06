"""3D Bin Packing with maximal free-space partitioning and local search."""
import time
import random
import math
from typing import List, Tuple
from collections import defaultdict


def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    deadline = time.time() + 115.0
    container_vol = W * H * D

    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]

    # Sort indices by volume descending
    indices = list(range(n))
    indices_sorted = sorted(indices, key=lambda i: box_vols[i], reverse=True)

    class FreeSpace:
        """Represents an axis-aligned empty cuboid."""
        __slots__ = ['x', 'y', 'z', 'w', 'h', 'd']

        def __init__(self, x, y, z, w, h, d):
            self.x = x
            self.y = y
            self.z = z
            self.w = w
            self.h = h
            self.d = d

        def contains(self, px, py, pz, pw, ph, pd):
            return (px >= self.x and py >= self.y and pz >= self.z and
                    px + pw <= self.x + self.w and
                    py + ph <= self.y + self.h and
                    pz + pd <= self.z + self.d)

        def key(self):
            return (self.x, self.y, self.z, self.w, self.h, self.d)

    def split_free_spaces(free_spaces, x, y, z, w, h, d):
        """
        Split free spaces that intersect with the placed box.
        Returns new list of free spaces after splitting.
        """
        new_spaces = []
        for fs in free_spaces:
            # Check if the box intersects this free space
            if not (x < fs.x + fs.w and fs.x < x + w and
                    y < fs.y + fs.h and fs.y < y + h and
                    z < fs.z + fs.d and fs.z < z + d):
                # No intersection, keep as is
                new_spaces.append(fs)
                continue

            # Split the free space into up to 6 parts
            # Left part
            if x > fs.x:
                new_spaces.append(FreeSpace(fs.x, fs.y, fs.z, x - fs.x, fs.h, fs.d))
            # Right part
            if x + w < fs.x + fs.w:
                new_spaces.append(FreeSpace(x + w, fs.y, fs.z, fs.x + fs.w - x - w, fs.h, fs.d))
            # Front part (in y)
            if y > fs.y:
                new_spaces.append(FreeSpace(fs.x, fs.y, fs.z, fs.w, y - fs.y, fs.d))
            # Back part (in y)
            if y + h < fs.y + fs.h:
                new_spaces.append(FreeSpace(fs.x, y + h, fs.z, fs.w, fs.y + fs.h - y - h, fs.d))
            # Bottom part (in z)
            if z > fs.z:
                new_spaces.append(FreeSpace(fs.x, fs.y, fs.z, fs.w, fs.h, z - fs.z))
            # Top part (in z)
            if z + d < fs.z + fs.d:
                new_spaces.append(FreeSpace(fs.x, fs.y, z + d, fs.w, fs.h, fs.z + fs.d - z - d))

        # Remove duplicates and zero-volume spaces
        seen = set()
        unique_spaces = []
        for fs in new_spaces:
            if fs.w <= 0 or fs.h <= 0 or fs.d <= 0:
                continue
            k = fs.key()
            if k not in seen:
                seen.add(k)
                unique_spaces.append(fs)
        return unique_spaces

    def find_best_position(free_spaces, w, h, d):
        """
        Find the best position (x, y, z) where box (w, h, d) fits.
        Uses free space list to check feasibility.
        Returns (x, y, z) or None.
        """
        if not free_spaces:
            return None

        # Find all free spaces that can contain the box
        valid_positions = []
        for fs in free_spaces:
            if fs.w >= w and fs.h >= h and fs.d >= d:
                # The box can be placed anywhere within this free space
                # We prefer positions closer to the origin for compactness
                # Try placing at the corner of the free space closest to origin
                valid_positions.append((fs.x, fs.y, fs.z, fs))

        if not valid_positions:
            return None

        # Sort by distance to origin
        valid_positions.sort(key=lambda v: v[0] + v[1] + v[2])

        # Try each candidate position
        for (px, py, pz, fs) in valid_positions:
            # Check if the box fits in this free space at this position
            # Since we're placing at the corner, it should fit
            # But we need to verify it doesn't overlap with other free spaces' boundaries
            # Actually, since free spaces are disjoint, if the box fits in one free space,
            # it won't overlap with any placed box
            return (px, py, pz)

        return None

    def pack_with_order(ordering):
        """Pack boxes in given order. Returns placements list or None."""
        # State: bin_id -> list of free spaces
        bin_free_spaces = {}
        placements = [None] * n
        all_placed = True

        for box_idx in ordering:
            if time.time() > deadline:
                all_placed = False
                break

            w, h, d = boxes[box_idx]

            # Try existing bins
            best_bin = None
            best_pos = None
            best_score = float('inf')

            # Sort bins by current fill volume descending (more filled bins first)
            bin_ids = list(bin_free_spaces.keys())
            # Estimate fill: container_vol - sum of free space volumes
            bin_fill = {}
            for bid in bin_ids:
                free_vol = sum(fs.w * fs.h * fs.d for fs in bin_free_spaces[bid])
                bin_fill[bid] = container_vol - free_vol
            bin_ids_sorted = sorted(bin_ids, key=lambda bid: bin_fill[bid], reverse=True)

            for bin_id in bin_ids_sorted:
                pos = find_best_position(bin_free_spaces[bin_id], w, h, d)
                if pos is not None:
                    # Score: prefer positions closer to origin
                    score = pos[0] + pos[1] + pos[2]
                    if score < best_score:
                        best_score = score
                        best_bin = bin_id
                        best_pos = pos

            if best_bin is not None:
                # Place the box and update free spaces
                new_free_spaces = split_free_spaces(bin_free_spaces[best_bin],
                                                    best_pos[0], best_pos[1], best_pos[2],
                                                    w, h, d)
                bin_free_spaces[best_bin] = new_free_spaces
                placements[box_idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)
            else:
                # Open new bin
                new_bin_id = max(bin_free_spaces.keys()) + 1 if bin_free_spaces else 0
                bin_free_spaces[new_bin_id] = [FreeSpace(0, 0, 0, W, H, D)]
                # Place at origin
                new_free_spaces = split_free_spaces(bin_free_spaces[new_bin_id], 0, 0, 0, w, h, d)
                bin_free_spaces[new_bin_id] = new_free_spaces
                placements[box_idx] = (new_bin_id, 0, 0, 0, w, h, d)

        if all_placed and all(p is not None for p in placements):
            return placements
        return None

    def compute_score(placements):
        if not placements:
            return 0.0
        bins_used = set(p[0] for p in placements)
        B = len(bins_used)
        last_bin = max(bins_used)
        last_bin_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == last_bin)
        return B + 0.1 * (1 - last_bin_vol / container_vol)

    def local_search_polish(placements, max_iterations=100):
        """
        Local search: try to move boxes to other bins to reduce total bins used.
        """
        if not placements:
            return placements

        bin_boxes = defaultdict(list)
        for i, (b, x, y, z, w, h, d) in enumerate(placements):
            bin_boxes[b].append(i)

        current_score = compute_score(placements)
        improved = True
        iterations = 0

        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1

            for box_idx in list(range(n)):
                if time.time() > deadline - 5:
                    break
                b, x, y, z, w, h, d = placements[box_idx]
                other_bins = [bid for bid in bin_boxes if bid != b]
                for target_bin in other_bins:
                    # Build free spaces for target bin from its current boxes
                    target_items = [(placements[j][1], placements[j][2], placements[j][3],
                                     placements[j][4], placements[j][5], placements[j][6])
                                    for j in bin_boxes[target_bin]]
                    # Rebuild free spaces for target bin
                    target_free = [FreeSpace(0, 0, 0, W, H, D)]
                    for (ex, ey, ez, ew, eh, ed) in target_items:
                        target_free = split_free_spaces(target_free, ex, ey, ez, ew, eh, ed)

                    pos = find_best_position(target_free, w, h, d)
                    if pos is not None:
                        new_placements = list(placements)
                        new_placements[box_idx] = (target_bin, pos[0], pos[1], pos[2], w, h, d)
                        new_score = compute_score(new_placements)
                        if new_score < current_score:
                            placements = new_placements
                            current_score = new_score
                            bin_boxes[b].remove(box_idx)
                            if not bin_boxes[b]:
                                del bin_boxes[b]
                            bin_boxes[target_bin].append(box_idx)
                            improved = True
                            break

        return placements

    # Generate diverse orderings
    orderings = []
    orderings.append(indices_sorted)
    orderings.append(list(reversed(indices_sorted)))
    orderings.append(indices)
    orderings.append(sorted(indices, key=lambda i: max(boxes[i]), reverse=True))
    orderings.append(sorted(indices, key=lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2], reverse=True))
    orderings.append(sorted(indices, key=lambda i: min(boxes[i]), reverse=True))
    orderings.append(sorted(indices, key=lambda i: math.sqrt(box_vols[i]), reverse=True))

    # Run all orderings and keep the best
    best_result = None
    best_score = float('inf')

    for ordering in orderings:
        if time.time() > deadline:
            break
        result = pack_with_order(ordering)
        if result is not None:
            score = compute_score(result)
            if score < best_score:
                best_score = score
                best_result = result

    # Fallback
    if best_result is None:
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_result = placements

    # Local search polish
    if best_result and time.time() < deadline - 10:
        best_result = local_search_polish(best_result)

    # Relabel bins to be consecutive
    if best_result:
        bin_labels = sorted(set(p[0] for p in best_result))
        label_map = {old: new for new, old in enumerate(bin_labels)}
        best_result = [(label_map[p[0]], p[1], p[2], p[3], p[4], p[5], p[6]) for p in best_result]

    return best_result
