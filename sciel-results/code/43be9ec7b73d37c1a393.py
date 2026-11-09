import time
import math
import random
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START
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

    orderings = [vol_sorted, max_dim_sorted, sum_dim_sorted, min_dim_sorted]
    for _ in range(6):
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

    def greedy_place(ordering):
        placements = [None] * n
        bins = []  # list of lists of (x, y, z, w, h, d)
        bin_volumes = []  # track volume used in each bin

        def find_position_in_bin(w, h, d, bin_boxes):
            if not bin_boxes:
                return (0, 0, 0)

            # Generate candidate positions from box edges
            x_cands = set([0])
            y_cands = set([0])
            z_cands = set([0])

            for bx, by, bz, bw, bh, bd in bin_boxes:
                x_cands.add(bx)
                x_cands.add(bx + bw)
                y_cands.add(by)
                y_cands.add(by + bh)
                z_cands.add(bz)
                z_cands.add(bz + bd)

            # Filter valid candidates
            x_cands = [x for x in x_cands if 0 <= x and x + w <= W]
            y_cands = [y for y in y_cands if 0 <= y and y + h <= H]
            z_cands = [z for z in z_cands if 0 <= z and z + d <= D]

            if not x_cands or not y_cands or not z_cands:
                return None

            best_pos = None
            best_cost = float('inf')

            # Use bottom-left-far heuristic: minimize z, then y, then x
            # But also consider "best fit" - minimize the maximum coordinate sum
            for z in z_cands:
                for y in y_cands:
                    for x in x_cands:
                        # Check overlap with all boxes in bin
                        overlaps = False
                        for bx, by, bz, bw, bh, bd in bin_boxes:
                            if (x < bx + bw and bx < x + w and
                                y < by + bh and by < y + h and
                                z < bz + bd and bz < z + d):
                                overlaps = True
                                break
                        if overlaps:
                            continue

                        # Cost: prefer positions that are "tucked in" - low z, low y, low x
                        # But also penalize creating gaps - use a combined cost
                        cost = z * (H + 1) * (W + 1) + y * (W + 1) + x

                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (x, y, z)
                            if best_pos == (0, 0, 0):
                                return best_pos

            return best_pos

        for box_idx in ordering:
            w, h, d = boxes[box_idx]

            # Try to place in existing bins
            placed = False

            # Sort bins by volume used (descending) to try fullest bins first
            # This helps pack bins tightly
            if bins:
                bin_indices = sorted(range(len(bins)), key=lambda bi: bin_volumes[bi], reverse=True)
            else:
                bin_indices = []

            for bin_i in bin_indices:
                pos = find_position_in_bin(w, h, d, bins[bin_i])
                if pos is not None:
                    x, y, z = pos
                    bins[bin_i].append((x, y, z, w, h, d))
                    bin_volumes[bin_i] += w * h * d
                    placements[box_idx] = (bin_i, x, y, z, w, h, d)
                    placed = True
                    break

            if not placed:
                new_bin_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                bin_volumes.append(w * h * d)
                placements[box_idx] = (new_bin_idx, 0, 0, 0, w, h, d)

        return placements

    def greedy_place_best_fit(ordering):
        """Variant that uses best-fit: minimize the gap created."""
        placements = [None] * n
        bins = []
        bin_volumes = []

        def find_position_best_fit(w, h, d, bin_boxes):
            if not bin_boxes:
                return (0, 0, 0)

            x_cands = set([0])
            y_cands = set([0])
            z_cands = set([0])

            for bx, by, bz, bw, bh, bd in bin_boxes:
                x_cands.add(bx)
                x_cands.add(bx + bw)
                y_cands.add(by)
                y_cands.add(by + bh)
                z_cands.add(bz)
                z_cands.add(bz + bd)

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
                        overlaps = False
                        for bx, by, bz, bw, bh, bd in bin_boxes:
                            if (x < bx + bw and bx < x + w and
                                y < by + bh and by < y + h and
                                z < bz + bd and bz < z + d):
                                overlaps = True
                                break
                        if overlaps:
                            continue

                        # Best fit: minimize the "tightness" - how close to other boxes
                        # Use sum of distances to nearest box in each direction
                        # Simpler: minimize z + y + x (tuck into corner)
                        # But weight z more heavily to fill from bottom
                        cost = z * 100 + y * 10 + x

                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (x, y, z)
                            if best_pos == (0, 0, 0):
                                return best_pos

            return best_pos

        for box_idx in ordering:
            w, h, d = boxes[box_idx]
            placed = False

            if bins:
                # Try bins with most empty space first (least full)
                # to balance load, or try fullest first to pack tightly
                bin_indices = sorted(range(len(bins)), key=lambda bi: bin_volumes[bi], reverse=True)
            else:
                bin_indices = []

            for bin_i in bin_indices:
                pos = find_position_best_fit(w, h, d, bins[bin_i])
                if pos is not None:
                    x, y, z = pos
                    bins[bin_i].append((x, y, z, w, h, d))
                    bin_volumes[bin_i] += w * h * d
                    placements[box_idx] = (bin_i, x, y, z, w, h, d)
                    placed = True
                    break

            if not placed:
                new_bin_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                bin_volumes.append(w * h * d)
                placements[box_idx] = (new_bin_idx, 0, 0, 0, w, h, d)

        return placements

    def greedy_place_with_shelf(ordering):
        """Shelf-based packing for each bin."""
        placements = [None] * n
        bins = []
        bin_volumes = []

        class Bin:
            def __init__(self):
                self.shelves = []  # list of (z, y, x_start, x_end)
                self.volume = 0
                self.boxes = []

            def add_shelf(self, z, y, x_start, x_end):
                self.shelves.append([z, y, x_start, x_end])

            def find_position(self, w, h, d):
                # Try to fit on existing shelves
                for shelf in self.shelves:
                    z, y, x_start, x_end = shelf
                    if z + d <= D and y + h <= H:
                        # Check if box fits on this shelf
                        if x_start + w <= x_end:
                            # Check for overlap with boxes on this shelf
                            overlaps = False
                            for bx, by, bz, bw, bh, bd in self.boxes:
                                if bz == z and by == y:
                                    if x_start < bx + bw and bx < x_start + w:
                                        overlaps = True
                                        break
                            if not overlaps:
                                return (x_start, y, z)
                        # Try to extend shelf
                        if x_start + w <= W:
                            overlaps = False
                            for bx, by, bz, bw, bh, bd in self.boxes:
                                if bz == z and by == y:
                                    if x_start < bx + bw and bx < x_start + w:
                                        overlaps = True
                                        break
                            if not overlaps:
                                return (x_start, y, z)

                # Create new shelf
                # Find lowest z where we can place a new shelf
                z = 0
                while z + d <= D:
                    # Check if there's space at this z level
                    can_place = True
                    for bx, by, bz, bw, bh, bd in self.boxes:
                        if z < bz + bd and bz < z + d:
                            # Check if y range overlaps
                            if 0 < by + bh and by < H:
                                # There's a box at this z level, need to check y
                                pass

                    # For simplicity, try y=0 first
                    for y in [0] + [by + bh for by, _, _, _, bh, _ in self.boxes]:
                        if y + h > H:
                            continue
                        # Check overlap
                        overlaps = False
                        for bx, by2, bz, bw, bh, bd in self.boxes:
                            if (0 < bx + bw and bx < w and
                                y < by2 + bh and by2 < y + h and
                                z < bz + bd and bz < z + d):
                                overlaps = True
                                break
                        if not overlaps:
                            return (0, y, z)

                    z += 1

                return None

        for box_idx in ordering:
            w, h, d = boxes[box_idx]
            placed = False

            if bins:
                bin_indices = sorted(range(len(bins)), key=lambda bi: bins[bi].volume, reverse=True)
            else:
                bin_indices = []

            for bin_i in bin_indices:
                pos = bins[bin_i].find_position(w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[bin_i].boxes.append((x, y, z, w, h, d))
                    bins[bin_i].volume += w * h * d
                    placements[box_idx] = (bin_i, x, y, z, w, h, d)
                    placed = True
                    break

            if not placed:
                new_bin_idx = len(bins)
                new_bin = Bin()
                new_bin.boxes.append((0, 0, 0, w, h, d))
                new_bin.volume = w * h * d
                bins.append(new_bin)
                placements[box_idx] = (new_bin_idx, 0, 0, 0, w, h, d)

        return placements

    best_result = None
    best_score = float('inf')

    strategies = [greedy_place, greedy_place_best_fit, greedy_place_with_shelf]

    for strategy in strategies:
        for order in orderings:
            if time.time() > deadline:
                break

            try:
                placements = strategy(order)

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

# EVOLVE-BLOCK-END
