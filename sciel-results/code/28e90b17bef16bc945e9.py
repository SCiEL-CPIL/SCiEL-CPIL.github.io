"""Rewritten 3D bin packing with BSP tree for fast placement queries."""
import time
import random
from collections import defaultdict


def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    n = len(boxes)
    container_vol = W * H * D
    if container_vol == 0:
        return [(0, 0, 0, w, h, d) for w, h, d in boxes]

    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]

    def is_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)

    class EmptySpace:
        """Represents an empty rectangular region in a bin."""
        __slots__ = ['x', 'y', 'z', 'w', 'h', 'd']

        def __init__(self, x, y, z, w, h, d):
            self.x = x
            self.y = y
            self.z = z
            self.w = w
            self.h = h
            self.d = d

        def volume(self):
            return self.w * self.h * self.d

        def contains(self, x, y, z, w, h, d):
            return (self.x <= x and self.y <= y and self.z <= z and
                    x + w <= self.x + self.w and
                    y + h <= self.y + self.h and
                    z + d <= self.z + self.d)

        def overlaps(self, x, y, z, w, h, d):
            return not (x + w <= self.x or self.x + self.w <= x or
                        y + h <= self.y or self.y + self.h <= y or
                        z + d <= self.z or self.z + self.d <= z)

        def split(self, x, y, z, w, h, d):
            """Split this empty space after placing a box at (x,y,z) with dims (w,h,d).
            Returns list of new empty spaces."""
            result = []
            # Split along x-axis
            if x > self.x:
                result.append(EmptySpace(self.x, self.y, self.z, x - self.x, self.h, self.d))
            if x + w < self.x + self.w:
                result.append(EmptySpace(x + w, self.y, self.z, self.x + self.w - x - w, self.h, self.d))
            # Split along y-axis (within the middle x-slab)
            mid_x_start = max(self.x, x)
            mid_x_end = min(self.x + self.w, x + w)
            mid_w = mid_x_end - mid_x_start
            if mid_w > 0:
                if y > self.y:
                    result.append(EmptySpace(mid_x_start, self.y, self.z, mid_w, y - self.y, self.d))
                if y + h < self.y + self.h:
                    result.append(EmptySpace(mid_x_start, y + h, self.z, mid_w, self.y + self.h - y - h, self.d))
                # Split along z-axis (within the middle y-slab)
                mid_y_start = max(self.y, y)
                mid_y_end = min(self.y + self.h, y + h)
                mid_h = mid_y_end - mid_y_start
                if mid_h > 0:
                    if z > self.z:
                        result.append(EmptySpace(mid_x_start, mid_y_start, self.z, mid_w, mid_h, z - self.z))
                    if z + d < self.z + self.d:
                        result.append(EmptySpace(mid_x_start, mid_y_start, z + d, mid_w, mid_h, self.z + self.d - z - d))

            # Filter out zero-volume spaces
            return [s for s in result if s.volume() > 0]

    class Bin:
        __slots__ = ['boxes', 'volume', 'empty_spaces']

        def __init__(self):
            self.boxes = []  # list of (idx, x, y, z, w, h, d)
            self.volume = 0
            self.empty_spaces = [EmptySpace(0, 0, 0, W, H, D)]

        def can_place(self, x, y, z, w, h, d):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            # Check against existing boxes
            for (_, bx, by, bz, bw, bh, bd) in self.boxes:
                if is_overlap(x, y, z, w, h, d, bx, by, bz, bw, bh, bd):
                    return False
            return True

        def find_position(self, w, h, d):
            """Find a valid position using empty spaces. Returns (x,y,z) or None."""
            # Sort empty spaces by volume descending (prefer larger spaces)
            # Actually, we want to find the "best" position - let's use bottom-left-front priority
            best_pos = None
            best_metric = float('inf')

            for space in self.empty_spaces:
                if space.w < w or space.h < h or space.d < d:
                    continue
                # Try placing at the bottom-left-front of this space
                px, py, pz = space.x, space.y, space.z
                if self.can_place(px, py, pz, w, h, d):
                    metric = px + py + pz
                    if metric < best_metric:
                        best_metric = metric
                        best_pos = (px, py, pz)

            if best_pos is not None:
                return best_pos

            # If no empty space works, try corner points from existing boxes
            cands = set([(0, 0, 0)])
            for (_, bx, by, bz, bw, bh, bd) in self.boxes:
                for dx in (0, bw):
                    for dy in (0, bh):
                        for dz in (0, bd):
                            cands.add((bx + dx, by + dy, bz + dz))

            for (cx, cy, cz) in cands:
                if self.can_place(cx, cy, cz, w, h, d):
                    return (cx, cy, cz)

            return None

        def place(self, idx, x, y, z, w, h, d):
            self.boxes.append((idx, x, y, z, w, h, d))
            self.volume += w * h * d
            # Update empty spaces
            new_spaces = []
            for space in self.empty_spaces:
                if not space.overlaps(x, y, z, w, h, d):
                    new_spaces.append(space)
                else:
                    new_spaces.extend(space.split(x, y, z, w, h, d))
            self.empty_spaces = new_spaces

        def remove(self, idx):
            for i, box in enumerate(self.boxes):
                if box[0] == idx:
                    w, h, d = box[4], box[5], box[6]
                    self.volume -= w * h * d
                    self.boxes.pop(i)
                    # Rebuild empty spaces from scratch
                    self._rebuild_empty_spaces()
                    return True
            return False

        def _rebuild_empty_spaces(self):
            """Rebuild empty spaces by starting with the full container and subtracting all boxes."""
            # Start with the full container
            current = [EmptySpace(0, 0, 0, W, H, D)]
            for (_, bx, by, bz, bw, bh, bd) in self.boxes:
                new_spaces = []
                for space in current:
                    if not space.overlaps(bx, by, bz, bw, bh, bd):
                        new_spaces.append(space)
                    else:
                        new_spaces.extend(space.split(bx, by, bz, bw, bh, bd))
                current = new_spaces
            self.empty_spaces = current

        def get_placement(self, idx):
            for box in self.boxes:
                if box[0] == idx:
                    return box[1], box[2], box[3], box[4], box[5], box[6]
            return None

        def is_empty(self):
            return len(self.boxes) == 0

    def compute_score(assignments):
        """Compute score: B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not assignments:
            return float('inf')
        bins_used = set(assignments.values())
        B = len(bins_used)
        last_bin = max(bins_used)
        last_vol = sum(box_vols[idx] for idx, b in assignments.items() if b == last_bin)
        frac = last_vol / container_vol
        return B + 0.1 * (1 - frac)

    def greedy_pack(order, bins, assignments):
        """Place boxes in given order using best-fit heuristic."""
        for idx, (w, h, d) in order:
            if time.time() > deadline:
                break

            placed = False
            if bins:
                # Try all existing bins, pick the one with best position
                best_bin_id = -1
                best_pos = None
                best_metric = float('inf')

                for b_id in range(len(bins)):
                    if bins[b_id].is_empty():
                        continue
                    pos = bins[b_id].find_position(w, h, d)
                    if pos is not None:
                        cx, cy, cz = pos
                        metric = cx + cy + cz
                        if metric < best_metric:
                            best_metric = metric
                            best_bin_id = b_id
                            best_pos = (cx, cy, cz)

                if best_bin_id >= 0:
                    cx, cy, cz = best_pos
                    bins[best_bin_id].place(idx, cx, cy, cz, w, h, d)
                    assignments[idx] = best_bin_id
                    placed = True

            if not placed:
                # Open new bin
                b_id = len(bins)
                bins.append(Bin())
                bins[b_id].place(idx, 0, 0, 0, w, h, d)
                assignments[idx] = b_id

    def local_search(bins, assignments):
        """Local search to minimize B, then maximize last-bin fill."""
        if time.time() > deadline:
            return

        # Phase 1: Minimize number of bins by moving boxes out of higher-numbered bins
        for _pass in range(20):
            if time.time() > deadline:
                break

            occupied_bins = sorted([b for b in range(len(bins)) if not bins[b].is_empty()], reverse=True)
            made_progress = False

            for b_id in occupied_bins:
                if time.time() > deadline:
                    break
                if bins[b_id].is_empty():
                    continue

                boxes_in_bin = list(bins[b_id].boxes)

                for (idx, x, y, z, w, h, d) in boxes_in_bin:
                    if time.time() > deadline:
                        break
                    # Verify box is still in this bin
                    if not any(b[0] == idx for b in bins[b_id].boxes):
                        continue

                    # Try to move to lower-numbered bins
                    moved = False
                    for target_b in range(b_id):
                        if bins[target_b].is_empty():
                            continue
                        pos = bins[target_b].find_position(w, h, d)
                        if pos is not None:
                            bins[b_id].remove(idx)
                            cx, cy, cz = pos
                            bins[target_b].place(idx, cx, cy, cz, w, h, d)
                            assignments[idx] = target_b
                            moved = True
                            made_progress = True
                            break

                    # If not moved to lower bin, try to any other non-empty bin
                    if not moved:
                        for target_b in range(len(bins)):
                            if target_b == b_id or bins[target_b].is_empty():
                                continue
                            pos = bins[target_b].find_position(w, h, d)
                            if pos is not None:
                                bins[b_id].remove(idx)
                                cx, cy, cz = pos
                                bins[target_b].place(idx, cx, cy, cz, w, h, d)
                                assignments[idx] = target_b
                                moved = True
                                made_progress = True
                                break

            if not made_progress:
                break

        # Phase 2: Maximize last-bin fill (minimize the 0.1 penalty term)
        if time.time() < deadline:
            occupied_bins = [b for b in range(len(bins)) if not bins[b].is_empty()]
            if occupied_bins:
                last_bin_id = max(occupied_bins)

                # Try to move boxes from other bins into the last bin to increase its fill
                for b_id in sorted(occupied_bins, reverse=True):
                    if time.time() > deadline:
                        break
                    if b_id == last_bin_id or bins[b_id].is_empty():
                        continue

                    boxes_in_bin = list(bins[b_id].boxes)
                    for (idx, x, y, z, w, h, d) in boxes_in_bin:
                        if time.time() > deadline:
                            break
                        if not any(b[0] == idx for b in bins[b_id].boxes):
                            continue

                        pos = bins[last_bin_id].find_position(w, h, d)
                        if pos is not None:
                            bins[b_id].remove(idx)
                            cx, cy, cz = pos
                            bins[last_bin_id].place(idx, cx, cy, cz, w, h, d)
                            assignments[idx] = last_bin_id

    def build_result(bins, assignments):
        """Construct result list from bins and assignments."""
        result = [None] * n
        for idx in range(n):
            if idx in assignments:
                b_id = assignments[idx]
                pos = bins[b_id].get_placement(idx)
                if pos:
                    x, y, z, w, h, d = pos
                    result[idx] = (b_id, x, y, z, w, h, d)
                else:
                    # Should not happen, but fallback
                    result[idx] = (0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            else:
                # Box wasn't placed (time out during initial packing)
                # Place in a new bin
                b_id = len(bins)
                bins.append(Bin())
                w, h, d = boxes[idx]
                bins[b_id].place(idx, 0, 0, 0, w, h, d)
                assignments[idx] = b_id
                result[idx] = (b_id, 0, 0, 0, w, h, d)

        # Renumber bins to be consecutive starting from 0
        if result:
            bin_labels = sorted(set(r[0] for r in result if r is not None))
            if bin_labels:
                label_map = {old: new for new, old in enumerate(bin_labels)}
                for i in range(n):
                    if result[i] is not None:
                        old_bin = result[i][0]
                        new_bin = label_map[old_bin]
                        result[i] = (new_bin, result[i][1], result[i][2], result[i][3], result[i][4], result[i][5], result[i][6])

        return result

    def validate_result(result):
        """Validate that result is feasible."""
        bins_boxes = {}
        for idx in range(n):
            b, x, y, z, w, h, d = result[idx]
            # Check bounds
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            if b not in bins_boxes:
                bins_boxes[b] = []
            bins_boxes[b].append((x, y, z, w, h, d))

        for b, bl in bins_boxes.items():
            for i in range(len(bl)):
                for j in range(i + 1, len(bl)):
                    if is_overlap(bl[i][0], bl[i][1], bl[i][2], bl[i][3], bl[i][4], bl[i][5],
                                  bl[j][0], bl[j][1], bl[j][2], bl[j][3], bl[j][4], bl[j][5]):
                        return False
        return True

    # --- Multi-strategy greedy packing with randomized restarts ---

    strategies = [
        lambda t: -(t[1][0] * t[1][1] * t[1][2]),  # Volume desc
        lambda t: t[1][0] * t[1][1] * t[1][2],      # Volume asc
        lambda t: -(max(t[1])),                      # Max dim desc
        lambda t: -(t[1][0] + t[1][1] + t[1][2]),   # Sum dim desc
    ]

    best_score = float('inf')
    best_result = None

    # Try multiple strategies and randomized restarts
    max_restarts = 20
    restart_count = 0

    for strat_idx, strat_func in enumerate(strategies):
        if time.time() > deadline:
            break

        # Multiple restarts per strategy with slight randomization
        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            order = list(enumerate(boxes))

            if restart == 0:
                order.sort(key=strat_func)
            else:
                # Add noise to the sorting key for diversity
                order.sort(key=lambda t: strat_func(t) + random.uniform(-0.1, 0.1) * (t[1][0] * t[1][1] * t[1][2] + 1))

            bins = []
            assignments = {}

            greedy_pack(order, bins, assignments)

            # Local search to improve
            local_search(bins, assignments)

            result = build_result(bins, assignments)

            # Validate
            if not validate_result(result):
                continue

            # Compute score
            score = compute_score(assignments)

            if score < best_score:
                best_score = score
                best_result = result

    # If no valid result found, use fallback
    if best_result is None:
        # Simple shelf packing fallback
        placements = {}
        b = 0
        x, y, z = 0, 0, 0
        row_h = 0
        layer_d = 0

        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=lambda t: -(t[1][0] * t[1][1] * t[1][2]))

        for idx, (w, h, d) in indexed_boxes:
            # Try to place in current position
            if x + w > W:
                # Move to next row
                x = 0
                y += row_h
                row_h = 0
                if y + h > H:
                    # Move to next layer
                    y = 0
                    z += layer_d
                    layer_d = 0
                    x, row_h = 0, 0
                    if z + d > D:
                        # New bin
                        b += 1
                        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0

            # Check if it fits
            if x + w > W or y + h > H or z + d > D:
                # Try next bin
                b += 1
                x, y, z, row_h, layer_d = 0, 0, 0, 0, 0

            placements[idx] = (b, x, y, z, w, h, d)
            x += w
            row_h = max(row_h, h)
            layer_d = max(layer_d, d)

        best_result = [None] * n
        for idx in range(n):
            b, x, y, z, w, h, d = placements[idx]
            best_result[idx] = (b, x, y, z, boxes[idx][0], boxes[idx][1], boxes[idx][2])

    return best_result
