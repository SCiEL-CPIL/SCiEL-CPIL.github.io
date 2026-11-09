import time
import itertools
import random
import numpy as np

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    W, H, D = container
    n = len(boxes)

    # Precompute unique rotations for each box
    rotations = []
    for w, h, d in boxes:
        rots = list(set(itertools.permutations((w, h, d))))
        rotations.append(rots)

    def try_placement(ordered_indices):
        """
        Attempt to place boxes in the given order using a candidate point set.
        Returns a list of placements.
        """
        placed = []  # List of (x, y, z, w, h, d)
        results = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        candidates = {(0, 0, 0)}

        for idx in ordered_indices:
            if time.time() > deadline:
                return results, False

            w, h, d = boxes[idx]
            best_pos = None
            best_rot = None
            best_score = float('inf')

            for rw, rh, rd in rotations[idx]:
                # Check if box fits in container
                if rw > W or rh > H or rd > D:
                    continue

                for cx, cy, cz in candidates:
                    # Check bounds
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue

                    # Check overlap
                    valid = True
                    for px, py, pz, pw, ph, pd in placed:
                        if (cx < px + pw and px < cx + rw and
                            cy < py + ph and py < cy + rh and
                            cz < pz + pd and pz < cz + rd):
                            valid = False
                            break

                    if valid:
                        # Score: prioritize lower z, then lower y, then lower x
                        score = cz * 1000000 + cy * 1000 + cx
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_rot = (rw, rh, rd)

            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                placed.append((bx, by, bz, bw, bh, bd))
                results[idx] = (0, bx, by, bz, bw, bh, bd)

                # Generate new candidate points
                # 1. Top
                candidates.add((bx, by, bz + bd))
                # 2. Right
                candidates.add((bx + bw, by, bz))
                # 3. Front
                candidates.add((bx, by + bh, bz))
                # 4. Top-Right
                candidates.add((bx + bw, by, bz + bd))
                # 5. Top-Front
                candidates.add((bx, by + bh, bz + bd))
                # 6. Right-Front
                candidates.add((bx + bw, by + bh, bz))
                # 7. Top-Right-Front
                candidates.add((bx + bw, by + bh, bz + bd))

        return results, True

    def calc_volume(results):
        vol = 0
        for r in results:
            if r[0] == 0:
                vol += r[4] * r[5] * r[6]
        return vol

    best_results = None
    best_vol = 0

    indices = list(range(n))

    # Strategy 1: Sort by volume (descending)
    for sort_key in [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],  # Volume
        lambda i: max(boxes[i]),                            # Max dim
        lambda i: boxes[i][0] * boxes[i][1] + boxes[i][1] * boxes[i][2] + boxes[i][0] * boxes[i][2],  # Surface area
        lambda i: min(boxes[i]),                            # Min dim (ascending)
        lambda i: -min(boxes[i]),                           # Min dim (descending)
    ]:
        if time.time() > deadline:
            break

        sorted_indices = sorted(indices, key=sort_key, reverse=True)
        results, completed = try_placement(sorted_indices)

        if completed:
            vol = calc_volume(results)
            if vol > best_vol:
                best_vol = vol
                best_results = results

    # Strategy 2: Random restarts
    if best_results is not None and time.time() < deadline - 10.0:
        attempts = 0
        while time.time() < deadline - 5.0 and attempts < 30:
            random.shuffle(indices)
            results, completed = try_placement(indices)
            attempts += 1

            if completed:
                vol = calc_volume(results)
                if vol > best_vol:
                    best_vol = vol
                    best_results = results

    # Fallback
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_results
