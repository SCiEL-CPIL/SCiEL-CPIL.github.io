import time
import random
import itertools

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit

    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    n = len(boxes)

    # Pre-calculate orientations for each box
    box_orientations = []
    for w, h, d in boxes:
        perms = list(set(itertools.permutations((w, h, d))))
        box_orientations.append(perms)

    volumes = [w * h * d for w, h, d in boxes]

    best_fill = 0
    best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    def can_place(x, y, z, w, h, d, placed):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W + 1e-9 or y + h > H + 1e-9 or z + d > D + 1e-9:
            return False
        for px, py, pz, pw, ph, pd in placed:
            if not (x + w <= px + 1e-9 or px + pw <= x + 1e-9 or
                    y + h <= py + 1e-9 or py + ph <= y + 1e-9 or
                    z + d <= pz + 1e-9 or pz + pd <= z + 1e-9):
                return False
        return True

    def find_position(w, h, d, placed):
        # Candidate positions: 0 and edges of existing boxes
        x_cands = [0]
        y_cands = [0]
        z_cands = [0]

        for px, py, pz, pw, ph, pd in placed:
            if px + pw <= W + 1e-9:
                x_cands.append(px + pw)
            if py + ph <= H + 1e-9:
                y_cands.append(py + ph)
            if pz + pd <= D + 1e-9:
                z_cands.append(pz + pd)

        x_cands = sorted(set(x_cands))
        y_cands = sorted(set(y_cands))
        z_cands = sorted(set(z_cands))

        best_pos = None
        best_score = float('inf')

        # Heuristic: minimize z (height), then y (depth), then x (width)
        # This tends to stack boxes and fill from the bottom up
        for z in z_cands:
            if z + d > D + 1e-9:
                continue
            for y in y_cands:
                if y + h > H + 1e-9:
                    continue
                for x in x_cands:
                    if x + w > W + 1e-9:
                        continue
                    if can_place(x, y, z, w, h, d, placed):
                        # Score: primarily minimize z, then y, then x
                        score = z * (H * W) + y * W + x
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        return best_pos

    def run_pack(order):
        placed = []
        placements = [None] * n
        current_vol = 0

        for idx in order:
            w, h, d = boxes[idx]
            orients = box_orientations[idx]

            best_pos = None
            best_orient = None
            best_score = float('inf')

            for ow, oh, od in orients:
                pos = find_position(ow, oh, od, placed)
                if pos is not None:
                    x, y, z = pos
                    score = z * (H * W) + y * W + x
                    if score < best_score:
                        best_score = score
                        best_pos = pos
                        best_orient = (ow, oh, od)

            if best_pos is not None:
                x, y, z = best_pos
                ow, oh, od = best_orient
                placed.append((x, y, z, ow, oh, od))
                placements[idx] = (0, x, y, z, ow, oh, od)
                current_vol += ow * oh * od
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)

        return current_vol, placements

    # Multi-start strategy with various heuristics
    orders = []

    # 1. Volume descending
    orders.append(sorted(range(n), key=lambda i: volumes[i], reverse=True))

    # 2. Volume ascending
    orders.append(sorted(range(n), key=lambda i: volumes[i]))

    # 3. Longest edge descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))

    # 4. Shortest edge descending
    orders.append(sorted(range(n), key=lambda i: min(boxes[i]), reverse=True))

    # 5. Sum of dimensions descending
    orders.append(sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True))

    # 6-25: Random permutations
    for _ in range(20):
        random_order = list(range(n))
        random.shuffle(random_order)
        orders.append(random_order)

    # Run all orders
    for i, order in enumerate(orders):
        if time.time() - start_time > DEADLINE:
            break

        vol, placements = run_pack(order)

        if vol > best_fill:
            best_fill = vol
            best_placements = placements

    # If we have time, do local search / refinement
    # Try to swap or re-order some boxes to improve fill
    if time.time() - start_time < DEADLINE - 10:
        # Try a few more random orders with slight perturbations
        for _ in range(10):
            if time.time() - start_time > DEADLINE:
                break
            random_order = list(range(n))
            random.shuffle(random_order)
            vol, placements = run_pack(random_order)
            if vol > best_fill:
                best_fill = vol
                best_placements = placements

    return best_placements
# EVOLVE-BLOCK-END
