import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 110.0  # safety margin before 120s/150s kill

    # Precompute all 6 orientations for each box
    # orientation: (w, h, d)
    orients = []
    for w, h, d in boxes:
        opts = set()
        for perm in itertools.permutations((w, h, d)):
            opts.add(perm)
        orients.append(list(opts))

    # Sort boxes by volume descending (largest first)
    order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)

    # Greedy heuristic: place boxes in sorted order, try all orientations,
    # use a "skyline" / corner-based placement to find a valid position.

    def greedy_place(order):
        """Place boxes in given order using greedy corner search."""
        placed = []  # list of (x, y, z, w, h, d)
        assignment = [-1] * n  # assignment[i] = index into placed if placed, else -1

        # Candidate positions: start with (0,0,0)
        # After each placement, add new corners.

        def corners():
            """Generate candidate corner positions from placed boxes."""
            cands = set()
            cands.add((0, 0, 0))
            for (px, py, pz, pw, ph, pd) in placed:
                # Right face
                cands.add((px + pw, py, pz))
                # Front face
                cands.add((px, py + ph, pz))
                # Top face
                cands.add((px, py, pz + pd))
                # Also add combinations for better coverage
                for (cx, cy, cz) in list(cands):
                    pass
            # Filter to those within container
            cands = {(cx, cy, cz) for (cx, cy, cz) in cands
                     if 0 <= cx < W and 0 <= cy < H and 0 <= cz < D}
            return cands

        def can_place(px, py, pz, bw, bh, bd):
            if px + bw > W or py + bh > H or pz + bd > D:
                return False
            for (qx, qy, qz, qw, qh, qd) in placed:
                if not (px + bw <= qx or qx + qw <= px or
                        py + bh <= qy or qy + qh <= py or
                        pz + bd <= qz or qz + qd <= pz):
                    return False
            return True

        for idx in order:
            best_pos = None
            best_orient = None
            best_vol = -1
            for (bw, bh, bd) in orients[idx]:
                for (px, py, pz) in corners():
                    if can_place(px, py, pz, bw, bh, bd):
                        # Prefer positions that are "lower" in some sense to pack tightly
                        # Use a simple heuristic: minimize (pz, py, px)
                        score = (pz, py, px)
                        if best_pos is None or score < best_pos[3]:
                            best_pos = (px, py, pz, score)
                            best_orient = (bw, bh, bd)
            if best_pos is not None:
                px, py, pz, _ = best_pos
                bw, bh, bd = best_orient
                placed.append((px, py, pz, bw, bh, bd))
                assignment[idx] = len(placed) - 1
            else:
                assignment[idx] = -1

        # Build result
        result = []
        for i in range(n):
            if assignment[i] >= 0:
                px, py, pz, bw, bh, bd = placed[assignment[i]]
                result.append((0, px, py, pz, bw, bh, bd))
            else:
                w, h, d = boxes[i]
                result.append((-1, 0, 0, 0, w, h, d))
        return result, sum(bw*bh*bd for (_, _, _, bw, bh, bd) in placed)

    # First pass: greedy
    best_result, best_vol = greedy_place(order)
    container_vol = W * H * D

    # Try a few different orderings to improve packing
    # 1. Volume descending (already done)
    # 2. Volume ascending
    # 3. Random shuffles
    attempts = 0
    max_attempts = 50
    rng = random.Random(42)

    while attempts < max_attempts and time.time() < deadline:
        if attempts == 0:
            continue  # already did volume desc
        if attempts == 1:
            order_try = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2])
        else:
            order_try = list(range(n))
            rng.shuffle(order_try)

        result_try, vol_try = greedy_place(order_try)
        if vol_try > best_vol:
            best_vol = vol_try
            best_result = result_try
        attempts += 1

    # Check deadline
    if time.time() > deadline - 5:
        return best_result

    # Return best result found
    return best_result
# EVOLVE-BLOCK-END
