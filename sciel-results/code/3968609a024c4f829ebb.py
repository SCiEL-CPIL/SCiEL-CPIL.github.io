import time
import itertools
import numpy as np
import random

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    n = len(boxes)
    container_vol = W * H * D

    # Precompute volumes and max dims
    box_vols = [w * h * d for w, h, d in boxes]
    box_max_dims = [max(w, h, d) for w, h, d in boxes]
    box_sums = [w + h + d for w, h, d in boxes]
    box_aspect = [max(w, h, d) / min(w, h, d) if min(w, h, d) > 0 else 0 for w, h, d in boxes]

    # Generate multiple sort orders
    def make_order(key_func, reverse=True):
        return sorted(range(n), key=key_func, reverse=reverse)

    orders = [
        make_order(lambda i: box_vols[i]),
        make_order(lambda i: box_max_dims[i]),
        make_order(lambda i: box_sums[i]),
        make_order(lambda i: box_aspect[i]),
        make_order(lambda i: (box_vols[i], box_max_dims[i])),
        make_order(lambda i: box_aspect[i], reverse=False),
        make_order(lambda i: box_vols[i] / (box_aspect[i] + 1)),
        # New orders
        make_order(lambda i: box_vols[i] / (box_max_dims[i] + 1)),
        make_order(lambda i: box_sums[i] / (box_aspect[i] + 1)),
        make_order(lambda i: -box_vols[i]), # Same as first, but explicit
    ]

    def get_orientations(w, h, d):
        """Unique orientations, sorted by z-height ascending for layer packing."""
        orients = list(set(itertools.permutations([w, h, d])))
        orients.sort(key=lambda o: (o[2], o[1], o[0]))
        return orients

    def _try_pack(order, noise_scale=0.0):
        """Try to pack boxes in given order using EPH with optional noise."""
        placed = []  # List of (x, y, z, w, h, d)
        results = [None] * n
        candidate_set = set()
        candidates = [(0, 0, 0)]
        candidate_set.add((0, 0, 0))

        MAX_CANDIDATES = 2000

        def fits(x, y, z, w, h, d):
            if x + w > W or y + h > H or z + d > D:
                return False
            for px, py, pz, pw, ph, pd in placed:
                if not (x + w <= px or px + pw <= x or
                        y + h <= py or py + ph <= y or
                        z + d <= pz or pz + pd <= z):
                    return False
            return True

        def prune_candidates():
            """Remove dominated candidates."""
            nonlocal candidates, candidate_set
            if len(candidates) <= MAX_CANDIDATES:
                return
            # Sort by z, then y, then x
            candidates.sort(key=lambda c: (c[2], c[1], c[0]))
            kept = []
            kept_set = set()
            for c in candidates:
                dominated = False
                for kc in kept:
                    if kc[0] <= c[0] and kc[1] <= c[1] and kc[2] <= c[2] and (kc[0], kc[1], kc[2]) != c:
                        dominated = True
                        break
                if not dominated:
                    kept.append(c)
                    kept_set.add(c)
                if len(kept) >= MAX_CANDIDATES:
                    break
            candidates = kept
            candidate_set = kept_set

        def cost_function(x, y, z, w, h, d):
            """Cost: prioritize low z, then compact in y and x. Add noise if requested."""
            cost = z * (H + 1) * (W + 1) + y * (W + 1) + x
            if noise_scale > 0:
                cost += random.uniform(0, noise_scale)
            return cost

        for idx in order:
            if time.time() > deadline:
                break

            w, h, d = boxes[idx]
            best_placement = None
            best_cost = float('inf')

            orientations = get_orientations(w, h, d)

            # Sort candidates by cost to find good placements faster
            # Note: Sorting 2000 candidates for each of 60 boxes is 120k sorts.
            # This might be slow. We can just iterate and keep the best.
            # But sorting allows early exit if we find cost 0?
            # Let's just iterate all candidates to be safe, it's O(N_cands * N_orients * N_placed)
            
            for cx, cy, cz in candidates:
                if (cx, cy, cz) not in candidate_set:
                    continue

                for ow, oh, od in orientations:
                    if not fits(cx, cy, cz, ow, oh, od):
                        continue
                    cost = cost_function(cx, cy, cz, ow, oh, od)
                    if cost < best_cost:
                        best_cost = cost
                        best_placement = (cx, cy, cz, ow, oh, od)
                        if cost == 0:
                            break
                if best_placement and best_cost == 0:
                    break

            if best_placement:
                bx, by, bz, bw, bh, bd = best_placement
                placed.append((bx, by, bz, bw, bh, bd))
                results[idx] = (0, bx, by, bz, bw, bh, bd)

                # Generate new candidates from faces of the placed box
                new_cands = [
                    (bx + bw, by, bz),
                    (bx, by + bh, bz),
                    (bx, by, bz + bd),
                    (bx + bw, by + bh, bz),
                    (bx + bw, by, bz + bd),
                    (bx, by + bh, bz + bd),
                    (bx + bw, by + bh, bz + bd),
                ]
                for nc in new_cands:
                    if 0 <= nc[0] <= W and 0 <= nc[1] <= H and 0 <= nc[2] <= D:
                        if nc not in candidate_set:
                            candidates.append(nc)
                            candidate_set.add(nc)

                prune_candidates()
            else:
                results[idx] = (-1, 0, 0, 0, w, h, d)

        # Fill remaining
        for i in range(n):
            if results[i] is None:
                w, h, d = boxes[i]
                results[i] = (-1, 0, 0, 0, w, h, d)

        # Compute filled volume
        filled = 0.0
        for r in results:
            if r[0] == 0:
                filled += r[4] * r[5] * r[6]
        return results, filled

    best_result = None
    best_filled = 0.0

    # 1. Deterministic runs with different orders
    for order in orders:
        if time.time() > deadline:
            break
        result, filled = _try_pack(order, noise_scale=0.0)
        if filled > best_filled:
            best_filled = filled
            best_result = result

    # 2. Randomized runs (Simulated Annealing-like noise)
    # Use remaining time for randomized restarts
    time_left = deadline - time.time()
    if time_left > 5:
        # Estimate time per run
        # Assume each run takes ~0.05s
        num_random_runs = int(time_left / 0.1) # Conservative
        num_random_runs = max(10, min(num_random_runs, 200))
        
        for i in range(num_random_runs):
            if time.time() > deadline - 1:
                break
            
            # Pick a random order from the existing ones or a random permutation
            # Random permutation is too expensive to generate and sort every time?
            # Just pick a random order from the list and add noise
            order = random.choice(orders)
            
            # Noise scale: small enough to not destroy the heuristic, large enough to explore
            # Max cost is approx D*(H+1)*(W+1). 
            # Let's use a relative noise
            noise_scale = 1.0 + random.random() * 10.0
            
            result, filled = _try_pack(order, noise_scale=noise_scale)
            if filled > best_filled:
                best_filled = filled
                best_result = result

    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_result
