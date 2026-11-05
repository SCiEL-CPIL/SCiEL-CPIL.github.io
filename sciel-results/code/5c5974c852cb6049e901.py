"""XOR-circuit synthesis via MST-inspired global CSE with balanced tree and polish."""
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)

    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs

    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}

    # Convert rows to integer bitmasks for fast XOR operations
    row_masks = []
    for i in range(n_rows):
        mask = 0
        for j in range(n_cols):
            if A[i, j] == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Wire representation: wire_mask[w] = bitmask of inputs that wire w represents
    wire_mask = [0] * (n_inputs + 1)
    for i in range(n_inputs):
        wire_mask[i] = (1 << i)
    wire_mask[n_inputs] = 0  # zero wire

    mask_to_wire = {}
    for i in range(n_inputs + 1):
        mask_to_wire[wire_mask[i]] = i

    # Each row is a list of wire indices whose XOR gives the row value
    row_wires = []
    for i in range(n_rows):
        wires = []
        mask = row_masks[i]
        for j in range(n_cols):
            if mask & (1 << j):
                wires.append(j)
        row_wires.append(wires)

    gates = []
    next_wire = n_inputs + 1

    # Track depth per wire for tie-breaking
    wire_depth = [0] * (n_inputs + 1)

    def compute_cost():
        cost = len(gates)
        for wires in row_wires:
            if len(wires) > 1:
                cost += len(wires) - 1
        return cost

    def check_time():
        return time.time() < deadline - 3.0

    max_iter = 50000
    iter_count = 0
    no_improvement = 0

    # Phase 1: MST-inspired greedy CSE
    # We build a priority queue of candidate pairs based on frequency
    while iter_count < max_iter and check_time():
        iter_count += 1

        current_cost = compute_cost()

        # Find co-occurring pairs and their frequencies
        pair_rows = defaultdict(list)
        for i, wires in enumerate(row_wires):
            n_w = len(wires)
            if n_w < 2:
                continue
            for x in range(n_w):
                for y in range(x + 1, n_w):
                    a, b = wires[x], wires[y]
                    if a > b:
                        a, b = b, a
                    pair_rows[(a, b)].append(i)

        if not pair_rows:
            break

        # Score each candidate pair using MST-inspired heuristic
        # Weight = frequency * (1 + transitive_bonus) - creation_cost
        best_score = -1e9
        best_pair = None
        best_new_mask = None
        best_rows = None
        best_existing = -1
        best_depth = 999

        for (a, b), rows in pair_rows.items():
            if len(rows) < 2:
                continue

            new_mask = wire_mask[a] ^ wire_mask[b]
            if new_mask == 0:
                continue

            direct_benefit = len(rows)

            # Transitive bonus: estimate future sharing opportunities
            transitive_bonus = 0.0
            for i in rows:
                wires_i = row_wires[i]
                n_other = len(wires_i) - 2
                if n_other > 0:
                    # Smooth surrogate: diminishing returns
                    transitive_bonus += 0.4 * min(n_other, 4) / (1.0 + 0.15 * n_other)

            if new_mask in mask_to_wire:
                existing = mask_to_wire[new_mask]
                score = direct_benefit + transitive_bonus
                depth = wire_depth[existing]
            else:
                existing = -1
                popcnt = bin(new_mask).count('1')
                # Creation cost: 1 gate + small penalty
                score = direct_benefit + transitive_bonus - 1.0 - 0.02 * popcnt
                depth = max(wire_depth[a], wire_depth[b]) + 1

            # Tie-break: prefer lower depth, then higher frequency
            if score > best_score + 1e-9 or (abs(score - best_score) < 1e-9 and (depth < best_depth or (depth == best_depth and len(rows) > len(best_rows) if best_rows else depth < best_depth))):
                best_score = score
                best_pair = (a, b)
                best_new_mask = new_mask
                best_rows = rows
                best_existing = existing
                best_depth = depth

        if best_pair is None or best_score <= 0:
            break

        a, b = best_pair

        if best_existing >= 0:
            new_wire = best_existing
        else:
            new_wire = next_wire
            next_wire += 1
            wire_mask.append(best_new_mask)
            mask_to_wire[best_new_mask] = new_wire
            wire_depth.append(max(wire_depth[a], wire_depth[b]) + 1)
            gates.append([a, b])

        # Update affected rows
        for i in best_rows:
            wires = row_wires[i]
            if a in wires and b in wires:
                wires.remove(a)
                wires.remove(b)
                if new_wire in wires:
                    wires.remove(new_wire)
                else:
                    wires.append(new_wire)

        new_cost = compute_cost()

        if new_cost < current_cost:
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement > 150:
                break

    # Phase 2: Polish pass - replace pairs with existing wires
    if check_time():
        improved = True
        polish_iter = 0
        while improved and polish_iter < 100 and check_time():
            improved = False
            polish_iter += 1
            for i in range(n_rows):
                wires = row_wires[i]
                if len(wires) < 2:
                    continue
                n_w = len(wires)
                for x in range(n_w):
                    for y in range(x + 1, n_w):
                        a, b = wires[x], wires[y]
                        if a > b:
                            a, b = b, a
                        new_mask = wire_mask[a] ^ wire_mask[b]
                        if new_mask == 0:
                            continue
                        if new_mask in mask_to_wire:
                            new_wire = mask_to_wire[new_mask]
                            if new_wire not in wires:
                                wires.remove(a)
                                wires.remove(b)
                                wires.append(new_wire)
                                improved = True
                                break
                    if improved:
                        break

    # Phase 3: Second round of CSE on remaining structure
    if check_time():
        no_improvement2 = 0
        for round2_iter in range(20000):
            if not check_time():
                break

            current_cost = compute_cost()

            pair_rows = defaultdict(list)
            for i, wires in enumerate(row_wires):
                n_w = len(wires)
                if n_w < 2:
                    continue
                for x in range(n_w):
                    for y in range(x + 1, n_w):
                        a, b = wires[x], wires[y]
                        if a > b:
                            a, b = b, a
                        pair_rows[(a, b)].append(i)

            if not pair_rows:
                break

            best_score = -1e9
            best_pair = None
            best_new_mask = None
            best_rows = None
            best_existing = -1
            best_depth = 999

            for (a, b), rows in pair_rows.items():
                if len(rows) < 2:
                    continue

                new_mask = wire_mask[a] ^ wire_mask[b]
                if new_mask == 0:
                    continue

                direct_benefit = len(rows)
                transitive_bonus = 0.0
                for i in rows:
                    wires_i = row_wires[i]
                    n_other = len(wires_i) - 2
                    if n_other > 0:
                        transitive_bonus += 0.3 * min(n_other, 3) / (1.0 + 0.1 * n_other)

                if new_mask in mask_to_wire:
                    existing = mask_to_wire[new_mask]
                    score = direct_benefit + transitive_bonus
                    depth = wire_depth[existing]
                else:
                    existing = -1
                    popcnt = bin(new_mask).count('1')
                    score = direct_benefit + transitive_bonus - 1.0 - 0.03 * popcnt
                    depth = max(wire_depth[a], wire_depth[b]) + 1

                if score > best_score + 1e-9 or (abs(score - best_score) < 1e-9 and depth < best_depth):
                    best_score = score
                    best_pair = (a, b)
                    best_new_mask = new_mask
                    best_rows = rows
                    best_existing = existing
                    best_depth = depth

            if best_pair is None or best_score <= 0:
                break

            a, b = best_pair
            if best_existing >= 0:
                new_wire = best_existing
            else:
                new_wire = next_wire
                next_wire += 1
                wire_mask.append(best_new_mask)
                mask_to_wire[best_new_mask] = new_wire
                wire_depth.append(max(wire_depth[a], wire_depth[b]) + 1)
                gates.append([a, b])

            for i in best_rows:
                wires = row_wires[i]
                if a in wires and b in wires:
                    wires.remove(a)
                    wires.remove(b)
                    if new_wire in wires:
                        wires.remove(new_wire)
                    else:
                        wires.append(new_wire)

            new_cost = compute_cost()
            if new_cost < current_cost:
                no_improvement2 = 0
            else:
                no_improvement2 += 1
                if no_improvement2 > 100:
                    break

    # Final output assembly with balanced binary tree to control depth
    outputs = []
    for i in range(n_rows):
        wires = row_wires[i]
        if len(wires) == 0:
            outputs.append(zero_wire)
        elif len(wires) == 1:
            outputs.append(wires[0])
        else:
            # Build balanced XOR tree
            current_level = list(wires)
            while len(current_level) > 1:
                next_level = []
                for k in range(0, len(current_level), 2):
                    if k + 1 < len(current_level):
                        w = next_wire
                        next_wire += 1
                        gates.append([current_level[k], current_level[k + 1]])
                        next_level.append(w)
                    else:
                        next_level.append(current_level[k])
                current_level = next_level
            outputs.append(current_level[0])

    return {'gates': gates, 'outputs': outputs}
