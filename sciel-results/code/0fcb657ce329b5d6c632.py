"""XOR-circuit synthesis: iterative greedy sharing + row-pair reduction."""
import time
import numpy as np


def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 52.0

    A = np.array(matrix, dtype=np.uint8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    n_rows, n_cols = A.shape

    if n_rows == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to integer masks
    row_masks = []
    for i in range(n_rows):
        m = 0
        for j in range(n_cols):
            if A[i, j]:
                m |= (1 << j)
        row_masks.append(m)

    # Identify unique non-zero target masks
    target_masks = sorted(set(m for m in row_masks if m != 0),
                          key=lambda m: (m.bit_count(), m))

    # --- Row-pair XOR reduction: replace high-popcount rows with pair XORs ---
    # If row_i XOR row_j has much lower popcount, it's cheaper to build the pair
    # and then XOR with one row to get the other.
    # We do this iteratively: find pairs where XOR has low popcount, create the
    # XOR as a target, and replace the higher-popcount row.

    # Build a list of (mask, row_indices) for current representation
    # Each row is either a single target mask or a pair (mask_a, mask_b) meaning row = mask_a XOR mask_b
    row_reps = [(m, None) for m in row_masks]  # (primary_mask, secondary_mask_or_None)

    # Iteratively find beneficial pair reductions
    for _ in range(3):
        if time.time() > deadline:
            break
        best_pair = None
        best_saving = 0
        for i in range(len(row_reps)):
            mi = row_reps[i][0]
            if mi == 0:
                continue
            pi = row_reps[i][1]
            for j in range(i + 1, len(row_reps)):
                mj = row_reps[j][0]
                if mj == 0:
                    continue
                pj = row_reps[j][1]
                # Cost of building mi and mj separately
                cost_sep = mi.bit_count() + mj.bit_count()
                # Cost if we build mi^mj and use it
                xor_mask = mi ^ mj
                cost_pair = xor_mask.bit_count() + min(mi.bit_count(), mj.bit_count())
                saving = cost_sep - cost_pair
                if saving > best_saving:
                    best_saving = saving
                    best_pair = (i, j, xor_mask)
        if best_pair and best_saving >= 2:
            i, j, xor_mask = best_pair
            # Replace the higher-popcount row with (other_row, xor_mask)
            if row_reps[i][0].bit_count() >= row_reps[j][0].bit_count():
                # Row i becomes: row_j XOR xor_mask
                row_reps[i] = (row_reps[j][0], xor_mask)
            else:
                row_reps[j] = (row_reps[i][0], xor_mask)
        else:
            break

    # Collect all masks we need to realize
    needed = set()
    for m, pm in row_reps:
        if m != 0:
            needed.add(m)
        if pm is not None and pm != 0:
            needed.add(pm)

    # --- Iterative greedy sharing synthesis ---
    gates = []
    mask_to_wire = {0: n_cols}
    wire_mask = [0] * (n_cols + 1)
    wire_mask[n_cols] = 0
    for i in range(n_cols):
        mask_to_wire[1 << i] = i
        wire_mask[i] = 1 << i

    def create_gate(w1, w2):
        m = wire_mask[w1] ^ wire_mask[w2]
        if m in mask_to_wire:
            return mask_to_wire[m]
        gid = n_cols + 1 + len(gates)
        gates.append([w1, w2])
        wire_mask.append(m)
        mask_to_wire[m] = gid
        return gid

    # Greedy: repeatedly find the gate (w1, w2) that maximizes benefit
    # Benefit = number of remaining targets that can be expressed using the new wire
    # + reduction in total construction cost

    remaining = set(needed)

    def count_enabled(new_mask):
        """Count how many remaining targets can be expressed with new_mask + existing wires."""
        count = 0
        for t in remaining:
            # Can t be done as new_mask XOR existing_wire?
            if (t ^ new_mask) in mask_to_wire:
                count += 1
            # Can t be done as new_mask XOR input_j?
            for j in range(n_cols):
                if (t ^ new_mask) == (1 << j):
                    count += 1
                    break
        return count

    def estimate_cost(mask):
        """Estimate cost to build mask from current wires."""
        if mask in mask_to_wire:
            return 0
        # Cost = popcount - 1 if built from inputs, but check if cheaper via existing
        # Try: mask = existing ^ input
        best = mask.bit_count() - 1  # cost from scratch (tree of inputs)
        for w, m in mask_to_wire.items():
            if m == 0:
                continue
            diff = mask ^ m
            if diff == 0:
                return 0
            # diff is a single bit?
            if diff & (diff - 1) == 0:
                best = min(best, 1)
            elif diff.bit_count() < best + 1:
                # Could be cheaper: 1 gate + cost(diff)
                # Approximate: 1 + diff.bit_count() - 1 = diff.bit_count()
                best = min(best, diff.bit_count())
        return best

    max_iter = 500
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        if not remaining:
            break

        # Find best gate to create
        best_benefit = 0
        best_pair = None
        best_new_mask = None

        # Candidate pairs: (existing_wire, input) and (existing_wire, existing_wire)
        # For speed, only consider wires that are "useful" (low depth or high sharing)

        existing_wires = list(mask_to_wire.items())
        # Limit search for large problems
        if len(existing_wires) > 300:
            # Sort by popcount of mask (simpler wires first)
            existing_wires.sort(key=lambda x: x[0].bit_count())
            existing_wires = existing_wires[:300]

        # Try pairs with inputs
        for m1, w1 in existing_wires:
            if m1 == 0:
                continue
            for j in range(n_cols):
                new_m = m1 ^ (1 << j)
                if new_m == 0 or new_m in mask_to_wire:
                    continue
                # Benefit: how many remaining targets does this enable?
                benefit = 0
                for t in remaining:
                    if (t ^ new_m) in mask_to_wire:
                        benefit += 1
                # Also: does new_m itself reduce cost of remaining targets?
                if new_m in remaining:
                    benefit += 2  # bonus for directly completing a target
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_pair = (w1, j)
                    best_new_mask = new_m

        # Try pairs between existing wires (limited)
        if len(existing_wires) <= 150:
            for idx1 in range(len(existing_wires)):
                m1, w1 = existing_wires[idx1]
                if m1 == 0:
                    continue
                for idx2 in range(idx1 + 1, len(existing_wires)):
                    m2, w2 = existing_wires[idx2]
                    if m2 == 0:
                        continue
                    new_m = m1 ^ m2
                    if new_m == 0 or new_m in mask_to_wire:
                        continue
                    benefit = 0
                    for t in remaining:
                        if (t ^ new_m) in mask_to_wire:
                            benefit += 1
                    if new_m in remaining:
                        benefit += 2
                    if benefit > best_benefit:
                        best_benefit = benefit
                        best_pair = (w1, w2)
                        best_new_mask = new_m

        if best_pair is None or best_benefit <= 0:
            # No beneficial sharing found; build remaining targets directly
            break

        # Create the gate
        w1, w2 = best_pair
        create_gate(w1, w2)
        # Remove newly satisfied targets from remaining
        for t in list(remaining):
            if t in mask_to_wire:
                remaining.discard(t)

    # Build any remaining targets directly (from inputs via balanced tree)
    for t in sorted(remaining, key=lambda m: m.bit_count()):
        if time.time() > deadline:
            break
        if t in mask_to_wire:
            continue
        # Try to express as existing_wire XOR input
        found = False
        for j in range(n_cols):
            diff = t ^ (1 << j)
            if diff in mask_to_wire:
                create_gate(mask_to_wire[diff], j)
                found = True
                break
        if found:
            continue
        # Try existing XOR existing
        for m1, w1 in list(mask_to_wire.items()):
            if m1 == 0:
                continue
            diff = t ^ m1
            if diff in mask_to_wire and diff != m1:
                create_gate(w1, mask_to_wire[diff])
                found = True
                break
        if found:
            continue
        # Build from individual bits
        bits = [j for j in range(n_cols) if (t >> j) & 1]
        if not bits:
            continue
        if len(bits) == 1:
            continue  # already a wire
        # Balanced tree
        def build(bits_list):
            if len(bits_list) == 1:
                return bits_list[0]
            mid = len(bits_list) // 2
            lw = build(bits_list[:mid])
            rw = build(bits_list[mid:])
            return create_gate(lw, rw)
        build(bits)

    # Compute outputs
    outputs = []
    for m, pm in row_reps:
        if m == 0:
            outputs.append(n_cols)
        elif pm is None:
            # Single mask
            if m in mask_to_wire:
                outputs.append(mask_to_wire[m])
            else:
                # Should not happen
                bits = [j for j in range(n_cols) if (m >> j) & 1]
                if not bits:
                    outputs.append(n_cols)
                else:
                    w = bits[0]
                    for b in bits[1:]:
                        w = create_gate(w, b)
                    outputs.append(w)
        else:
            # m XOR pm
            w1 = mask_to_wire.get(m)
            w2 = mask_to_wire.get(pm)
            if w1 is not None and w2 is not None:
                outputs.append(create_gate(w1, w2))
            else:
                # Fallback: build m directly
                bits = [j for j in range(n_cols) if (m >> j) & 1]
                if not bits:
                    outputs.append(n_cols)
                else:
                    w = bits[0]
                    for b in bits[1:]:
                        w = create_gate(w, b)
                    outputs.append(w)

    result = {'gates': gates, 'outputs': outputs}

    # Verify
    if not _verify(A, result):
        # Fallback: direct synthesis
        gates = []
        outputs = []
        for i in range(n_rows):
            idx = [j for j in range(n_cols) if A[i, j] == 1]
            if not idx:
                outputs.append(n_cols)
            elif len(idx) == 1:
                outputs.append(idx[0])
            else:
                cur = idx[0]
                for w in idx[1:]:
                    gid = n_cols + 1 + len(gates)
                    gates.append([cur, w])
                    cur = gid
                outputs.append(cur)
        result = {'gates': gates, 'outputs': outputs}

    return result


def _verify(A, result):
    n_cols = A.shape[1]
    gates = result['gates']
    outputs = result['outputs']

    if n_cols > 64:
        return True

    wire_values = [1 << i for i in range(n_cols)]
    wire_values.append(0)

    for gate in gates:
        w1, w2 = gate
        if w1 >= len(wire_values) or w2 >= len(wire_values):
            return False
        wire_values.append(wire_values[w1] ^ wire_values[w2])

    for i, ow in enumerate(outputs):
        if ow >= len(wire_values):
            return False
        expected = 0
        for j in range(n_cols):
            if A[i, j] == 1:
                expected ^= (1 << j)
        if wire_values[ow] != expected:
            return False

    return True
