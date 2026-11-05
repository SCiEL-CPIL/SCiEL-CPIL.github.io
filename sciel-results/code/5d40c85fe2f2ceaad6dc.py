"""XOR-circuit synthesis with bounded candidate interface."""
import time
import numpy as np
from collections import defaultdict
import heapq


# EVOLVE-BLOCK-START
def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0

    A = np.array(matrix, dtype=np.uint8)
    if A.ndim == 1:
        A = A.reshape(1, -1)

    n_rows, n_cols = A.shape

    row_masks = []
    for i in range(n_rows):
        mask = 0
        for j in range(n_cols):
            if A[i, j] == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Gaussian elimination with minimum-weight pivot
    A_work = [[row_masks[i], 1 << i] for i in range(n_rows)]

    r = 0
    pivot_cols = []

    for col in range(n_cols):
        pivot_idx = -1
        min_weight = n_cols + 1
        for i in range(r, n_rows):
            if (A_work[i][0] >> col) & 1:
                w = A_work[i][0].bit_count()
                if w < min_weight:
                    min_weight = w
                    pivot_idx = i

        if pivot_idx == -1:
            continue

        A_work[r], A_work[pivot_idx] = A_work[pivot_idx], A_work[r]

        for i in range(n_rows):
            if i != r and ((A_work[i][0] >> col) & 1):
                A_work[i][0] ^= A_work[r][0]
                A_work[i][1] ^= A_work[r][1]

        pivot_cols.append(col)
        r += 1

    basis_vectors = [A_work[i][0] for i in range(r)]
    pivot_to_basis = {col: i for i, col in enumerate(pivot_cols)}

    # Compute basis representation for each row
    row_basis_reprs = []
    for i in range(n_rows):
        M = row_masks[i]
        repr_mask = 0
        temp = M
        for col in pivot_cols:
            if (temp >> col) & 1:
                b_idx = pivot_to_basis[col]
                repr_mask |= (1 << b_idx)
                temp ^= basis_vectors[b_idx]
        row_basis_reprs.append(repr_mask)

    # Determine all unique masks we need to realize
    needed_masks = set()
    for m in row_masks:
        if m != 0:
            needed_masks.add(m)
    for bv in basis_vectors:
        if bv != 0:
            needed_masks.add(bv)

    # Initialize wire infrastructure
    gates = []
    wire_mask = [1 << i for i in range(n_cols)]
    wire_mask.append(0)  # wire n_cols is zero
    mask_to_wire = {}
    for i in range(n_cols):
        mask_to_wire[1 << i] = i
    mask_to_wire[0] = n_cols
    wire_depth = [0] * n_cols + [0]

    def create_gate(w1, w2):
        """Create a new XOR gate from w1 and w2, return new wire id."""
        mask = wire_mask[w1] ^ wire_mask[w2]
        if mask in mask_to_wire:
            return mask_to_wire[mask]
        gate_id = n_cols + 1 + len(gates)
        gates.append([w1, w2])
        wire_mask.append(mask)
        mask_to_wire[mask] = gate_id
        wire_depth.append(max(wire_depth[w1], wire_depth[w2]) + 1)
        return gate_id

    def get_or_create_wire(mask):
        """Get wire for mask, creating gates if needed."""
        if mask in mask_to_wire:
            return mask_to_wire[mask]
        if mask == 0:
            return n_cols

        # Try to find a pair of existing wires whose XOR gives mask
        # Use a limited search: try all existing wires but with early exit
        best_pair = None
        best_depth = 256

        # First try: check if mask can be formed by XOR with a single input wire
        # (i.e., mask = input_j ^ existing_wire)
        for j in range(n_cols):
            m2 = mask ^ (1 << j)
            if m2 in mask_to_wire:
                w2 = mask_to_wire[m2]
                if w2 != j:
                    d = max(wire_depth[j], wire_depth[w2]) + 1
                    if d < best_depth:
                        best_depth = d
                        best_pair = (j, w2)

        # Then try: check pairs among existing wires
        # To keep this fast, only check a subset if too many wires
        existing_items = list(mask_to_wire.items())
        # Sort by depth for better pruning
        existing_items.sort(key=lambda x: wire_depth[x[1]])

        # Limit search to first 200 wires for speed
        search_limit = min(len(existing_items), 200)
        for i in range(search_limit):
            m1, w1 = existing_items[i]
            if m1 == 0:
                continue
            m2 = mask ^ m1
            if m2 in mask_to_wire:
                w2 = mask_to_wire[m2]
                if w1 == w2:
                    continue
                d = max(wire_depth[w1], wire_depth[w2]) + 1
                if d < best_depth:
                    best_depth = d
                    best_pair = (w1, w2)
                    if best_depth <= 2:
                        break  # Can't do better than depth 2

        if best_pair is not None:
            w1, w2 = best_pair
            return create_gate(w1, w2)

        # Fallback: build from individual bits using balanced tree
        bits = []
        for j in range(n_cols):
            if (mask >> j) & 1:
                bits.append(j)

        if not bits:
            return n_cols
        if len(bits) == 1:
            return bits[0]

        # Balanced tree construction
        def build_tree(bit_list):
            if len(bit_list) == 1:
                return bit_list[0]
            # Try to find existing wire for half XORs
            mid = len(bit_list) // 2
            left_w = build_tree(bit_list[:mid])
            right_w = build_tree(bit_list[mid:])
            return create_gate(left_w, right_w)

        return build_tree(bits)

    # Strategy: resolve all needed masks in order of increasing popcount
    # This maximizes sharing because simpler masks are available when
    # complex masks need to be decomposed
    sorted_needed = sorted(needed_masks, key=lambda m: (m.bit_count(), m))

    for mask in sorted_needed:
        if time.time() > deadline:
            break
        if mask not in mask_to_wire:
            get_or_create_wire(mask)

    # Compute outputs
    outputs = [0] * n_rows
    for i in range(n_rows):
        m = row_masks[i]
        if m == 0:
            outputs[i] = n_cols
        elif m in mask_to_wire:
            outputs[i] = mask_to_wire[m]
        else:
            # Should not happen, but fallback
            outputs[i] = get_or_create_wire(m)

    result = {'gates': gates, 'outputs': outputs}

    if not _verify(A, result):
        # Fallback: simple direct synthesis
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
# EVOLVE-BLOCK-END
