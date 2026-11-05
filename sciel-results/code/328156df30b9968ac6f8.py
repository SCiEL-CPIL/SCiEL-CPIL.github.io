"""XOR-circuit synthesis with bounded candidate interface."""
import time
import numpy as np
from collections import defaultdict


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

    # Gate synthesis with smarter sharing
    gates = []
    wire_mask = [1 << i for i in range(n_cols)]
    wire_mask.append(0)
    mask_to_wire = {}
    for i in range(n_cols):
        mask_to_wire[1 << i] = i
    mask_to_wire[0] = n_cols

    wire_depth = [0] * n_cols + [0]

    def get_or_create_wire(mask):
        nonlocal gates, wire_mask, mask_to_wire, wire_depth

        if mask in mask_to_wire:
            return mask_to_wire[mask]

        if mask == 0:
            return n_cols

        # Strategy: try to decompose mask = m1 ^ m2 where m1, m2 are existing
        # Prefer decompositions where both m1 and m2 have low depth
        # Instead of scanning all, try masks with low bit-count first
        
        # Collect existing masks sorted by (bit_count, depth)
        existing_masks = []
        for m, w in mask_to_wire.items():
            if m != 0:
                existing_masks.append((m.bit_count(), wire_depth[w], m, w))
        
        existing_masks.sort()
        
        best_pair = None
        best_depth = 256
        
        # Try a limited number of candidates to avoid O(N^2)
        max_candidates = min(len(existing_masks), 200)
        
        for _bc, _dep, m1, w1 in existing_masks[:max_candidates]:
            m2 = mask ^ m1
            if m2 in mask_to_wire:
                w2 = mask_to_wire[m2]
                if w1 != w2:
                    d = max(wire_depth[w1], wire_depth[w2]) + 1
                    if d < best_depth:
                        best_depth = d
                        best_pair = (w1, w2)
        
        if best_pair is not None:
            w1, w2 = best_pair
            gate_id = n_cols + 1 + len(gates)
            gates.append([w1, w2])
            wire_mask.append(mask)
            mask_to_wire[mask] = gate_id
            wire_depth.append(best_depth)
            return gate_id

        # Fallback: build from individual bits using a different strategy
        bits = []
        for j in range(n_cols):
            if (mask >> j) & 1:
                bits.append(j)

        if not bits:
            return n_cols
        if len(bits) == 1:
            return bits[0]

        # Build incrementally, trying to reuse existing partial XORs
        cur_wire = bits[0]
        for b in bits[1:]:
            target = wire_mask[cur_wire] ^ (1 << b)
            if target in mask_to_wire:
                cur_wire = mask_to_wire[target]
            else:
                gid = n_cols + 1 + len(gates)
                gates.append([cur_wire, b])
                wire_mask.append(target)
                mask_to_wire[target] = gid
                wire_depth.append(wire_depth[cur_wire] + 1)
                cur_wire = gid

        return cur_wire

    # Compute basis wires first
    basis_wires = []
    for b in range(r):
        bw = get_or_create_wire(basis_vectors[b])
        basis_wires.append(bw)

    # Compute output wires
    repr_count = defaultdict(int)
    for rm in row_basis_reprs:
        repr_count[rm] += 1

    row_order = sorted(range(n_rows), key=lambda i: (-repr_count[row_basis_reprs[i]], i))

    outputs = [0] * n_rows
    repr_to_wire = {0: n_cols}

    for i in row_order:
        if time.time() > deadline:
            break

        repr_mask = row_basis_reprs[i]

        if repr_mask in repr_to_wire:
            outputs[i] = repr_to_wire[repr_mask]
            continue

        active = []
        for b in range(r):
            if (repr_mask >> b) & 1:
                active.append(b)

        if not active:
            outputs[i] = n_cols
            repr_to_wire[repr_mask] = n_cols
            continue

        if len(active) == 1:
            w = basis_wires[active[0]]
            outputs[i] = w
            repr_to_wire[repr_mask] = w
            continue

        w = get_or_create_wire(row_masks[i])
        outputs[i] = w
        repr_to_wire[repr_mask] = w

    # Ensure all outputs are set
    for i in range(n_rows):
        if outputs[i] == 0 and n_cols > 0:
            repr_mask = row_basis_reprs[i]
            active = []
            for b in range(r):
                if (repr_mask >> b) & 1:
                    active.append(b)
            if not active:
                outputs[i] = n_cols
            elif len(active) == 1:
                outputs[i] = basis_wires[active[0]]
            else:
                bits = []
                for j in range(n_cols):
                    if (row_masks[i] >> j) & 1:
                        bits.append(j)
                if not bits:
                    outputs[i] = n_cols
                elif len(bits) == 1:
                    outputs[i] = bits[0]
                else:
                    cur = bits[0]
                    for b in bits[1:]:
                        gid = n_cols + 1 + len(gates)
                        gates.append([cur, b])
                        wire_mask.append(wire_mask[cur] ^ (1 << b))
                        mask_to_wire[wire_mask[cur] ^ (1 << b)] = gid
                        wire_depth.append(wire_depth[cur] + 1)
                        cur = gid
                    outputs[i] = cur

    result = {'gates': gates, 'outputs': outputs}

    if not _verify(A, result):
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
