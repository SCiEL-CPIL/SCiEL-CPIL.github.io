"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 90.0
_MARGIN = 8.0

def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _T0)

def _row_to_int(row):
    """Encode a GF(2) row (list of 0/1) as an integer bitmask (index i -> bit i)."""
    v = 0
    for i, b in enumerate(row):
        if b:
            v |= (1 << i)
    return v

def _int_to_row(v, n):
    return [1 if (v >> i) & 1 else 0 for i in range(n)]

def _baseline_gate_count(matrix):
    """Independent-row baseline: sum of popcount of each output row (XORs of inputs)."""
    rows = np.asarray(matrix, dtype=np.int64)
    if rows.size == 0:
        return 0
    return int(rows.sum())

def _depth_of(v, depth_of_wires):
    return depth_of_wires[v]

def _build_balanced_xor(terms, wire_id, depth_of_wires, gates):
    """
    Build a balanced binary tree XORing `terms` (list of wire ids) producing
    wire `wire_id`. Appends [a,b] pairs to `gates` (each new gate gets the next
    wire id starting from wire_id, in order). Returns the final wire id.
    """
    terms = list(terms)
    if len(terms) == 1:
        # No gate needed; the output is just the input wire.
        return terms[0]
    # We will consume wire ids starting at wire_id.
    # Simulate the balanced reduction, assigning wire ids as we go.
    # We need exactly (len(terms)-1) gates.
    # Build iteratively level by level; assign ids in creation order.
    level = list(terms)
    next_wire = wire_id
    # Precompute how many wires each level produces.
    while len(level) > 1:
        pairs = len(level) // 2
        carry = len(level) % 2
        new_level = []
        for p in range(pairs):
            a = level[2 * p]
            b = level[2 * p + 1]
            w = next_wire
            next_wire += 1
            gates.append([a, b])
            depth_of_wires[w] = 1 + max(depth_of_wires[a], depth_of_wires[b])
            new_level.append(w)
        for c in range(carry):
            new_level.append(level[2 * pairs + c])
        level = new_level
    return level[0]

def synthesize(matrix, depth_limit):
    """
    Exact XOR-circuit synthesis minimizing mean gate-count / independent-row
    baseline via global basis reduction + balanced-tree shared synthesis.
    """
    start = time.time()
    rows = np.asarray(matrix, dtype=np.int64)
    if rows.size == 0:
        return {'gates': [], 'outputs': []}
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    m, n = rows.shape

    # Constant-zero wire is n.
    # Input wires 0..n-1, constant zero = n.
    # Gate k produces wire n+1+k.

    # Convert rows to integer bitmasks over the n input positions.
    row_ints = [_row_to_int(rows[i]) for i in range(m)]

    # Gaussian elimination over GF(2) to find a minimal basis of the row space.
    # basis: list of (pivot_bit, int_mask)
    basis = []
    for r in row_ints:
        cur = r
        for (pb, mask) in basis:
            if (cur >> pb) & 1:
                cur ^= mask
        if cur == 0:
            continue
        # Find highest set bit as pivot.
        pb = cur.bit_length() - 1
        # Eliminate this pivot from existing basis (to keep reduced form) - optional.
        basis.append((pb, cur))
        # Re-reduce existing basis against this new pivot for cleanliness.
        new_basis = []
        for (qpb, qmask) in basis:
            if (qpb, qmask) == (pb, cur):
                new_basis.append((pb, cur))
                continue
            if (qmask >> pb) & 1:
                qmask ^= cur
            new_basis.append((qpb, qmask))
        basis = new_basis

    # Express each row as combination of basis vectors (which basis indices).
    # basis is reduced enough; reduce each row against basis.
    row_basis_combo = []
    for r in row_ints:
        cur = r
        combo = [0] * len(basis)
        for i, (pb, mask) in enumerate(basis):
            if (cur >> pb) & 1:
                cur ^= mask
                combo[i] = 1
        # cur should be 0
        row_basis_combo.append(combo)

    # Now synthesize each basis vector as a gate network producing a wire.
    # Each basis vector is an XOR of a subset of input wires (0..n-1).
    # We synthesize all basis vectors, sharing nothing between them initially,
    # but each basis wire can be reused by many outputs (that's the win).

    # depth_of_wires: index by wire id. Inputs 0..n-1 have depth 0. Constant zero depth 0.
    depth_of_wires = [0] * (n + 1 + max(1, len(basis)) + sum(c.count(1) for c in row_basis_combo) + 8)
    for i in range(n + 1):
        depth_of_wires[i] = 0

    gates = []
    next_wire = n + 1

    # Synthesize each basis vector with a balanced tree.
    basis_wire = []
    for i, (pb, mask) in enumerate(basis):
        terms = [j for j in range(n) if (mask >> j) & 1]
        if not terms:
            # basis vector is zero - shouldn't happen since we skip zero rows.
            basis_wire.append(n)  # constant zero
            continue
        # Reserve wire ids for this basis vector's internal gates.
        num_gates_needed = max(0, len(terms) - 1)
        start_wire = next_wire
        # We'll build the tree; final wire is the last created.
        # Use _build_balanced_xor which appends gates and returns final wire.
        # But it needs to know starting wire id = next_wire.
        # We must reserve enough depth_of_wires slots.
        final_wire = _build_balanced_xor(terms, start_wire, depth_of_wires, gates)
        next_wire = start_wire + num_gates_needed
        basis_wire.append(final_wire)

    # Now each output = XOR of basis wires where combo[i]==1.
    outputs = []
    for combo in row_basis_combo:
        sel = [basis_wire[i] for i in range(len(basis)) if combo[i]]
        if not sel:
            outputs.append(n)  # zero
            continue
        if len(sel) == 1:
            outputs.append(sel[0])
            continue
        # XOR selected basis wires (they are all already computed) with balanced tree.
        start_wire = next_wire
        final_wire = _build_balanced_xor(sel, start_wire, depth_of_wires, gates)
        next_wire = start_wire + max(0, len(sel) - 1)
        outputs.append(final_wire)

    # Verify depth <= depth_limit; if exceeded, fall back to independent-row synthesis.
    max_depth = max(depth_of_wires[v] for v in outputs) if outputs else 0
    if max_depth > depth_limit:
        # Fallback: independent rows, each a balanced tree over its input terms.
        return _fallback_independent(rows, depth_limit)

    return {'gates': gates, 'outputs': outputs}

def _fallback_independent(rows, depth_limit):
    m, n = rows.shape
    gates = []
    outputs = []
    depth_of_wires = [0] * (n + 1 + m * (n + 1))
    for i in range(n + 1):
        depth_of_wires[i] = 0
    next_wire = n + 1
    for i in range(m):
        terms = [j for j in range(n) if rows[i, j]]
        if not terms:
            outputs.append(n)
            continue
        if len(terms) == 1:
            outputs.append(terms[0])
            continue
        start_wire = next_wire
        final = _build_balanced_xor(terms, start_wire, depth_of_wires, gates)
        next_wire = start_wire + max(0, len(terms) - 1)
        outputs.append(final)
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
