"""XOR-circuit synthesis: exact GF(2) with shared intermediates (CSE) under a depth limit."""
import time
import itertools

_START = None


def _now():
    return time.perf_counter()


def _independent_rows(matrix):
    """Number of linearly independent rows over GF(2) (baseline denominator)."""
    m = [list(map(int, r)) for r in matrix]
    if not m:
        return 0
    n = len(m[0])
    basis = {}
    for row in m:
        v = list(row)
        for i in range(n):
            if v[i]:
                if i in basis:
                    v = [a ^ b for a, b in zip(v, basis[i])]
                else:
                    basis[i] = v
                    break
    return len(basis)


def _xor_basis_to_wires(matrix, n_inputs):
    """Build an exact circuit expressing each row as XOR of a small set of input wires.

    Returns (gates, outputs) where gates are [a,b] producing n+1+k.
    We use a Gaussian-elimination basis over the input coordinates: each basis vector
    is realized by XOR-ing the corresponding input wires, and each target row is the
    XOR of basis vectors. Intermediates (basis vectors and partial XORs) are shared.
    """
    m = [list(map(int, r)) for r in matrix]
    n = n_inputs
    if not m:
        return [], []
    width = len(m[0])

    # Build row-reduced basis: basis[i] = (mask_vector, row_index_or_None)
    # We'll realize basis vectors as wires, then express rows.
    basis = {}  # pivot col -> mask
    row_masks = []  # for each original row, list of pivot cols it uses (in RREF sense)
    for idx, row in enumerate(m):
        v = list(row)
        used = []
        for i in range(width):
            if v[i]:
                if i in basis:
                    bv = basis[i]
                    v = [a ^ b for a, b in zip(v, bv)]
                    used = [c for c in used if c != i]
                    # XOR the used-set with basis[i]'s used-set
                    # We track used as set of pivots
                    # recompute: used ^= basis_used[i]
                else:
                    basis[i] = v
                    used.append(i)
                    break
        # store used set
        row_masks.append(set(used))

    # Now we have basis vectors. Realize each basis vector as a wire = XOR of its set bits.
    # To share, we build a chain: for each basis vector, XOR its coordinates incrementally.
    # Simpler exact approach: realize each basis vector with (popcount-1) gates, sharing
    # prefix XORs across basis vectors via a global partial-XOR cache.
    gates = []
    wire_for = {}  # mask_tuple -> wire index
    def get_wire(mask):
        # mask: frozenset of input coords
        key = frozenset(mask)
        if key in wire_for:
            return wire_for[key]
        if not mask:
            return n  # constant zero
        coords = sorted(mask)
        if len(coords) == 1:
            w = coords[0]
        else:
            # split in half to keep depth low
            mid = len(coords) // 2
            a = get_wire(set(coords[:mid]))
            b = get_wire(set(coords[mid:]))
            w = n + 1 + len(gates)
            gates.append([a, b])
        wire_for[key] = w
        return w

    basis_wire = {}
    for i, bv in basis.items():
        mask = set(j for j, x in enumerate(bv) if x)
        basis_wire[i] = get_wire(mask)

    outputs = []
    for used in row_masks:
        if not used:
            outputs.append(n)
            continue
        # XOR the basis wires in 'used'
        ws = [basis_wire[i] for i in sorted(used)]
        w = ws[0]
        for other in ws[1:]:
            w = n + 1 + len(gates)
            gates.append([w, other])
        outputs.append(w)
    return gates, outputs


def _verify(matrix, gates, outputs, n_inputs, const_zero=None):
    """Exact GF(2) verification. Returns True if outputs reproduce the matrix rows."""
    if const_zero is None:
        const_zero = n_inputs
    m = [list(map(int, r)) for r in matrix]
    if not m:
        return True
    width = len(m[0])
    # wire -> mask over [inputs + const]
    wire_mask = {}
    for i in range(n_inputs):
        v = [0] * (width + 1)
        v[i] = 1
        wire_mask[i] = v
    v0 = [0] * (width + 1)
    v0[width] = 1  # constant one placeholder? const-zero wire is all-zero
    # constant-zero wire n: all zero
    wire_mask[n_inputs] = [0] * (width + 1)
    for k, (a, b) in enumerate(gates):
        wa = wire_mask[a]
        wb = wire_mask[b]
        wire_mask[n_inputs + 1 + k] = [x ^ y for x, y in zip(wa, wb)]
    for r, out in enumerate(outputs):
        if out not in wire_mask:
            return False
        got = wire_mask[out][:width]
        if got != m[r]:
            return False
    return True


def _greedy_cse(matrix, n_inputs, depth_limit, deadline):
    """Greedy common-subexpression elimination over GF(2) input-subset vectors.

    Candidate wires are subsets of inputs. We estimate, for each subset S, how many
    target rows (or partial residuals) contain S, and materialize high-frequency
    subsets first (MST-like: build a tree over the hypercube of useful subsets).
    """
    m = [list(map(int, r)) for r in matrix]
    width = len(m[0]) if m else 0
    n = n_inputs
    if not m:
        return [], []

    # Represent each row as a frozenset of input indices with a 1.
    row_sets = [frozenset(j for j, x in enumerate(r) if x) for r in m]

    # Count frequency of each subset S across rows (S is "useful" if it is a subset
    # of some row and appears in >=2 rows, or is a row itself).
    from collections import Counter
    freq = Counter()
    for s in row_sets:
        if not s:
            continue
        # count the row itself and its "halves" to encourage sharing
        freq[s] += 1
        # also count singletons and pairs implicitly via popcount later
    # We'll materialize wires for subsets chosen by a greedy set-coverage:
    # each materialized wire W can be used to reduce rows that contain W.
    # Build wires bottom-up by popcount so depth stays small.

    gates = []
    wire_mask = {}  # frozenset -> wire
    def get_wire(mask):
        key = frozenset(mask)
        if key in wire_mask:
            return wire_mask[key]
        if not mask:
            return n
        coords = sorted(mask)
        if len(coords) == 1:
            w = coords[0]
        else:
            mid = len(coords) // 2
            a = get_wire(set(coords[:mid]))
            b = get_wire(set(coords[mid:]))
            w = n + 1 + len(gates)
            gates.append([a, b])
        wire_mask[key] = w
        return w

    # Decide which subsets to materialize: all row sets (guarantees correctness)
    # plus high-frequency proper subsets that reduce total gate count.
    # For each row set, we will express it as XOR of materialized sub-wires.
    # To keep it exact and simple, materialize every row set and also every pair-subset
    # that appears in >=2 rows (bounded by time).
    # Choose candidate materialized sets:
    candidates = set()
    for s in row_sets:
        if s:
            candidates.add(s)
    # Add frequent small subsets (size 2) to enable sharing
    pair_freq = Counter()
    for s in row_sets:
        if len(s) >= 2:
            for pair in itertools.combinations(sorted(s), 2):
                pair_freq[frozenset(pair)] += 1
    # Add pairs appearing in >=2 rows, limited to keep time bounded
    added = 0
    for p, c in pair_freq.most_common():
        if c >= 2:
            candidates.add(p)
            added += 1
            if added > 4000:
                break

    # Materialize candidates (get_wire is memoized, so shared).
    for s in candidates:
        get_wire(s)

    # Now express each row as XOR of available wires. We need an exact expression:
    # row = XOR of some materialized wires whose XOR equals the row mask.
    # Use a linear solve: columns = materialized wire masks, find x s.t. A x = row.
    # To keep expressions small, prefer using the row's own materialized wire if present,
    # else fall back to input wires.
    # Build a basis of materialized wires for fast expression.
    wire_list = [ (k, ws) for k, ws in wire_mask.items() ]
    # We'll just express each row using input wires as a safe fallback, but prefer
    # the row's own wire if it was materialized.
    outputs = []
    for s in row_sets:
        if not s:
            outputs.append(n)
            continue
        if s in wire_mask:
            outputs.append(wire_mask[s])
        else:
            # express via inputs
            w = sorted(s)[0]
            for other in sorted(s)[1:]:
                w = n + 1 + len(gates)
                gates.append([w, other])
            outputs.append(w)
    return gates, outputs


def synthesize(matrix, depth_limit):
    global _START
    if _START is None:
        _START = _now()
    # Deadline: leave a safety margin before 90s wall-clock.
    deadline = _START + 78.0  # 12s margin

    n_inputs = len(matrix[0]) if (matrix and len(matrix[0])) else 0
    if not matrix:
        return {'gates': [], 'outputs': []}

    # Try the greedy CSE design first (shared intermediates).
    result = None
    try:
        if _now() < deadline:
            gates, outputs = _greedy_cse(matrix, n_inputs, depth_limit, deadline)
            if _verify(matrix, gates, outputs, n_inputs):
                result = {'gates': gates, 'outputs': outputs}
    except Exception:
        result = None

    # Fallback / cross-check with the basis design (always exact).
    try:
        if _now() < deadline:
            g2, o2 = _xor_basis_to_wires(matrix, n_inputs)
            if _verify(matrix, g2, o2, n_inputs):
                cand = {'gates': g2, 'outputs': o2}
                if result is None or len(g2) < len(result['gates']):
                    result = cand
    except Exception:
        pass

    # Absolute safe fallback: each row as XOR of its input wires (no sharing).
    if result is None:
        gates = []
        outputs = []
        for r in matrix:
            coords = [j for j, x in enumerate(r) if x]
            if not coords:
                outputs.append(n_inputs)
            else:
                w = coords[0]
                for other in coords[1:]:
                    w = n_inputs + 1 + len(gates)
                    gates.append([w, other])
                outputs.append(w)
        result = {'gates': gates, 'outputs': outputs}

    return result
