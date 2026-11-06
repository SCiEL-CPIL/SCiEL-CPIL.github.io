"""XOR-circuit synthesis: global tree-based synthesis with sub-expression sharing."""
# EVOLVE-BLOCK-START
import time
import math

_DEADLINE = 85.0  # safety margin below 90s wall-clock kill
_START = None


def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START


def _bits_to_int(bits):
    """Convert a sequence of bits to an integer (bit i of bits -> bit i of int)."""
    v = 0
    for i, b in enumerate(bits):
        if b:
            v |= (1 << i)
    return v


def _int_to_bits(v, n):
    """Convert int to list of n bits."""
    return [(v >> i) & 1 for i in range(n)]


def _xor(a, b):
    return a ^ b


def _popcount(v):
    return bin(v).count('1')


def _tree_cost(n):
    """Cost (gate count) to compute XOR of n inputs using a balanced tree."""
    if n <= 1:
        return 0
    return n - 1


def _balanced_tree_gates(inputs, base_wire):
    """
    Produce a balanced XOR tree for a list of input wires.
    Returns (list_of_gates, output_wire).
    gates is list of [w1, w2] (each gate computes w1 XOR w2).
    base_wire: next free wire index to start allocating.
    """
    if len(inputs) == 0:
        return [], None
    if len(inputs) == 1:
        return [], inputs[0]

    gates = []
    wire = base_wire
    level = list(inputs)
    while len(level) > 1:
        next_level = []
        # Pair up; if odd, carry last one
        i = 0
        while i + 1 < len(level):
            gates.append([level[i], level[i + 1]])
            next_level.append(wire)
            wire += 1
            i += 2
        if i < len(level):
            next_level.append(level[i])
        level = next_level
    return gates, level[0]


def _tree_cost_with_depth(n):
    """Return (gate_count, depth) for balanced XOR tree of n inputs."""
    if n <= 1:
        return 0, 0
    gates = n - 1
    depth = 0
    m = n
    while m > 1:
        m = (m + 1) // 2
        depth += 1
    return gates, depth


def _build_shared_tree(rows_bits, n_cols):
    """
    Build a global XOR circuit for a list of row-vectors (each a list of 0/1 of length n_cols).
    Strategy:
      1. For each row, compute the set of columns that are 1.
      2. Use a shared "column tree" approach: build a balanced tree over each column's
         set of rows that use it, then combine.
    
    Actually, a cleaner global approach:
      - Represent each output as a linear combination of input columns.
      - Use a shared intermediate-wire pool: for each column c, we may need to XOR it
        into multiple outputs. Instead of duplicating, we build a tree per output but
        share sub-expressions when two outputs share a large subset.
    
    A practical global strategy that maximizes sharing:
      - Build a single "column aggregation" tree: for each column, create a chain/tree
        of XORs so that the column's value is available at a shared wire, then each
        output XORs the relevant shared column-wires together.
    
    But columns are inputs (free wires), so XORing a column into multiple outputs
    just means each output references the same input wire — no gate needed.
    
    The real sharing opportunity: intermediate XOR results. If two outputs both need
    (a XOR b), we compute it once.
    
    Global memoized approach:
      - Define a recursive function f(S) where S is a frozenset of column indices.
        f(S) returns the wire that computes XOR of columns in S.
      - Memoize: same set S -> same wire.
      - For a set S of size k, split into S1, S2 (balanced), compute f(S1), f(S2),
        then one gate combines them.
      - Base: |S|==0 -> constant-zero wire (n); |S|==1 -> the column wire itself.
    
    This maximizes sharing: any sub-subset that appears in multiple rows is computed once.
    """
    n = n_cols  # input wires are 0..n-1, constant-zero is n
    memo = {}  # frozenset -> wire
    gates = []
    next_wire = n + 1  # gate 0 produces wire n+1

    def get_wire(S):
        """Return the wire that computes XOR of columns in S (S is a frozenset of ints)."""
        if len(S) == 0:
            return n  # constant-zero wire
        if len(S) == 1:
            return next(iter(S))
        if S in memo:
            return memo[S]
        # Split S into two roughly equal halves
        cols = sorted(S)
        mid = len(cols) // 2
        S1 = frozenset(cols[:mid])
        S2 = frozenset(cols[mid:])
        w1 = get_wire(S1)
        w2 = get_wire(S2)
        # One gate: w1 XOR w2
        w = next_wire
        next_wire += 1
        gates.append([w1, w2])
        memo[S] = w
        return w

    # Compute wire for each row
    output_wires = []
    for row in rows_bits:
        S = frozenset(i for i, b in enumerate(row) if b)
        output_wires.append(get_wire(S))

    return gates, output_wires


def _build_shared_tree_with_depth(rows_bits, n_cols):
    """
    Same as _build_shared_tree but tracks depth.
    Returns (gates, output_wires, max_depth).
    """
    n = n_cols
    memo = {}  # frozenset -> (wire, depth)
    gates = []
    next_wire = n + 1

    def get_wire(S):
        if len(S) == 0:
            return (n, 0)
        if len(S) == 1:
            return (next(iter(S)), 0)
        if S in memo:
            return memo[S]
        cols = sorted(S)
        mid = len(cols) // 2
        S1 = frozenset(cols[:mid])
        S2 = frozenset(cols[mid:])
        w1, d1 = get_wire(S1)
        w2, d2 = get_wire(S2)
        w = next_wire
        next_wire += 1
        gates.append([w1, w2])
        d = max(d1, d2) + 1
        memo[S] = (w, d)
        return (w, d)

    output_wires = []
    max_depth = 0
    for row in rows_bits:
        S = frozenset(i for i, b in enumerate(row) if b)
        w, d = get_wire(S)
        output_wires.append(w)
        max_depth = max(max_depth, d)

    return gates, output_wires, max_depth


def _verify(matrix, gates, outputs, n_cols, n_rows):
    """
    Exact GF(2) verification.
    matrix: n_rows x n_cols list of lists of 0/1.
    gates: list of [w1, w2].
    outputs: list of wire indices.
    Returns True if valid.
    """
    # Track the set of input columns each wire represents (as a frozenset).
    wire_set = {}
    for i in range(n_cols):
        wire_set[i] = frozenset([i])
    # Constant-zero wire n
    wire_set[n_cols] = frozenset()

    for k, (w1, w2) in enumerate(gates):
        w = n_cols + 1 + k
        s1 = wire_set.get(w1, frozenset())
        s2 = wire_set.get(w2, frozenset())
        wire_set[w] = s1 ^ s2  # symmetric difference

    for r in range(n_rows):
        expected = frozenset(i for i, b in enumerate(matrix[r]) if b)
        actual = wire_set.get(outputs[r], frozenset())
        if expected != actual:
            return False
    return True


def _count_independent_rows(matrix):
    """Count the rank of the matrix over GF(2) (number of independent rows)."""
    rows = [list(r) for r in matrix]
    n_rows = len(rows)
    n_cols = len(rows[0]) if n_rows > 0 else 0
    if n_cols == 0:
        return 0

    # Gaussian elimination to find rank
    rank = 0
    col = 0
    # Work on a copy
    mat = [row[:] for row in rows]
    for c in range(n_cols):
        # Find pivot
        pivot = -1
        for r in range(rank, n_rows):
            if mat[r][c]:
                pivot = r
                break
        if pivot == -1:
            continue
        # Swap
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        # Eliminate
        for r in range(n_rows):
            if r != rank and mat[r][c]:
                for cc in range(c, n_cols):
                    mat[r][cc] ^= mat[rank][cc]
        rank += 1
    return rank


def _baseline_gate_count(matrix):
    """
    Independent-row baseline: for each independent row, compute XOR of its 1-columns
    using a balanced tree. Total gates = sum over independent rows of (weight - 1).
    """
    # First, find a basis of independent rows
    rows = [list(r) for r in matrix]
    n_rows = len(rows)
    n_cols = len(rows[0]) if n_rows > 0 else 0
    if n_cols == 0:
        return 0

    mat = [row[:] for row in rows]
    basis_indices = []
    rank = 0
    for c in range(n_cols):
        pivot = -1
        for r in range(rank, n_rows):
            if mat[r][c]:
                pivot = r
                break
        if pivot == -1:
            continue
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        for r in range(n_rows):
            if r != rank and mat[r][c]:
                for cc in range(c, n_cols):
                    mat[r][cc] ^= mat[rank][cc]
        basis_indices.append(rank)
        rank += 1

    # For each basis row, count gates = popcount - 1 (balanced tree cost)
    total = 0
    for idx in basis_indices:
        w = sum(mat[idx])
        if w > 1:
            total += w - 1
    return total


def synthesize(matrix, depth_limit):
    """
    Synthesize an exact XOR circuit for the given binary matrix.
    
    matrix: list of lists of 0/1, shape (n_rows, n_cols).
    depth_limit: maximum allowed circuit depth.
    
    Returns: {'gates': [[wire, wire], ...], 'outputs': [wire, ...]}
    """
    global _START
    if _START is None:
        _START = time.time()

    n_rows = len(matrix)
    n_cols = len(matrix[0]) if n_rows > 0 else 0

    if n_rows == 0:
        return {'gates': [], 'outputs': []}

    # Trivial: all-zero matrix
    all_zero = all(all(b == 0 for b in row) for row in matrix)
    if all_zero:
        zero_wire = n_cols
        return {'gates': [], 'outputs': [zero_wire] * n_rows}

    # Primary strategy: global memoized balanced-tree synthesis
    gates, outputs = _build_shared_tree(matrix, n_cols)

    # Check depth
    _, _, max_depth = _build_shared_tree_with_depth(matrix, n_cols)
    if max_depth > depth_limit:
        # Fallback: per-row balanced trees (may exceed depth limit but is valid)
        # Actually, let's try to reduce depth by using a different split strategy
        # For now, use per-row trees which have depth = ceil(log2(weight))
        gates2 = []
        outputs2 = []
        next_wire = n_cols + 1
        for row in matrix:
            ones = [i for i, b in enumerate(row) if b]
            if len(ones) == 0:
                outputs2.append(n_cols)
                continue
            if len(ones) == 1:
                outputs2.append(ones[0])
                continue
            row_gates, out_wire = _balanced_tree_gates(ones, next_wire)
            gates2.extend(row_gates)
            next_wire += len(row_gates)
            outputs2.append(out_wire)
        gates, outputs = gates2, outputs2

    # Verify
    if not _verify(matrix, gates, outputs, n_cols, n_rows):
        # Should not happen, but fallback to per-row
        gates2 = []
        outputs2 = []
        next_wire = n_cols + 1
        for row in matrix:
            ones = [i for i, b in enumerate(row) if b]
            if len(ones) == 0:
                outputs2.append(n_cols)
                continue
            if len(ones) == 1:
                outputs2.append(ones[0])
                continue
            row_gates, out_wire = _balanced_tree_gates(ones, next_wire)
            gates2.extend(row_gates)
            next_wire += len(row_gates)
            outputs2.append(out_wire)
        gates, outputs = gates2, outputs2

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
