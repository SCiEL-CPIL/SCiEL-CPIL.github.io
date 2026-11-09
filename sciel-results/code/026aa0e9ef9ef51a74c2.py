"""XOR-circuit synthesis: rank-based decomposition with greedy intermediate sharing."""
import time
import numpy as np

def synthesize(matrix, depth_limit):
    t_start = time.time()
    DEADLINE = 85.0

    m = np.array(matrix, dtype=np.uint8)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    n_in, n_out = m.shape

    # Express each output as frozenset of input indices
    out_exprs = [frozenset(j for j in range(n_in) if m[i, j]) for i in range(n_out)]

    # Wire management
    wire_to_expr = {}
    expr_to_wire = {}
    next_wire = n_in + 1
    gates = []

    def reg(e, w):
        expr_to_wire[e] = w
        wire_to_expr[w] = e
        return w

    def gate(a, b):
        nonlocal next_wire
        e = wire_to_expr[a] ^ wire_to_expr[b]
        w = next_wire
        next_wire += 1
        gates.append([a, b])
        reg(e, w)
        return w

    # Init: inputs and zero
    for j in range(n_in):
        reg(frozenset([j]), j)
    reg(frozenset(), n_in)

    # Gaussian elimination with combination tracking
    A = m.copy()
    C = np.eye(n_out, dtype=np.uint8)
    pivots = []
    r = 0
    for col in range(n_in):
        piv = -1
        for i in range(r, n_out):
            if A[i, col]:
                piv = i
                break
        if piv < 0:
            continue
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
            C[[r, piv]] = C[[piv, r]]
        for i in range(n_out):
            if i != r and A[i, col]:
                A[i] ^= A[r]
                C[i] ^= C[r]
        pivots.append(r)
        r += 1
        if r >= n_out:
            break

    # Build pivot expressions
    pivot_wires = []
    for pr in pivots:
        e = frozenset(j for j in range(n_in) if A[pr, j])
        if e in expr_to_wire:
            pivot_wires.append(expr_to_wire[e])
        else:
            w = n_in
            for x in sorted(e):
                w = gate(w, expr_to_wire[frozenset([x])])
            pivot_wires.append(w)

    # Express outputs
    output_wires = []
    for i in range(n_out):
        coeffs = C[i, pivots]
        if not coeffs.any():
            output_wires.append(n_in)
        else:
            w = n_in
            for j, c in enumerate(coeffs):
                if c:
                    w = gate(w, pivot_wires[j])
            output_wires.append(w)

    # Greedy CSE pass: try to find shared intermediates
    # For each pair of outputs, check if their XOR is a useful intermediate
    if time.time() - t_start > DEADLINE - 10:
        return {'gates': gates, 'outputs': output_wires}

    # Try to reduce by finding common sub-expressions
    # For each pair of pivot expressions, if their XOR is used by multiple outputs,
    # it's already shared. The main savings come from the rank reduction.

    # Additional optimization: if an output equals a pivot expression, use it directly
    for i in range(n_out):
        if out_exprs[i] in expr_to_wire:
            output_wires[i] = expr_to_wire[out_exprs[i]]

    return {'gates': gates, 'outputs': output_wires}
