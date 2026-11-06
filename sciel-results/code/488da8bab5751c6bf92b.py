# EVOLVE-BLOCK-START
import time
import heapq

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 55.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Wire assignment: inputs 0..n-1, const-zero n, gates n+1...
    next_wire = n + 1
    gates = []

    # memo: bitmask -> wire_id
    memo = {}
    # Initialize with inputs
    for j in range(n):
        memo[1 << j] = j
    # Const zero
    memo[0] = n

    def elapsed():
        return time.time() - start_time

    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w

    # Parse matrix to masks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Collect all unique non-zero masks that need to be synthesized
    all_masks = set(row_masks)
    all_masks.discard(0)

    # Greedy basis construction:
    # We want to find a set of intermediate XOR masks (the "basis") such that
    # every target mask can be expressed as XOR of some subset of basis masks.
    # The cost of synthesizing a mask of popcount k using a basis is:
    #   cost = popcount(mask) - sum(popcount(b) for b in basis_used) + len(basis_used)
    # because each basis element of popcount b saves (b-1) gates compared to direct XOR.
    #
    # Greedy: repeatedly pick the intermediate mask that maximizes total savings
    # across all remaining unsatisfied targets, where a target "uses" an intermediate
    # if XORing it reduces the target's popcount.

    # State: for each target, track its current residual mask (what's left to synthesize)
    # Initially, residual = target itself.
    # When we add an intermediate mask I, for any target t where (t_residual XOR I) has
    # smaller popcount than t_residual, we can "use" I to reduce t_residual.

    # More precisely: we build a set of "available" masks (inputs + constructed gates).
    # For each target, we want to express it as XOR of available masks.
    # The number of gates needed for target t = (number of available masks used in its expression) - 1
    # if we use a tree, or equivalently, the total cost is sum over targets of (count_used - 1).
    # But with memoization, shared sub-expressions are built once.
    
    # Alternative formulation: 
    # Total gates = (number of distinct intermediate masks we construct) 
    #   + sum over targets of (number of basis elements in target's expression - 1)
    # Wait, that's not quite right either. Let me think more carefully.
    
    # Each gate computes XOR of two existing wires. If we have a set of "primitive" wires
    # (inputs and constructed intermediates), and each target is expressed as XOR of some
    # subset of these primitives, then:
    # - Each primitive that is an input costs 0 gates.
    # - Each primitive that is a constructed gate costs 1 gate (and its two operands must
    #   already be available).
    # - Each target that uses k primitives costs (k-1) gates to combine them (if k>1),
    #   or 0 gates if k=1 (just use the wire directly), or 0 if k=0 (constant zero).
    #
    # But this double-counts: the gates to build primitives are already counted.
    # Actually, the total gate count = (gates to build all intermediate primitives) 
    #   + (gates to combine primitives for each target).
    #
    # For a target expressed as XOR of primitives p1, p2, ..., pk:
    #   If k == 0: cost 0 (constant zero)
    #   If k == 1: cost 0 (just use the wire)
    #   If k >= 2: cost k-1 (XOR tree)
    #
    # Total = sum_over_intermediates(1) + sum_over_targets(max(0, k_t - 1))
    # where k_t is the number of primitives used for target t.
    
    # Greedy strategy:
    # Start with primitives = {all input bits}. 
    # Repeatedly add the intermediate mask I that minimizes the increase in total cost.
    # Adding I costs 1 gate to construct I (from two existing primitives).
    # For each target t, if using I in t's expression reduces k_t, the savings is the
    # reduction in (k_t - 1).
    
    # This is complex. Let me use a simpler but effective approach:
    # Build a basis via Gaussian elimination, then for each row, find the minimum-weight
    # combination of basis vectors that equals the row. Use memoization for shared sub-expressions.

    # Actually, let me try a cleaner greedy:
    # 1. Compute the row space basis via Gaussian elimination.
    # 2. For each basis vector, synthesize it using a balanced tree.
    # 3. For each target row, express it as XOR of basis vectors, then combine those
    #    basis wires.
    # 4. Use memoization so that if two targets share the same combination of basis
    #    vectors, they share the combination wire.

    # Step 1: Gaussian elimination to get RREF basis
    # Work with the row_masks as vectors over GF(2)
    
    # Convert to list of lists for manipulation
    rows = []
    for mask in row_masks:
        row = [(mask >> j) & 1 for j in range(n)]
        rows.append(row)
    
    # Gaussian elimination to RREF
    rref = []
    pivot_cols = []
    for col in range(n):
        # Find pivot row
        pivot_row = -1
        for i in range(len(rref), len(rows)):
            if rows[i][col] == 1:
                pivot_row = i
                break
        if pivot_row == -1:
            continue
        
        # Swap to position len(rref)
        if pivot_row != len(rref):
            rows[len(rref)], rows[pivot_row] = rows[pivot_row], rows[len(rref)]
        
        # Eliminate col from other rows
        for i in range(len(rows)):
            if i != len(rref) and rows[i][col] == 1:
                for j in range(col, n):
                    rows[i][j] ^= rows[len(rref)][j]
        
        rref.append(rows[len(rref)])
        pivot_cols.append(col)
    
    # Now rref has rank r rows, each with a unique pivot column.
    # Each original row can be expressed as XOR of some subset of rref rows.
    # Let's compute the coefficients for each original row.
    
    # Actually, let me redo this more carefully. I'll track the original row index
    # and compute the expression of each original row in terms of the RREF basis.
    
    # Let me use a different approach: work directly with masks.
    
    # Compute rank and basis via Gaussian elimination on masks
    basis = []  # list of (pivot_col, mask)
    for mask in row_masks:
        m_cur = mask
        for pivot_col, b_mask in basis:
            if (m_cur >> pivot_col) & 1:
                m_cur ^= b_mask
        if m_cur != 0:
            # Find the highest set bit as pivot
            pivot = m_cur.bit_length() - 1
            # Eliminate this pivot from existing basis
            new_basis = []
            for pc, bm in basis:
                if (bm >> pivot) & 1:
                    bm ^= m_cur
                new_basis.append((pc, bm))
            new_basis.append((pivot, m_cur))
            basis = new_basis
    
    # Now basis has rank r elements, each with a unique pivot column.
    # For each original row, express it as XOR of basis elements.
    row_expressions = []  # list of sets of basis indices
    for mask in row_masks:
        m_cur = mask
        expr = set()
        for idx in range(len(basis) - 1, -1, -1):
            pivot_col, b_mask = basis[idx]
            if (m_cur >> pivot_col) & 1:
                m_cur ^= b_mask
                expr.add(idx)
        # m_cur should be 0 now
        row_expressions.append(frozenset(expr))
    
    # Now we need to:
    # 1. Synthesize each basis vector.
    # 2. For each unique expression (frozenset of basis indices), synthesize the XOR
    #    of those basis wires.
    # 3. Map each row to its expression's wire.
    
    # Step 1: Synthesize basis vectors using balanced tree with memoization
    basis_masks = [bm for _, bm in basis]
    
    # Collect all sub-masks needed for basis synthesis
    needed = set()
    def collect_needed_balanced(mask):
        if mask in memo or mask == 0:
            return
        if mask in needed:
            return
        needed.add(mask)
        bits = [j for j in range(n) if (mask >> j) & 1]
        if len(bits) <= 1:
            return
        mid = len(bits) // 2
        sub1 = sum(1 << b for b in bits[:mid])
        sub2 = sum(1 << b for b in bits[mid:])
        collect_needed_balanced(sub1)
        collect_needed_balanced(sub2)
    
    for bm in basis_masks:
        collect_needed_balanced(bm)
    
    # Sort by popcount and synthesize
    sorted_needed = sorted(needed, key=lambda x: x.bit_count())
    for mask in sorted_needed:
        if mask in memo:
            continue
        bits = [j for j in range(n) if (mask >> j) & 1]
        if len(bits) == 0:
            memo[mask] = n
        elif len(bits) == 1:
            memo[mask] = bits[0]
        else:
            mid = len(bits) // 2
            sub1 = sum(1 << b for b in bits[:mid])
            sub2 = sum(1 << b for b in bits[mid:])
            if sub1 not in memo or sub2 not in memo:
                continue
            w = add_gate(memo[sub1], memo[sub2])
            memo[mask] = w
    
    # Now basis wires are in memo
    basis_wires = [memo[bm] for bm in basis_masks]
    
    # Step 2: For each unique expression, synthesize the XOR of the corresponding basis wires
    # An expression is a frozenset of basis indices.
    # We need to XOR the wires for those indices.
    # Use memoization: expr_wire_cache: frozenset -> wire
    expr_wire_cache = {}
    expr_wire_cache[frozenset()] = n  # empty set = constant zero
    
    # For singleton expressions, just use the basis wire directly
    for idx in range(len(basis_masks)):
        expr_wire_cache[frozenset([idx])] = basis_wires[idx]
    
    # For larger expressions, we need to combine.
    # Process expressions in order of increasing size.
    unique_exprs = list(set(row_expressions))
    unique_exprs.sort(key=lambda e: len(e))
    
    for expr in unique_exprs:
        if expr in expr_wire_cache:
            continue
        if len(expr) == 0:
            expr_wire_cache[expr] = n
            continue
        if len(expr) == 1:
            idx = list(expr)[0]
            expr_wire_cache[expr] = basis_wires[idx]
            continue
        
        # Split the expression into two halves
        idxs = sorted(expr)
        mid = len(idxs) // 2
        sub_expr1 = frozenset(idxs[:mid])
        sub_expr2 = frozenset(idxs[mid:])
        
        # These should already be in cache (since they're smaller)
        if sub_expr1 not in expr_wire_cache or sub_expr2 not in expr_wire_cache:
            # Fallback: just XOR all wires one by one
            w = basis_wires[idxs[0]]
            for i in range(1, len(idxs)):
                w = add_gate(w, basis_wires[idxs[i]])
            expr_wire_cache[expr] = w
            continue
        
        w = add_gate(expr_wire_cache[sub_expr1], expr_wire_cache[sub_expr2])
        expr_wire_cache[expr] = w
    
    # Step 3: Map each row to its output wire
    outputs = []
    for expr in row_expressions:
        outputs.append(expr_wire_cache[expr])
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
