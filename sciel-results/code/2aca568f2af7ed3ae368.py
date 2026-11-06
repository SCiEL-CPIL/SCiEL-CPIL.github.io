import time
from itertools import combinations

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 55.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Parse matrix to masks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Build frequency map
    freq = {}
    for mask in row_masks:
        freq[mask] = freq.get(mask, 0) + 1

    # Zero wire
    zero_wire = n

    # Gate list
    gates = []
    next_wire = n + 1
    memo = {}  # mask -> wire

    # Initialize memo with unit vectors
    for j in range(n):
        memo[1 << j] = j
    memo[0] = zero_wire

    def elapsed():
        return time.time() - start_time

    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w

    def popcount(x):
        return x.bit_count()

    # Collect all needed masks: for each unique row mask, collect all submasks
    # that would appear in a balanced decomposition
    needed = {}  # mask -> total frequency (weighted by row frequency)

    def collect_decomp(mask, weight):
        """Recursively collect masks from balanced decomposition"""
        if mask == 0:
            return
        needed[mask] = needed.get(mask, 0) + weight
        if mask in memo:
            return
        bits = [j for j in range(n) if (mask >> j) & 1]
        if len(bits) <= 1:
            return
        mid = len(bits) // 2
        sub1 = 0
        sub2 = 0
        for i, b in enumerate(bits):
            if i < mid:
                sub1 |= (1 << b)
            else:
                sub2 |= (1 << b)
        if sub1 != 0:
            collect_decomp(sub1, weight)
        if sub2 != 0:
            collect_decomp(sub2, weight)

    for mask, w in freq.items():
        if mask != 0:
            collect_decomp(mask, w)

    # Sort by (popcount, -frequency) to synthesize small, high-frequency masks first
    sorted_needed = sorted(needed.keys(), key=lambda x: (x.bit_count(), -needed[x], x))

    # Synthesize each needed mask
    for mask in sorted_needed:
        if elapsed() > DEADLINE:
            break
        if mask in memo:
            continue

        bits = [j for j in range(n) if (mask >> j) & 1]
        if len(bits) == 0:
            memo[mask] = zero_wire
        elif len(bits) == 1:
            memo[mask] = bits[0]
        else:
            # Try to find the best split
            best_cost = len(bits) - 1  # fallback: chain
            best_split = None

            # Strategy 1: balanced split
            mid = len(bits) // 2
            sub1 = 0
            sub2 = 0
            for i, b in enumerate(bits):
                if i < mid:
                    sub1 |= (1 << b)
                else:
                    sub2 |= (1 << b)

            if sub1 in memo and sub2 in memo:
                best_cost = 1
                best_split = (sub1, sub2)
            elif sub1 in memo:
                cost = 1 + (sub2.bit_count() - 1) if sub2.bit_count() > 1 else 1
                if cost < best_cost:
                    best_cost = cost
                    best_split = (sub1, sub2)
            elif sub2 in memo:
                cost = 1 + (sub1.bit_count() - 1) if sub1.bit_count() > 1 else 1
                if cost < best_cost:
                    best_cost = cost
                    best_split = (sub1, sub2)

            # Strategy 2: try splits where one part is in memo
            # For efficiency, only try small subsets
            if len(bits) <= 10:
                max_subset = min(4, len(bits) // 2)
                for size in range(1, max_subset + 1):
                    if elapsed() > DEADLINE:
                        break
                    for combo in combinations(bits, size):
                        sub1 = 0
                        for b in combo:
                            sub1 |= (1 << b)
                        sub2 = mask ^ sub1
                        if sub2 == 0:
                            continue
                        # Check if either part is in memo
                        if sub1 in memo and sub2 in memo:
                            if 1 < best_cost:
                                best_cost = 1
                                best_split = (sub1, sub2)
                        elif sub1 in memo:
                            cost2 = sub2.bit_count() - 1 if sub2.bit_count() > 1 else 0
                            cost = 1 + cost2
                            if cost < best_cost:
                                best_cost = cost
                                best_split = (sub1, sub2)
                        elif sub2 in memo:
                            cost1 = sub1.bit_count() - 1 if sub1.bit_count() > 1 else 0
                            cost = 1 + cost1
                            if cost < best_cost:
                                best_cost = cost
                                best_split = (sub1, sub2)

            # Execute the best split
            if best_split is not None and best_cost < len(bits) - 1:
                sub1, sub2 = best_split
                if sub1 in memo and sub2 in memo:
                    memo[mask] = add_gate(memo[sub1], memo[sub2])
                elif sub1 in memo:
                    # Need to build sub2
                    b2 = [j for j in range(n) if (sub2 >> j) & 1]
                    w = memo[sub1]
                    for b in b2:
                        w = add_gate(w, b)
                    memo[mask] = w
                elif sub2 in memo:
                    b1 = [j for j in range(n) if (sub1 >> j) & 1]
                    w = memo[sub2]
                    for b in b1:
                        w = add_gate(w, b)
                    memo[mask] = w
            else:
                # Fallback: balanced tree
                def build_tree(b_list):
                    if len(b_list) == 1:
                        return b_list[0]
                    mid = len(b_list) // 2
                    left = build_tree(b_list[:mid])
                    right = build_tree(b_list[mid:])
                    return add_gate(left, right)
                memo[mask] = build_tree(bits)

    # Now try to optimize: for each row mask, see if we can express it more cheaply
    # using existing memo entries
    # Key insight: if mask = a ^ b where a, b are in memo, we might use 1 gate instead of the current expression
    # But the current expression IS 1 gate (from add_gate), so we can't improve on that.
    # However, if the current expression used multiple gates (fallback), we might improve.

    # Actually, let's try a different optimization:
    # For each row mask, try to find if it can be expressed as XOR of 2 existing memo terms
    # This would give cost 1, which is optimal for a single output.
    # But if it's already 1 gate, no improvement.

    # Let's try to reduce total gates by re-expressing some masks.
    # For masks that currently cost > 1 gate, see if they can be done in 1.

    # Count current gate usage per mask
    # Actually, this is complex. Let's just ensure all outputs are valid.

    # Verify and output
    outputs = []
    for mask in row_masks:
        if mask == 0:
            outputs.append(zero_wire)
        elif mask in memo:
            outputs.append(memo[mask])
        else:
            # Should not happen, but fallback
            bits = [j for j in range(n) if (mask >> j) & 1]
            if len(bits) == 0:
                outputs.append(zero_wire)
            elif len(bits) == 1:
                outputs.append(bits[0])
            else:
                w = bits[0]
                for b in bits[1:]:
                    w = add_gate(w, b)
                outputs.append(w)

    return {'gates': gates, 'outputs': outputs}
