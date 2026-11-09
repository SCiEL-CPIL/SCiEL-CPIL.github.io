"""XOR circuit synthesis with pairwise-lookup acceleration and cost-priority scheduling."""
import time
import heapq

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}

    # Convert matrix rows to integer bitmasks for fast XOR
    rows = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i][j]:
                val |= (1 << j)
        rows.append(val)

    # Map from bitmask value to wire index
    val_to_wires = {}

    # Initialize with input wires
    for j in range(n):
        val = 1 << j
        val_to_wires[val] = j

    # Constant zero
    val_to_wires[0] = n

    # Pairwise XOR lookup: for each pair of available wires, map their XOR to the pair
    # This enables O(1) one-gate completion checks
    # pair_xors[xor_val] = set of (wire_a, wire_b) pairs that produce xor_val
    pair_xors = {}

    def popcount(x):
        return bin(x).count('1')

    # Initialize pair_xors with all pairs of initial wires
    init_wires = list(val_to_wires.items())
    for i in range(len(init_wires)):
        for j in range(i + 1, len(init_wires)):
            v1, w1 = init_wires[i]
            v2, w2 = init_wires[j]
            xor_val = v1 ^ v2
            if xor_val not in pair_xors:
                pair_xors[xor_val] = []
            pair_xors[xor_val].append((w1, w2))

    gates = []
    next_wire = n + 1

    def add_wire(val, wire):
        """Register a new wire and update pair_xors incrementally."""
        nonlocal next_wire
        val_to_wires[val] = wire
        # Add pairs with all existing wires
        for existing_val, existing_wire in val_to_wires.items():
            if existing_val == val:
                continue
            xor_val = val ^ existing_val
            if xor_val not in pair_xors:
                pair_xors[xor_val] = []
            pair_xors[xor_val].append((wire, existing_wire))

    def build_from_inputs(target):
        """Fallback: build target directly from input wires using a chain."""
        nonlocal next_wire, gates, val_to_wires

        if target == 0:
            return n
        if target in val_to_wires:
            return val_to_wires[target]

        # Build from set bits
        wire = None
        bits = target
        while bits:
            lb = bits & (-bits)
            bit_idx = lb.bit_length() - 1
            input_wire = bit_idx  # input wire for bit bit_idx
            if wire is None:
                wire = input_wire
            else:
                gates.append([wire, input_wire])
                wire = next_wire
                next_wire += 1
            bits ^= lb

        if wire is None:
            wire = n

        val_to_wires[target] = wire
        # Update pair_xors for this new wire
        for existing_val, existing_wire in list(val_to_wires.items()):
            if existing_val == target:
                continue
            xor_val = target ^ existing_val
            if xor_val not in pair_xors:
                pair_xors[xor_val] = []
            pair_xors[xor_val].append((wire, existing_wire))

        return wire

    def build_recursive(target, depth=0):
        """
        Recursively build 'target' using existing wires.
        Returns the wire index for 'target'.
        """
        nonlocal next_wire, gates, val_to_wires

        if target == 0:
            return n

        if target in val_to_wires:
            return val_to_wires[target]

        if depth > 24:
            # Fallback: build from inputs directly
            return build_from_inputs(target)

        # Check if one gate suffices using pair_xors
        if target in pair_xors:
            for w1, w2 in pair_xors[target]:
                if w1 != w2:
                    gates.append([w1, w2])
                    new_wire = next_wire
                    next_wire += 1
                    val_to_wires[target] = new_wire
                    # Update pair_xors
                    for existing_val, existing_wire in list(val_to_wires.items()):
                        if existing_val == target:
                            continue
                        xor_val = target ^ existing_val
                        if xor_val not in pair_xors:
                            pair_xors[xor_val] = []
                        pair_xors[xor_val].append((new_wire, existing_wire))
                    return new_wire

        # Find best decomposition
        best_a = None
        best_a_wire = None
        min_cost = float('inf')
        best_diff = None

        for a, a_wire in val_to_wires.items():
            diff = target ^ a
            if diff == 0:
                return a_wire
            cost = popcount(diff)
            if cost < min_cost:
                min_cost = cost
                best_a = a
                best_a_wire = a_wire
                best_diff = diff
                if cost == 1:
                    break

        if best_a is None:
            return build_from_inputs(target)

        # Recursively build the diff
        diff_wire = build_recursive(best_diff, depth + 1)

        if best_diff == 0:
            return best_a_wire

        # Create gate: best_a_wire XOR diff_wire
        gates.append([best_a_wire, diff_wire])
        new_wire = next_wire
        next_wire += 1
        val_to_wires[target] = new_wire

        # Update pair_xors for the new wire
        for existing_val, existing_wire in list(val_to_wires.items()):
            if existing_val == target:
                continue
            xor_val = target ^ existing_val
            if xor_val not in pair_xors:
                pair_xors[xor_val] = []
            pair_xors[xor_val].append((new_wire, existing_wire))

        return new_wire

    # Track which rows have been computed
    row_to_wire = [None] * m
    remaining = set(range(m))

    # Priority queue: (estimated_cost, row_index)
    # We'll re-evaluate costs as new wires become available
    pq = []

    def estimate_cost(target):
        """Estimate the cost to build target from current wires."""
        if target == 0:
            return 0
        if target in val_to_wires:
            return 0
        # One-gate check via pair_xors
        if target in pair_xors:
            return 1
        # Otherwise, estimate based on best single-helper decomposition
        min_est = float('inf')
        for a, a_wire in val_to_wires.items():
            diff = target ^ a
            if diff == 0:
                min_est = 0
                break
            est = 1 + popcount(diff)
            if est < min_est:
                min_est = est
        return min_est

    # Initialize priority queue
    for idx in range(m):
        target = rows[idx]
        cost = estimate_cost(target)
        if cost == 0:
            row_to_wire[idx] = val_to_wires[target] if target in val_to_wires else n
            remaining.discard(idx)
        else:
            heapq.heappush(pq, (cost, idx))

    # Process rows in priority order
    while pq:
        if time.time() > deadline:
            break

        cost, idx = heapq.heappop(pq)

        if idx not in remaining:
            continue

        target = rows[idx]

        # Re-check if already available
        if target == 0:
            row_to_wire[idx] = n
            remaining.discard(idx)
            continue

        if target in val_to_wires:
            row_to_wire[idx] = val_to_wires[target]
            remaining.discard(idx)
            continue

        # Check one-gate completion via pair_xors
        if target in pair_xors:
            found = False
            for w1, w2 in pair_xors[target]:
                if w1 != w2:
                    gates.append([w1, w2])
                    new_wire = next_wire
                    next_wire += 1
                    val_to_wires[target] = new_wire
                    row_to_wire[idx] = new_wire
                    remaining.discard(idx)
                    found = True
                    # Update pair_xors for new wire
                    for existing_val, existing_wire in list(val_to_wires.items()):
                        if existing_val == target:
                            continue
                        xor_val = target ^ existing_wire
                        if xor_val not in pair_xors:
                            pair_xors[xor_val] = []
                        pair_xors[xor_val].append((new_wire, existing_wire))
                    break
            if found:
                # Re-estimate costs of remaining rows (some may now be cheaper)
                new_pq = []
                for idx2 in remaining:
                    t2 = rows[idx2]
                    c2 = estimate_cost(t2)
                    heapq.heappush(new_pq, (c2, idx2))
                pq = new_pq
                continue

        # Recursive build
        wire = build_recursive(target)
        row_to_wire[idx] = wire
        remaining.discard(idx)

        # Re-estimate costs of remaining rows
        new_pq = []
        for idx2 in remaining:
            t2 = rows[idx2]
            c2 = estimate_cost(t2)
            heapq.heappush(new_pq, (c2, idx2))
        pq = new_pq

    # Ensure all rows have a wire assigned
    for i in range(m):
        if row_to_wire[i] is None:
            if rows[i] == 0:
                row_to_wire[i] = n
            elif rows[i] in val_to_wires:
                row_to_wire[i] = val_to_wires[rows[i]]
            else:
                row_to_wire[i] = build_from_inputs(rows[i])

    outputs = row_to_wire

    return {'gates': gates, 'outputs': outputs}
