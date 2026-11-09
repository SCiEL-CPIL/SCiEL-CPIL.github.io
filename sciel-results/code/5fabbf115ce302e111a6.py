"""XOR circuit synthesis with basis-aware greedy minimization and CSE."""
import time
import heapq

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}

    # Convert matrix rows to integer bitmasks
    rows = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i][j]:
                val |= (1 << j)
        rows.append(val)

    def popcount(x):
        return bin(x).count('1')

    # State: wires and their values
    val_to_wire = {0: n}  # constant zero
    for j in range(n):
        val_to_wire[1 << j] = j

    gates = []
    next_wire = n + 1

    def register_wire(val, wire):
        """Register a new wire value."""
        nonlocal next_wire
        if val in val_to_wire:
            return val_to_wire[val]
        val_to_wire[val] = wire
        return wire

    def build_from_inputs(target):
        """Build target from input wires using minimal gates."""
        nonlocal next_wire, gates
        if target == 0:
            return n
        if target in val_to_wire:
            return val_to_wire[target]

        # Find sparsest decomposition: XOR with existing wire to minimize popcount
        best_diff = target
        best_helper = None
        for existing_val, existing_wire in val_to_wire.items():
            diff = target ^ existing_val
            if diff == 0:
                return existing_wire
            if best_helper is None or popcount(diff) < popcount(best_diff):
                best_diff = diff
                best_helper = existing_wire

        # Recursively build the difference
        if best_diff == 0:
            return best_helper

        diff_wire = build_from_inputs(best_diff)
        gates.append([best_helper, diff_wire])
        new_wire = next_wire
        next_wire += 1
        val_to_wire[target] = new_wire
        return new_wire

    def find_one_gate_solution(target):
        """Check if target can be made with one XOR gate from existing wires."""
        # target = a ^ b where a, b are existing wires
        for va, wa in val_to_wire.items():
            vb = target ^ va
            if vb in val_to_wire:
                wb = val_to_wire[vb]
                if wa != wb:
                    return (wa, wb)
        return None

    def estimate_cost(target):
        """Estimate gate count to build target."""
        if target == 0:
            return 0
        if target in val_to_wire:
            return 0
        # Check one-gate solution
        if find_one_gate_solution(target) is not None:
            return 1
        # Estimate: 1 + min popcount over helpers
        min_est = float('inf')
        for existing_val in val_to_wire:
            diff = target ^ existing_val
            if diff == 0:
                min_est = 0
                break
            est = 1 + popcount(diff)
            if est < min_est:
                min_est = est
        return min_est

    # Process rows greedily: always pick the row with lowest current cost
    row_to_wire = [None] * m
    remaining = list(range(m))

    # Initial costs
    costs = {}
    for i in range(m):
        costs[i] = estimate_cost(rows[i])
        if costs[i] == 0:
            row_to_wire[i] = val_to_wire[rows[i]] if rows[i] in val_to_wire else n
            remaining.remove(i)

    while remaining:
        if time.time() > deadline:
            break

        # Find row with minimum estimated cost
        min_cost = float('inf')
        min_idx = -1
        for i in remaining:
            c = estimate_cost(rows[i])
            if c < min_cost:
                min_cost = c
                min_idx = i

        if min_idx == -1:
            break

        target = rows[min_idx]

        # Check if already available
        if target == 0:
            row_to_wire[min_idx] = n
            remaining.remove(min_idx)
            continue

        if target in val_to_wire:
            row_to_wire[min_idx] = val_to_wire[target]
            remaining.remove(min_idx)
            continue

        # Try one-gate solution first
        one_gate = find_one_gate_solution(target)
        if one_gate is not None:
            w1, w2 = one_gate
            gates.append([w1, w2])
            new_wire = next_wire
            next_wire += 1
            val_to_wire[target] = new_wire
            row_to_wire[min_idx] = new_wire
            remaining.remove(min_idx)
            continue

        # Build recursively
        wire = build_from_inputs(target)
        row_to_wire[min_idx] = wire
        remaining.remove(min_idx)

    # Ensure all rows have wires
    for i in range(m):
        if row_to_wire[i] is None:
            if rows[i] == 0:
                row_to_wire[i] = n
            elif rows[i] in val_to_wire:
                row_to_wire[i] = val_to_wire[rows[i]]
            else:
                row_to_wire[i] = build_from_inputs(rows[i])

    return {'gates': gates, 'outputs': row_to_wire}
