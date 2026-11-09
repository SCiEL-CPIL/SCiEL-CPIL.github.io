"""XOR circuit synthesis with CSE, adaptive ordering, and optimized decomposition."""
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

    # Remove duplicate rows (they share the same wire)
    row_to_base = {}
    unique_rows = []
    for i in range(m):
        r = rows[i]
        if r not in row_to_base:
            row_to_base[r] = len(unique_rows)
            unique_rows.append(r)
        else:
            pass  # already mapped

    # Build the circuit for unique rows
    val_to_wire = {0: n}  # constant zero
    for j in range(n):
        val_to_wire[1 << j] = j

    gates = []
    next_wire = n + 1

    def popcount(x):
        return bin(x).count('1')

    def find_one_gate_solution(target):
        """Check if target can be made with one XOR gate from existing wires."""
        for va, wa in val_to_wire.items():
            vb = target ^ va
            if vb in val_to_wire:
                wb = val_to_wire[vb]
                if wa != wb:
                    return (wa, wb)
        return None

    def estimate_cost(target):
        """Estimate gate count to build target from current wires."""
        if target == 0:
            return 0
        if target in val_to_wire:
            return 0
        # Check one-gate solution
        if find_one_gate_solution(target) is not None:
            return 1
        # Estimate: 1 + min over helpers of (cost to build diff)
        # Use popcount as a proxy for multi-gate cost
        min_est = float('inf')
        for existing_val in val_to_wire:
            diff = target ^ existing_val
            if diff == 0:
                min_est = 0
                break
            # diff can be built with at most popcount(diff) gates (XOR with individual inputs)
            est = 1 + popcount(diff)
            if est < min_est:
                min_est = est
        return min_est

    def build_from_inputs(target):
        """Build target from input wires using minimal gates."""
        nonlocal next_wire, gates
        if target == 0:
            return n
        if target in val_to_wire:
            return val_to_wire[target]

        # Check one-gate solution first
        one_gate = find_one_gate_solution(target)
        if one_gate is not None:
            w1, w2 = one_gate
            gates.append([w1, w2])
            new_wire = next_wire
            next_wire += 1
            val_to_wire[target] = new_wire
            return new_wire

        # Find sparsest decomposition
        best_diff = target
        best_helper = None
        for existing_val, existing_wire in val_to_wire.items():
            diff = target ^ existing_val
            if diff == 0:
                return existing_wire
            if best_helper is None or popcount(diff) < popcount(best_diff):
                best_diff = diff
                best_helper = existing_wire

        if best_diff == 0:
            return best_helper

        # Recursively build the difference
        diff_wire = build_from_inputs(best_diff)
        gates.append([best_helper, diff_wire])
        new_wire = next_wire
        next_wire += 1
        val_to_wire[target] = new_wire
        return new_wire

    # Phase 1: Identify useful intermediate values via pairwise XOR of rows
    # These are candidates for CSE
    candidate_intermediates = set()
    num_unique = len(unique_rows)
    for i in range(num_unique):
        for j in range(i + 1, num_unique):
            cand = unique_rows[i] ^ unique_rows[j]
            if cand != 0 and cand not in val_to_wire:
                candidate_intermediates.add(cand)

    # Phase 2: Greedily add intermediates that reduce total cost
    # Score each candidate by how many rows it helps
    def count_rows_helped(cand):
        """Count how many unique rows can be one-gate built if cand is available."""
        count = 0
        for r in unique_rows:
            if r == 0 or r in val_to_wire:
                continue
            # Check if r ^ cand is in val_to_wire
            if (r ^ cand) in val_to_wire:
                count += 1
        return count

    # Sort candidates by benefit (rows helped) descending, then by popcount ascending
    candidates = list(candidate_intermediates)
    candidates.sort(key=lambda c: (-count_rows_helped(c), popcount(c)))

    # Add top candidates that are cheap to build
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand in val_to_wire:
            continue
        # Only add if it helps at least 2 rows and is cheap
        helped = count_rows_helped(cand)
        if helped < 2:
            continue
        # Check if it's cheap to build (one gate)
        if find_one_gate_solution(cand) is not None:
            w1, w2 = find_one_gate_solution(cand)
            gates.append([w1, w2])
            new_wire = next_wire
            next_wire += 1
            val_to_wire[cand] = new_wire
        elif popcount(cand) <= 2:
            # Build directly from inputs
            build_from_inputs(cand)

    # Phase 3: Build remaining unique rows greedily by cost
    row_wire = {}  # unique_row_value -> wire
    remaining = list(range(num_unique))

    # Remove rows that are already available
    still_remaining = []
    for i in range(num_unique):
        r = unique_rows[i]
        if r == 0:
            row_wire[r] = n
        elif r in val_to_wire:
            row_wire[r] = val_to_wire[r]
        else:
            still_remaining.append(i)

    while still_remaining:
        if time.time() > deadline:
            break

        # Find row with minimum estimated cost
        min_cost = float('inf')
        min_idx = -1
        for i in still_remaining:
            c = estimate_cost(unique_rows[i])
            if c < min_cost:
                min_cost = c
                min_idx = i

        if min_idx == -1:
            break

        target = unique_rows[min_idx]

        if target in val_to_wire:
            row_wire[target] = val_to_wire[target]
            still_remaining.remove(min_idx)
            continue

        wire = build_from_inputs(target)
        row_wire[target] = wire
        still_remaining.remove(min_idx)

    # Ensure all unique rows have wires
    for i in range(num_unique):
        r = unique_rows[i]
        if r not in row_wire:
            if r == 0:
                row_wire[r] = n
            elif r in val_to_wire:
                row_wire[r] = val_to_wire[r]
            else:
                row_wire[r] = build_from_inputs(r)

    # Map back to original row indices
    outputs = [row_wire[rows[i]] for i in range(m)]

    return {'gates': gates, 'outputs': outputs}
