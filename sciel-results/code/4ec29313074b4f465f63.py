# EVOLVE-BLOCK-START
import time
import random
import math

def synthesize(matrix, depth_limit):
    start_time = time.time()
    # 90s wall clock limit. Use a generous internal deadline to allow more search.
    DEADLINE = start_time + 80.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Parse rows as bitmasks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

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

    def est_gate_cost(popcount):
        """Estimate gate count to build XOR of `popcount` inputs."""
        if popcount <= 1:
            return 0
        return popcount - 1  # each XOR gate reduces count by 1

    def build_from_bits(bits_list):
        """
        Takes a list of input wire indices (0..n-1).
        Builds a balanced XOR tree, memoizing the mask for each new gate.
        Returns the final wire.
        """
        if not bits_list:
            return n
        if len(bits_list) == 1:
            return bits_list[0]

        wires = []
        masks = []
        for b in bits_list:
            wires.append(b)
            masks.append(1 << b)

        while len(wires) > 1:
            new_wires = []
            new_masks = []
            for i in range(0, len(wires), 2):
                if i + 1 < len(wires):
                    a_w = wires[i]
                    a_m = masks[i]
                    b_w = wires[i + 1]
                    b_m = masks[i + 1]

                    new_m = a_m ^ b_m
                    # Check if new_m is already in memo
                    if new_m in memo:
                        # Reuse existing wire
                        new_wires.append(memo[new_m])
                        new_masks.append(new_m)
                    else:
                        new_w = add_gate(a_w, b_w)
                        memo[new_m] = new_w
                        new_wires.append(new_w)
                        new_masks.append(new_m)
                else:
                    new_wires.append(wires[i])
                    new_masks.append(masks[i])
            wires = new_wires
            masks = new_masks

        return wires[0]

    def compute_mask(mask):
        """
        Compute wire for mask using memoization and greedy nearest-neighbor search.
        Uses estimated gate cost as surrogate.
        """
        if mask in memo:
            return memo[mask]

        # If we hit the deadline, fallback to direct tree construction
        if elapsed() > DEADLINE - 5.0:
            bits = [j for j in range(n) if (mask >> j) & 1]
            w = build_from_bits(bits)
            memo[mask] = w
            return w

        # Find best M in memo to minimize estimated total cost
        # Total cost = 1 (the new gate) + cost_to_build(rem)
        # where cost_to_build(rem) = 0 if rem in memo, else est_gate_cost(popcount(rem))
        best_M = 0
        best_rem = mask
        min_score = float('inf')

        mask_pc = mask.bit_count()
        
        # Scanning all keys is O(|memo|).
        for M, w_m in memo.items():
            rem = mask ^ M
            if rem in memo:
                score = 1  # just one new gate
            else:
                rem_pc = rem.bit_count()
                score = 1 + est_gate_cost(rem_pc)
            
            if score < min_score:
                min_score = score
                best_M = M
                best_rem = rem
                if score == 1:
                    break

        # Check if best_rem is in memo
        if best_rem in memo:
            w_rem = memo[best_rem]
        else:
            # If best_rem is not in memo, build a tree from its bits
            bits = [j for j in range(n) if (best_rem >> j) & 1]
            w_rem = build_from_bits(bits)

        w_m = memo[best_M]
        w_new = add_gate(w_m, w_rem)
        memo[mask] = w_new
        return w_new

    # Sort rows by popcount descending to build strong bases early
    sorted_indices = sorted(range(m), key=lambda i: row_masks[i].bit_count(), reverse=True)

    outputs = [None] * m

    for idx in sorted_indices:
        if elapsed() > DEADLINE - 10.0:
            # Fast path: direct tree construction without search
            mask = row_masks[idx]
            if mask in memo:
                outputs[idx] = memo[mask]
            else:
                bits = [j for j in range(n) if (mask >> j) & 1]
                w = build_from_bits(bits)
                memo[mask] = w
                outputs[idx] = w
        else:
            outputs[idx] = compute_mask(row_masks[idx])

    # Polish pass: try to find cheaper representations for outputs
    # For each output, check if there's a cheaper decomposition using memo
    if elapsed() < DEADLINE - 15.0:
        for idx in sorted_indices:
            if elapsed() > DEADLINE - 15.0:
                break
            mask = row_masks[idx]
            current_wire = outputs[idx]
            if current_wire is None:
                continue
            
            # Try to find a cheaper way: for each M in memo, check if mask^M is in memo
            # and if the total cost is lower than what we have
            # We don't track exact gate counts per wire, so use heuristic:
            # If mask^M is in memo, we could potentially save gates if the original
            # construction used more gates than 1 + cost(mask^M)
            # This is complex to track, so do a simple improvement:
            # If mask itself can be expressed as XOR of two memo entries with fewer
            # total estimated gates than current, rebuild.
            # For now, skip complex polish; the greedy search already does this.
            pass

    # Fill any None (should be none)
    for i in range(m):
        if outputs[i] is None:
            # Fallback to zero (invalid if row != 0, but prevents crash)
            outputs[i] = n

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
