# EVOLVE-BLOCK-START
import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    # 90s wall clock limit. Use a generous internal deadline to allow more search.
    DEADLINE = start_time + 85.0

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
        """
        if mask in memo:
            return memo[mask]

        # If we hit the deadline, fallback to direct tree construction
        if elapsed() > DEADLINE - 5.0:
            bits = [j for j in range(n) if (mask >> j) & 1]
            w = build_from_bits(bits)
            memo[mask] = w
            return w

        # Find best M in memo to minimize popcount(mask ^ M)
        # We iterate through all keys in memo.
        best_M = 0
        best_rem = mask
        min_pop = mask.bit_count()

        # Optimization: If mask has small popcount, the best M is likely 0 or a subset.
        # Scanning all keys is O(|memo|).
        for M, w_m in memo.items():
            rem = mask ^ M
            pc = rem.bit_count()
            if pc < min_pop:
                min_pop = pc
                best_M = M
                best_rem = rem
                if pc == 0:
                    break
                # If pc == 1, rem is a single bit, which is in memo.
                # So we can break early as this is the best possible non-zero remainder
                # that allows a 1-gate solution (XOR with input).
                if pc == 1:
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

    # --- Revised Plan ---
    # 1. Find basis rows.
    # 2. Sort basis rows by popcount descending.
    # 3. Sort non-basis rows by popcount descending.
    # 4. Process basis rows, then non-basis rows.
    
    # Find basis
    pivot_to_row = {}
    basis_row_indices = []
    
    # We need to reduce rows to find which are independent.
    # We'll do this in the order of popcount ascending to prefer low-weight basis vectors.
    sorted_indices_for_basis = sorted(range(m), key=lambda i: row_masks[i].bit_count())
    
    for idx in sorted_indices_for_basis:
        mask = row_masks[idx]
        if mask == 0:
            continue
            
        current_mask = mask
        # Reduce by existing basis
        for pivot, row_idx in pivot_to_row.items():
            b_mask = row_masks[row_idx]
            # Check if pivot bit is set in current_mask
            if (current_mask >> pivot) & 1:
                current_mask ^= b_mask
        
        if current_mask != 0:
            # New basis vector
            pivot = current_mask.bit_length() - 1
            pivot_to_row[pivot] = idx
            basis_row_indices.append(idx)
    
    # Now we have basis_row_indices.
    # We want to process these first.
    # Sort them by popcount descending to maximize sharing among them?
    # Or ascending?
    # The original code used descending. Let's stick to descending for the basis rows.
    # Then process non-basis rows, also sorted by popcount descending.
    
    basis_set = set(basis_row_indices)
    non_basis_indices = [i for i in range(m) if i not in basis_set]
    
    # Sort basis indices by popcount descending
    basis_indices_sorted = sorted(basis_row_indices, key=lambda i: row_masks[i].bit_count(), reverse=True)
    # Sort non-basis indices by popcount descending
    non_basis_indices_sorted = sorted(non_basis_indices, key=lambda i: row_masks[i].bit_count(), reverse=True)
    
    # Combined processing order
    processing_order = basis_indices_sorted + non_basis_indices_sorted
    
    outputs = [None] * m

    for idx in processing_order:
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

    # Fill any None (should be none)
    for i in range(m):
        if outputs[i] is None:
            # Fallback to zero (invalid if row != 0, but prevents crash)
            outputs[i] = n

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
