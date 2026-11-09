import time
import numpy as np

def synthesize(matrix, depth_limit):
    """
    Synthesize an exact XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D binary array (n_rows x n_cols) over GF(2).
        depth_limit: maximum allowed depth (<=16).
    
    Returns:
        dict with 'gates' (list of [wire, wire]) and 'outputs' (list of wires).
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Handle edge cases
    if matrix is None:
        return {'gates': [], 'outputs': []}
    
    matrix = np.array(matrix, dtype=np.uint8)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    
    n_rows, n_cols = matrix.shape
    n = n_cols  # input wires are 0..n-1
    const_zero = n  # constant-zero wire
    
    # If no rows, return empty
    if n_rows == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert to GF(2) rows as bitmasks for efficient XOR operations
    # Each row is a vector of length n over GF(2)
    # We'll use Python integers as bit vectors for speed
    
    def row_to_int(row):
        """Convert a binary row to an integer bitmask."""
        val = 0
        for j in range(n):
            if row[j] != 0:
                val |= (1 << j)
        return val
    
    rows = [row_to_int(matrix[i]) for i in range(n_rows)]
    
    # Baseline: one gate per row (directly wire input to output)
    # Actually, if a row is just a single input, no gate needed
    # If a row is all zeros, output is const_zero
    # If a row has multiple 1s, we need XOR gates
    
    # Strategy: Greedy XOR circuit synthesis
    # We process rows one by one, building up intermediate wires
    # Each output wire is the result of XORing a subset of input wires
    # We want to minimize total gate count
    
    # Approach: For each row, determine which inputs are needed
    # Then build a tree of XOR gates to compute the row's value
    
    # Better approach: Use a shared intermediate wire strategy
    # 1. Compute frequency of each input across all rows
    # 2. For inputs that appear in multiple rows, create a shared wire
    # 3. Build XOR trees for each row using available wires
    
    # Even better: Greedy set-cover-like approach
    # Each gate computes XOR of two existing wires
    # We want to minimize total gates
    
    # Simple but effective: For each row, if it has k ones, 
    # we need k-1 XOR gates to combine them (if no sharing)
    # With sharing, we can do better
    
    # Let's use a greedy approach:
    # 1. Start with all input wires available
    # 2. For each row, greedily select pairs to XOR, preferring pairs that
    #    are also needed by other rows
    
    # Actually, let's use a cleaner approach:
    # Represent each row as a set of input indices
    # Use a greedy algorithm to build XOR trees with maximum sharing
    
    row_sets = []
    for i in range(n_rows):
        s = set()
        r = rows[i]
        for j in range(n):
            if r & (1 << j):
                s.add(j)
        row_sets.append(s)
    
    # Count how many rows need each input
    input_freq = [0] * n
    for s in row_sets:
        for j in s:
            input_freq[j] += 1
    
    # Strategy: 
    # For each row, we'll build an XOR tree
    # To minimize gates, we want to share intermediate results
    # 
    # Simple greedy: 
    # - If a row has 0 elements, output = const_zero
    # - If a row has 1 element, output = that input wire
    # - If a row has k elements, we need k-1 gates minimum
    # - Total gates = sum over rows of (|row| - 1) for non-trivial rows
    # 
    # This is the baseline. To do better, we need sharing.
    # 
    # Sharing strategy: 
    # If two rows share a common subset, we can compute that subset once
    # 
    # Let's use a simple but effective approach:
    # 1. Sort rows by size (largest first)
    # 2. For each row, greedily combine inputs into an XOR tree
    # 3. Track which intermediate wires have been created
    # 4. If an intermediate wire matches a needed subset for another row, reuse it
    
    # Actually, for the given problem sizes (8x8, 12x16, 16x16),
    # the baseline of one gate per pair is already quite good.
    # Let's implement a simple greedy with some sharing.
    
    # Wire management
    # Wires: 0..n-1 are inputs, n is const_zero, n+1+ is gates
    next_wire = n + 1
    gates = []  # list of [wire1, wire2]
    wire_values = {}  # wire_id -> bitmask of inputs it represents
    
    # Initialize input wires
    for j in range(n):
        wire_values[j] = (1 << j)
    wire_values[const_zero] = 0
    
    # Create a helper to compute XOR of two wires
    def xor_wires(w1, w2):
        """Compute XOR of two wires, creating a new gate if needed."""
        nonlocal next_wire
        v1 = wire_values[w1]
        v2 = wire_values[w2]
        result = v1 ^ v2
        # Check if this result already exists as a wire
        for wid, val in wire_values.items():
            if val == result:
                return wid
        # Create new gate
        new_wire = next_wire
        next_wire += 1
        gates.append([w1, w2])
        wire_values[new_wire] = result
        return new_wire
    
    # For each row, build an XOR tree
    # Use a greedy pairing strategy
    outputs = []
    
    for i in range(n_rows):
        if time.time() > deadline:
            # Return best effort so far
            break
        
        s = row_sets[i]
        if not s:
            outputs.append(const_zero)
            continue
        if len(s) == 1:
            outputs.append(list(s)[0])
            continue
        
        # Build XOR tree for this row
        # Greedy: repeatedly pick two wires, XOR them, continue
        # To minimize gates, we just need |s|-1 gates
        # The order doesn't matter for gate count, but might for sharing
        
        # Collect the input wires for this row
        wires_to_combine = list(s)
        
        # Greedily combine
        result_wire = wires_to_combine[0]
        for j in range(1, len(wires_to_combine)):
            result_wire = xor_wires(result_wire, wires_to_combine[j])
        
        outputs.append(result_wire)
    
    # If we didn't process all rows due to time, fill with baseline
    while len(outputs) < n_rows:
        s = row_sets[len(outputs)]
        if not s:
            outputs.append(const_zero)
        elif len(s) == 1:
            outputs.append(list(s)[0])
        else:
            # Fallback: use first input as approximation (invalid, but better than nothing)
            # Actually, we need to be correct. Let's just use a simple approach.
            # For correctness, we must compute the full XOR.
            # If we're out of time, we'll use the naive approach
            wires_to_combine = list(s)
            result_wire = wires_to_combine[0]
            for j in range(1, len(wires_to_combine)):
                result_wire = xor_wires(result_wire, wires_to_combine[j])
            outputs.append(result_wire)
    
    return {'gates': gates, 'outputs': outputs}
