"""Exact XOR-circuit synthesis with shared group-XOR decomposition."""

def synthesize(matrix, depth_limit):
    import numpy as np
    
    matrix = np.array(matrix, dtype=np.uint8)
    m, n = matrix.shape
    
    # Wire numbering: 0..n-1 are inputs, n is constant zero (unused),
    # gates produce wires n+1, n+2, ...
    
    gates = []  # list of [wire_a, wire_b]
    wire_count = n  # next new wire will be n+1+gates_count
    
    def new_gate(a, b):
        gates.append([int(a), int(b)])
        return n + 1 + (len(gates) - 1)
    
    # Build shared XOR groups using a binary decomposition approach
    # We'll compute XOR of blocks of size 2^k starting at various positions
    # Then each output is XOR of O(log n) such blocks
    
    # For simplicity and effectiveness, use a standard approach:
    # Build a complete binary tree for the entire input vector
    # Then for each output, decompose its support into tree nodes
    
    # Step 1: Build a balanced XOR tree over all n inputs
    # Tree nodes: level 0 = inputs (0..n-1)
    # Level k has XORs of 2^k consecutive inputs (if they align)
    
    # Actually, let's use a simpler but effective method:
    # For each output, just XOR the required inputs directly,
    # but cache and reuse any XOR of two inputs that appears in multiple outputs.
    
    # Even simpler and often near-optimal: 
    # Build a "prefix XOR" structure won't work for arbitrary subsets.
    
    # Best practical approach: 
    # 1. Compute all needed pairwise/group XORs
    # 2. Use a greedy sharing strategy
    
    # Let's use the following clean approach:
    # Build a circuit where we have a "binary counter" style network.
    # For each output, decompose its bitmask into powers of 2 aligned to the grid.
    
    # Precompute XOR of blocks of size 2^k starting at position i (if i is divisible by 2^k)
    block_cache = {}  # (k, i) -> wire, where block covers inputs i..i+2^k-1
    
    def get_block(k, i):
        """Get wire for XOR of inputs i to i+2^k-1 (i must be divisible by 2^k)"""
        key = (k, i)
        if key in block_cache:
            return block_cache[key]
        
        if k == 0:
            wire = i  # input wire
        else:
            half = 1 << (k - 1)
            left = get_block(k - 1, i)
            right = get_block(k - 1, i + half)
            wire = new_gate(left, right)
        
        block_cache[key] = wire
        return wire
    
    # For each output, decompose its support into maximal aligned blocks
    outputs = []
    for row in matrix:
        # Get the set of input indices that are 1
        indices = list(np.where(row > 0)[0])
        if len(indices) == 0:
            outputs.append(n)  # constant zero
            continue
        if len(indices) == 1:
            outputs.append(indices[0])
            continue
        
        # Decompose the sorted indices into maximal aligned power-of-2 blocks
        # A block of size 2^k starting at i requires i % 2^k == 0 and i+2^k <= n
        # We greedily take the largest aligned block containing each remaining index
        
        remaining = set(indices)
        wires_for_row = []
        
        while remaining:
            # Find the largest block we can take
            # Try from largest k down
            placed = False
            for k in range(n.bit_length() - 1, -1, -1):
                size = 1 << k
                for i in sorted(remaining):
                    if i % size == 0 and i + size <= n:
                        # Check if all indices in [i, i+size) are in remaining
                        block_indices = set(range(i, i + size))
                        if block_indices.issubset(remaining):
                            wire = get_block(k, i)
                            wires_for_row.append(wire)
                            remaining -= block_indices
                            placed = True
                            break
                if placed:
                    break
            if not placed:
                # Take individual elements
                for idx in sorted(remaining):
                    wires_for_row.append(idx)
                break
        
        # XOR all the wires together
        if len(wires_for_row) == 1:
            outputs.append(wires_for_row[0])
        else:
            result = wires_for_row[0]
            for w in wires_for_row[1:]:
                result = new_gate(result, w)
            outputs.append(result)
    
    return {'gates': gates, 'outputs': outputs}
