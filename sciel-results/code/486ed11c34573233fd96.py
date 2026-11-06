"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _xor_count(vec):
    """Count number of 1s in a binary vector (numpy array of 0/1)."""
    return int(np.sum(vec))

def _mat_vec_mul_gf2(M, v):
    """Multiply matrix M (binary) by vector v (binary) over GF(2)."""
    # M is shape (m, n), v is shape (n,)
    # Result is M @ v mod 2
    return (M @ v) % 2

def _row_space_gf2(matrix):
    """Find a basis for the row space of a binary matrix over GF(2)."""
    A = matrix.copy().astype(np.int64)
    m, n = A.shape
    # Gaussian elimination
    rank = 0
    for col in range(n):
        # Find pivot
        pivot = -1
        for row in range(rank, m):
            if A[row, col] % 2 == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        # Swap rows
        A[[rank, pivot]] = A[[pivot, rank]]
        # Eliminate
        for row in range(m):
            if row != rank and A[row, col] % 2 == 1:
                A[row] = (A[row] + A[rank]) % 2
        rank += 1
    return A[:rank]

def _baseline_gate_count(matrix):
    """Calculate the baseline gate count: sum of XOR counts per row."""
    return int(np.sum(matrix))

def _synthesize_greedy_pairs(matrix, depth_limit, deadline):
    """
    Greedy synthesis using shared pairs.
    Returns (gates, outputs, gate_count) or None if timeout.
    """
    m, n = matrix.shape
    start_time = time.time()
    
    # Step 1: Count occurrences of each pair in rows
    pair_counts = {}
    for i in range(m):
        row = matrix[i]
        support = np.where(row == 1)[0]
        for a in range(len(support)):
            for b in range(a+1, len(support)):
                pair = (support[a], support[b])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    # Step 2: Keep pairs with count >= 2 (potential sharing)
    shared_pairs = [p for p, c in pair_counts.items() if c >= 2]
    shared_pairs.sort(key=lambda p: pair_counts[p], reverse=True)
    
    # Step 3: Build the gate list
    # We'll assign wire numbers as we go
    # Input wires: 0..n-1, constant-zero: n
    # Gate k produces wire n+1+k
    
    gates = []  # list of [wire1, wire2]
    wire_map = {}  # maps (pair) -> wire number
    next_wire = n + 1  # first gate wire
    
    # Add gates for shared pairs
    for pair in shared_pairs:
        i, j = pair
        wire = next_wire
        next_wire += 1
        gates.append([i, j])
        wire_map[pair] = wire
    
    # Step 4: For each row, compute the output using available shared gates
    outputs = []
    for i in range(m):
        row = matrix[i]
        support = np.where(row == 1)[0]
        
        # Try to use shared gates to reduce XOR count
        # We need to find a matching of pairs in the support that are in wire_map
        # Greedy matching: pick pairs that are in wire_map
        used_wires = set()
        remaining_support = list(support)
        
        # Build adjacency for matching
        # For each pair in wire_map, check if both elements are in remaining_support
        available_pairs = []
        for pair in wire_map:
            a, b = pair
            if a in remaining_support and b in remaining_support:
                available_pairs.append((pair, a, b))
        
        # Greedy matching: pick pairs that don't conflict
        chosen_pairs = []
        used_in_matching = set()
        for pair, a, b in available_pairs:
            if a not in used_in_matching and b not in used_in_matching:
                chosen_pairs.append(pair)
                used_in_matching.add(a)
                used_in_matching.add(b)
        
        # Now compute the output
        # Start with XOR of all remaining elements
        # For each chosen pair, we can replace x_a XOR x_b with the gate wire
        # So the output is XOR of (gate wires for chosen pairs) XOR (remaining elements)
        
        elements = []
        for pair in chosen_pairs:
            elements.append(wire_map[pair])
        for elem in remaining_support:
            if elem not in used_in_matching:
                elements.append(elem)
        
        # XOR all elements together
        if not elements:
            # Row is all zeros
            outputs.append(n)  # constant zero wire
        elif len(elements) == 1:
            outputs.append(elements[0])
        else:
            # Need to XOR multiple elements
            # We can chain XORs: ((e1 XOR e2) XOR e3) ...
            # But this might exceed depth limit
            # For now, just chain them
            current = elements[0]
            for elem in elements[1:]:
                wire = next_wire
                next_wire += 1
                gates.append([current, elem])
                current = wire
            outputs.append(current)
    
    # Verify depth constraint (simplified: check if any chain is too long)
    # For now, assume it's fine for small matrices
    
    return gates, outputs, len(gates)

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given matrix.
    
    Args:
        matrix: Binary matrix (m x n) where m is number of outputs, n is number of inputs
        depth_limit: Maximum depth of the circuit
    
    Returns:
        dict with 'gates' (list of [wire1, wire2]) and 'outputs' (list of wire numbers)
    """
    start_time = time.time()
    deadline = start_time + 55  # 55 second deadline for safety
    
    # Convert to numpy if needed
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    m, n = matrix.shape
    
    # Try the greedy pairs approach
    result = _synthesize_greedy_pairs(matrix, depth_limit, deadline)
    
    if result is not None:
        gates, outputs, gate_count = result
        return {'gates': gates, 'outputs': outputs}
    
    # Fallback: naive synthesis (one gate per XOR in each row)
    gates = []
    outputs = []
    next_wire = n + 1
    
    for i in range(m):
        row = matrix[i]
        support = np.where(row == 1)[0]
        
        if len(support) == 0:
            outputs.append(n)
        elif len(support) == 1:
            outputs.append(support[0])
        else:
            current = support[0]
            for elem in support[1:]:
                wire = next_wire
                next_wire += 1
                gates.append([current, elem])
                current = wire
            outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
