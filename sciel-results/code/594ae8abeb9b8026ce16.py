"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D array-like of 0s and 1s (m x n).
        depth_limit: Maximum allowed circuit depth (>= 1).
    
    Returns:
        dict with 'gates' (list of [w1, w2] pairs) and 'outputs' (list of wire indices).
    """
    # Convert to numpy array of uint8 for efficiency
    mat = np.array(matrix, dtype=np.uint8)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    
    m, n = mat.shape
    
    # Constant-zero wire is n
    # Input wires are 0..n-1
    # Gate k produces wire n + 1 + k
    
    # We want to compute each row as a linear combination of inputs.
    # Each row is a vector in GF(2)^n.
    # We need to build a circuit that computes these m linear forms.
    
    # Strategy: Use Gaussian elimination over GF(2) to find a basis,
    # then express each target row in terms of the basis.
    # But we need to minimize gate count.
    
    # Alternative: Greedy approach - for each output, find the cheapest way
    # to compute it using existing wires and new gates.
    
    # Even better: Process all rows together. Find common sub-expressions.
    
    # Let's use a more systematic approach:
    # 1. Collect all unique row vectors.
    # 2. Find a basis for the span of these vectors.
    # 3. Express each row in terms of the basis.
    # 4. Build a circuit that computes the basis vectors, then combines them.
    
    # But this might not be optimal. Let's try a greedy approach with memoization.
    
    # Actually, let's think about this differently. The cost is the number of XOR gates.
    # Each XOR gate takes two wires and produces a new wire.
    # We start with wires 0..n-1 (inputs) and wire n (constant 0).
    
    # Key insight: We can compute linear combinations. The cheapest way to compute
    # a vector v is to find a sequence of XORs that builds it up.
    
    # Let's use the following approach:
    # 1. Start with all input wires available.
    # 2. For each target row, find the minimum-cost way to compute it.
    # 3. Cache intermediate results.
    
    # A better approach: Build a circuit that computes a basis, then express
    # each output in terms of the basis.
    
    # Let's implement a Gaussian elimination based approach:
    
    # Step 1: Find a basis for the row space.
    rows = [mat[i] for i in range(m)]
    
    # Gaussian elimination over GF(2)
    basis = []
    basis_vectors = []  # (pivot_col, vector)
    
    for row in rows:
        r = row.copy()
        for pivot_col, bv in basis_vectors:
            if r[pivot_col]:
                r = np.bitwise_xor(r, bv)
        if np.any(r):
            # Find pivot
            pivot_col = np.argmax(r)
            basis_vectors.append((pivot_col, r))
            basis.append(r)
    
    # Now express each row in terms of the basis.
    # For each row, find the combination of basis vectors that produces it.
    
    # Build a system: row = sum of (coeff_i * basis_i)
    # This is equivalent to solving basis_matrix^T * coeffs = row^T
    
    # Actually, let's just do it directly.
    
    # For each target row, find which basis vectors to XOR together.
    row_to_basis = []
    for row in rows:
        coeffs = []
        r = row.copy()
        for pivot_col, bv in basis_vectors:
            if r[pivot_col]:
                coeffs.append(pivot_col)
                r = np.bitwise_xor(r, bv)
        row_to_basis.append(coeffs)
    
    # Now we need to build a circuit that:
    # 1. Computes each basis vector.
    # 2. For each output, XORs together the appropriate basis vectors.
    
    # But computing basis vectors also costs gates.
    
    # Let's think about the cost:
    # - Each basis vector is a linear combination of inputs.
    # - To compute a basis vector with k ones, we need k-1 gates.
    # - To combine b basis vectors for an output, we need b-1 gates.
    
    # Total cost = sum over basis vectors (popcount(bv) - 1) + sum over outputs (num_basis_for_output - 1)
    
    # But this is an upper bound. We might be able to share intermediate results.
    
    # Let's just build the circuit directly.
    
    # Wire mapping:
    # 0..n-1: inputs
    # n: constant 0
    # n+1, n+2, ...: gate outputs
    
    gates = []
    next_wire = n + 1
    
    # Compute each basis vector.
    # A basis vector is a set of input indices where it has 1s.
    basis_wires = []  # wire index for each basis vector
    
    for i, (pivot_col, bv) in enumerate(basis_vectors):
        # Find the inputs where bv has 1s.
        ones = np.where(bv)[0]
        if len(ones) == 0:
            basis_wires.append(n)  # constant 0
            continue
        if len(ones) == 1:
            basis_wires.append(int(ones[0]))
            continue
        
        # XOR together all the inputs.
        w = int(ones[0])
        for j in range(1, len(ones)):
            gates.append([w, int(ones[j])])
            w = next_wire
            next_wire += 1
        basis_wires.append(w)
    
    # Now compute each output.
    outputs = []
    for i in range(m):
        coeffs = row_to_basis[i]
        if len(coeffs) == 0:
            outputs.append(n)  # constant 0
            continue
        if len(coeffs) == 1:
            outputs.append(basis_wires[coeffs[0]])
            continue
        
        w = basis_wires[coeffs[0]]
        for j in range(1, len(coeffs)):
            gates.append([w, basis_wires[coeffs[j]]])
            w = next_wire
            next_wire += 1
        outputs.append(w)
    
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
