"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D binary array (n_rows x n_cols) where each row represents an output function
                of the input variables (0..n_cols-1).
        depth_limit: Maximum allowed depth of the circuit.
                     
    Returns:
        dict with keys:
            'gates': list of [wire1, wire2] pairs, where each gate computes XOR of wire1 and wire2.
            'outputs': list of wire indices corresponding to the outputs of the circuit.
    """
    start_time = time.time()
    deadline = start_time + 55.0
    
    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs
    
    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}
    
    # Perform Gaussian elimination with minimum weight pivots
    rows = [list(A[i]) for i in range(n_rows)]
    n = n_rows
    
    # Track representation: rep[i] = set of original row indices that XOR to give current row i
    rep = [set([i]) for i in range(n)]
    
    basis_idx = []
    
    for col in range(n_cols):
        # Find pivot: row with 1 in this column, prefer minimum weight
        best_row = -1
        best_weight = n + 1
        for r in range(n):
            if r not in basis_idx and rows[r][col] == 1:
                w = sum(rows[r])
                if w < best_weight:
                    best_weight = w
                    best_row = r
        
        if best_row == -1:
            continue
        
        basis_idx.append(best_row)
        
        # Eliminate this column from all other rows
        for r in range(n):
            if r != best_row and rows[r][col] == 1:
                for c in range(col, n_cols):
                    rows[r][c] ^= rows[best_row][c]
                rep[r] = rep[r] ^ rep[best_row]
    
    # basis_idx: indices of basis rows
    # rep[i]: set of original row indices for row i's representation
    
    # Map original row index to basis index
    orig_to_basis = {orig: bi for bi, orig in enumerate(basis_idx)}
    
    # Extract basis vectors
    basis_vecs = [list(rows[bi]) for bi in basis_idx]
    
    # For each output, determine which basis vectors to XOR
    output_basis_combo = []
    for i in range(n_rows):
        combo = set()
        for orig_idx in rep[i]:
            if orig_idx in orig_to_basis:
                combo.add(orig_to_basis[orig_idx])
        output_basis_combo.append(combo)
    
    # Build circuit
    gates = []
    next_wire = zero_wire + 1
    
    # Compute each basis vector
    basis_wires = []
    for bi in range(len(basis_vecs)):
        vec = basis_vecs[bi]
        ones = [c for c in range(n_cols) if vec[c] == 1]
        if len(ones) == 0:
            basis_wires.append(zero_wire)
        elif len(ones) == 1:
            basis_wires.append(ones[0])
        else:
            current = ones[0]
            for c in ones[1:]:
                gates.append([current, c])
                current = next_wire
                next_wire += 1
            basis_wires.append(current)
    
    # Compute each output as XOR of its basis components
    outputs = []
    for i in range(n_rows):
        combo = output_basis_combo[i]
        if len(combo) == 0:
            outputs.append(zero_wire)
        elif len(combo) == 1:
            outputs.append(basis_wires[list(combo)[0]])
        else:
            indices = sorted(combo)
            current = basis_wires[indices[0]]
            for bi in indices[1:]:
                gates.append([current, basis_wires[bi]])
                current = next_wire
                next_wire += 1
            outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
