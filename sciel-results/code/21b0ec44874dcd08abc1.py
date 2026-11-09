"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D array-like of 0s and 1s (m x n).
        depth_limit: Maximum allowed circuit depth (>= 1).
    
    Returns:
        dict with 'gates' (list of [w1, w2] pairs) and 'outputs' (list of wire indices).
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Convert to numpy array of uint8 for efficiency
    mat = np.array(matrix, dtype=np.uint8)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    
    m, n = mat.shape
    
    # Constant-zero wire is n
    # Input wires are 0..n-1
    # Gate k produces wire n + 1 + k
    
    # Helper to build a balanced XOR tree.
    # Uses a mutable list for next_wire to allow updates in recursion.
    def _build_xor_tree(wires, gates, next_wire_ref):
        if not wires:
            return n
        if len(wires) == 1:
            return wires[0]
        
        mid = len(wires) // 2
        left_wires = wires[:mid]
        right_wires = wires[mid:]
        
        left_out = _build_xor_tree(left_wires, gates, next_wire_ref)
        right_out = _build_xor_tree(right_wires, gates, next_wire_ref)
        
        w = next_wire_ref[0]
        gates.append([left_out, right_out])
        next_wire_ref[0] += 1
        return w

    # Step 1: Find a basis for the row space using Gaussian elimination.
    # We store basis vectors as numpy arrays.
    basis_vectors = []  # List of numpy arrays
    
    for i in range(m):
        if time.time() > deadline:
            # Fallback: return empty circuit if time runs out (should not happen for small inputs)
            return {'gates': [], 'outputs': [n] * m}
            
        r = mat[i].copy()
        # Reduce by existing basis vectors
        for bv in basis_vectors:
            # Find pivot of bv
            # Optimization: Store pivot index with basis vector for faster lookup
            # But for small n, np.argmax is fine. Let's store (pivot, vector)
            pass
        
        # Refine Step 1: Store (pivot, vector)
        basis_info = [] # List of (pivot_index, vector_array)
        basis_vectors = [] # Just the vectors for quick access if needed, but we need pivots
        
        # Let's restart the basis construction properly with pivot tracking
        basis_info = []
        for i in range(m):
            r = mat[i].copy()
            for pivot, bv in basis_info:
                if r[pivot]:
                    r = np.bitwise_xor(r, bv)
            
            if np.any(r):
                pivot = int(np.argmax(r))
                basis_info.append((pivot, r))
    
    # Step 2: Express each row in terms of the basis indices.
    # row_to_basis_indices[i] is a list of indices into basis_info
    row_to_basis_indices = []
    for i in range(m):
        r = mat[i].copy()
        indices = []
        for idx, (pivot, bv) in enumerate(basis_info):
            if r[pivot]:
                indices.append(idx)
                r = np.bitwise_xor(r, bv)
        row_to_basis_indices.append(indices)
    
    # Step 3: Build the circuit with memoization of row combinations.
    gates = []
    next_wire_ref = [n + 1]
    basis_wires = []  # wire index for each basis vector
    
    # Build wires for each basis vector
    for pivot, bv in basis_info:
        ones = np.where(bv)[0].tolist()
        if len(ones) == 0:
            basis_wires.append(n)
        else:
            w = _build_xor_tree(ones, gates, next_wire_ref)
            basis_wires.append(w)
            
    # Memoize the wire for each unique combination of basis vectors
    # Key: tuple of basis indices, Value: wire index
    combo_cache = {}
    
    # Build outputs for each row
    outputs = []
    for i in range(m):
        indices = row_to_basis_indices[i]
        key = tuple(indices)
        
        if key in combo_cache:
            outputs.append(combo_cache[key])
            continue
            
        if not indices:
            w = n
        elif len(indices) == 1:
            w = basis_wires[indices[0]]
        else:
            wires_to_xor = [basis_wires[idx] for idx in indices]
            w = _build_xor_tree(wires_to_xor, gates, next_wire_ref)
            
        combo_cache[key] = w
        outputs.append(w)
    
    return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
