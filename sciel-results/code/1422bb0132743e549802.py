"""Exact XOR-circuit synthesis: minimize mean gate-count / independent-row baseline over 8x8,12x16,16x16 maps."""
import time
import numpy as np

def _rref_gf2_with_basis(M):
    """
    Compute RREF of M over GF(2) and return the basis rows and the transformation matrix.
    Returns:
    - basis: list of non-zero rows in RREF (numpy arrays)
    - pivot_cols: list of pivot column indices
    - rank: integer rank
    - coeffs: 2D array where coeffs[i] gives the coefficients of the i-th row of M in terms of the basis rows.
             Specifically, M[i] = sum_j coeffs[i][j] * basis[j]
    """
    M = M.copy()
    rows, cols = M.shape
    if rows == 0 or cols == 0:
        return [], [], 0, np.zeros((rows, 0), dtype=np.uint8)

    # Augment with identity to track row operations
    # We want to find P such that M * P = RREF.
    # Actually, standard Gaussian elimination performs row operations.
    # Let E be the product of elementary matrices. RREF = E * M.
    # Then M = E^{-1} * RREF.
    # The rows of M are linear combinations of the rows of RREF.
    # If we perform elimination on [M | I], we get [RREF | E^{-1}].
    # So the right half gives the coefficients of original rows in terms of RREF rows.
    
    augmented = np.hstack([M, np.eye(rows, dtype=np.uint8)])
    
    pivot_cols = []
    rank = 0
    
    for col in range(cols):
        if rank >= rows:
            break
        # Find pivot
        pivot = -1
        for r in range(rank, rows):
            if augmented[r, col]:
                pivot = r
                break
        if pivot == -1:
            continue
        
        # Swap
        augmented[[rank, pivot]] = augmented[[pivot, rank]]
        
        # Eliminate
        for r in range(rows):
            if r != rank and augmented[r, col]:
                augmented[r, :] ^= augmented[rank, :]
        
        pivot_cols.append(col)
        rank += 1
    
    basis = [augmented[i, :cols] for i in range(rank)]
    coeffs = augmented[:, cols:]
    
    return basis, pivot_cols, rank, coeffs

def _synthesize_xor_set(indices, wire_map, next_wire, gates):
    """
    Synthesize the XOR of a set of input indices.
    Returns the wire number representing the XOR sum.
    Updates wire_map and gates.
    """
    if not indices:
        return wire_map[frozenset()] # Zero wire
    
    if len(indices) == 1:
        return indices[0]
    
    # Use a balanced tree approach for depth efficiency, but rely on global map for sharing
    # For small sets, a simple iterative pairwise XOR is fine and often optimal for gate count if sharing is high.
    # However, depth limit is 16. For 16x16, max set size is 16.
    # Log2(16) = 4 levels. So balanced tree is safe for depth.
    
    level = [frozenset([x]) for x in indices]
    
    while len(level) > 1:
        new_level = []
        for i in range(0, len(level) - 1, 2):
            s1 = level[i]
            s2 = level[i + 1]
            combined = s1 | s2
            if combined in wire_map:
                new_level.append(combined)
            else:
                w1 = wire_map[s1]
                w2 = wire_map[s2]
                gates.append([w1, w2])
                wire_map[combined] = next_wire
                new_level.append(combined)
                next_wire += 1
        if len(level) % 2 == 1:
            new_level.append(level[-1])
        level = new_level
        
    return wire_map[level[0]]

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0
    
    M = np.array(matrix, dtype=np.uint8)
    n_rows, n_cols = M.shape
    
    # Wires:
    # 0 .. n_cols-1 : Inputs
    # n_cols        : Constant 0
    # n_cols + 1 ... : Gates
    
    zero_wire = n_cols
    next_wire = n_cols + 1
    gates = []
    wire_map = {}
    
    # Initialize wire_map with inputs and zero
    for i in range(n_cols):
        wire_map[frozenset([i])] = i
    wire_map[frozenset()] = zero_wire
    
    # Step 1: Global Algebraic Decomposition
    # Find basis of the row space
    basis, pivot_cols, rank, coeffs = _rref_gf2_with_basis(M)
    
    # Step 2: Synthesize Basis Vectors
    # Each basis vector is a linear combination of input columns.
    # We synthesize each basis vector and store its wire in wire_map.
    # Note: Basis vectors are linearly independent, so their sub-expression sets
    # might still overlap in terms of input columns, but the global wire_map
    # will handle sharing of common input subsets (e.g., if two basis vectors
    # both use input 0, they share the wire for {0}).
    
    basis_wires = []
    for i in range(rank):
        b_vec = basis[i]
        indices = [j for j in range(n_cols) if b_vec[j] == 1]
        
        if not indices:
            w = zero_wire
        else:
            w = _synthesize_xor_set(indices, wire_map, next_wire, gates)
            # Update next_wire inside the function? 
            # The function modifies gates and wire_map, but next_wire is passed by value.
            # We need to return the new next_wire or handle it differently.
            # Let's refactor _synthesize_xor_set to return (wire, next_wire)
            # Wait, I defined it to update local next_wire. I need to fix this.
            
    # Let's rewrite _synthesize_xor_set to return the new next_wire as well.
    
    # Re-implementing for correctness
    def _syn_set(indices, wm, nw, gs):
        if not indices:
            return wm[frozenset()], nw
        if len(indices) == 1:
            return indices[0], nw
        
        level = [frozenset([x]) for x in indices]
        while len(level) > 1:
            new_level = []
            for i in range(0, len(level) - 1, 2):
                s1 = level[i]
                s2 = level[i + 1]
                combined = s1 | s2
                if combined in wm:
                    new_level.append(combined)
                else:
                    w1 = wm[s1]
                    w2 = wm[s2]
                    gs.append([w1, w2])
                    wm[combined] = nw
                    new_level.append(combined)
                    nw += 1
            if len(level) % 2 == 1:
                new_level.append(level[-1])
            level = new_level
        return wm[level[0]], nw

    # Reset state for the corrected implementation
    next_wire = n_cols + 1
    gates = []
    wire_map = {frozenset([i]): i for i in range(n_cols)}
    wire_map[frozenset()] = zero_wire
    
    basis_wires = []
    for i in range(rank):
        b_vec = basis[i]
        indices = [j for j in range(n_cols) if b_vec[j] == 1]
        w, next_wire = _syn_set(indices, wire_map, next_wire, gates)
        basis_wires.append(w)
        
        if time.time() > deadline:
            # Fallback: if we run out of time, we can't finish.
            # But for 16x16, rank <= 16. Synthesis is very fast.
            pass

    # Step 3: Synthesize Outputs
    # Each output row M[i] is a linear combination of the basis rows.
    # coeffs[i] contains the coefficients.
    # M[i] = sum_j coeffs[i][j] * basis[j]
    # So the wire for M[i] is the XOR of basis_wires[j] for all j where coeffs[i][j] == 1.
    
    outputs = []
    for i in range(n_rows):
        if time.time() > deadline:
            # Emergency fallback: if we haven't computed all, 
            # we must return a valid circuit. 
            # Since the loop is fast, this is unlikely.
            # If it happens, we can just return what we have or a trivial (wrong) circuit?
            # Better to ensure we don't hit deadline.
            pass
            
        c_vec = coeffs[i]
        active_basis_indices = [j for j in range(rank) if c_vec[j] == 1]
        
        if not active_basis_indices:
            outputs.append(zero_wire)
        else:
            # We need to XOR the wires corresponding to these basis indices.
            # These are already computed wires (basis_wires).
            # We can treat this as a set of "virtual" inputs to the XOR tree,
            # but they are just wires.
            # We can create a set representation for these wires?
            # Or just do a pairwise XOR of the wires.
            
            # Since basis_wires are just integers (wire IDs), we can XOR them directly.
            # To minimize gates, we should check if the combination of these basis wires
            # has been computed before? 
            # The set of basis indices is a subset of {0..rank-1}.
            # We can memoize these combinations too.
            
            # Let's use a separate memo for basis combinations?
            # Or just use the same wire_map logic?
            # The wire_map keys are frozensets of INPUT indices.
            # Here we are combining BASIS wires.
            # We can map frozenset(active_basis_indices) to a wire.
            
            key = frozenset(active_basis_indices)
            if key in wire_map:
                outputs.append(wire_map[key])
            else:
                # Synthesize XOR of these specific wires
                wires_to_xor = [basis_wires[j] for j in active_basis_indices]
                
                # Simple pairwise XOR for small sets (rank <= 16)
                # Depth of log2(16) = 4, well within limit.
                # We can use a similar tree structure.
                
                # Actually, since these are just wires, we can just XOR them in a chain or tree.
                # Let's use a tree to be safe on depth, though chain is also <= 16 for 16 items.
                
                level = wires_to_xor
                while len(level) > 1:
                    new_level = []
                    for i in range(0, len(level) - 1, 2):
                        w1 = level[i]
                        w2 = level[i + 1]
                        # Check if this pair has been seen? 
                        # It's hard to key on wires directly without a map.
                        # Let's just create gates.
                        gates.append([w1, w2])
                        new_level.append(next_wire)
                        next_wire += 1
                    if len(level) % 2 == 1:
                        new_level.append(level[-1])
                    level = new_level
                
                final_wire = level[0]
                wire_map[key] = final_wire
                outputs.append(final_wire)
                
    return {'gates': gates, 'outputs': outputs}
