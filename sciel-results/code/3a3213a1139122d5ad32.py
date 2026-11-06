import time
import numpy as np

def _rref_gf2_with_basis(M):
    """
    Compute RREF of M over GF(2) and return:
      - rref: the RREF matrix
      - rank: the rank
      - pivot_cols: list of pivot column indices
      - basis_rows: indices of original rows that form the basis (in order)
      - coeff_matrix: (n_rows x rank) matrix where row i gives coefficients
        of original row i in terms of the basis rows
    """
    M_work = M.copy().astype(np.uint8)
    n_rows, n_cols = M_work.shape
    rank = 0
    pivot_cols = []
    basis_rows = []
    # Track the transformation: start with identity, apply same row ops
    # coeff_matrix will be the inverse of the row-operation matrix
    # Actually, easier: track which original row each current row is a combo of
    # Let's use a different approach: 
    # Perform elimination, and for each pivot row, record which original rows were combined.
    # But that's complex. Instead, we can compute the basis, then solve for coefficients.
    
    # Step 1: Find basis rows and RREF
    for col in range(n_cols):
        if rank >= n_rows:
            break
        pivot = -1
        for r in range(rank, n_rows):
            if M_work[r, col]:
                pivot = r
                break
        if pivot == -1:
            continue
        M_work[[rank, pivot]] = M_work[[pivot, rank]]
        basis_rows.append(pivot)  # This is the original row index? No, it's the current row index after swaps.
        # Actually, we need to track the original row indices. Let's use a permutation array.
        # Let's restart with a cleaner approach.
        
    # Cleaner approach:
    M_work = M.copy().astype(np.uint8)
    n_rows, n_cols = M_work.shape
    rank = 0
    pivot_cols = []
    basis_row_indices = []  # Original row indices that form the basis
    
    # Track row operations on an identity matrix to get the transformation
    # T such that T @ M = RREF(M)
    T = np.eye(n_rows, dtype=np.uint8)
    
    for col in range(n_cols):
        if rank >= n_rows:
            break
        pivot = -1
        for r in range(rank, n_rows):
            if M_work[r, col]:
                pivot = r
                break
        if pivot == -1:
            continue
        # Swap rows in both M_work and T
        M_work[[rank, pivot]] = M_work[[pivot, rank]]
        T[[rank, pivot]] = T[[pivot, rank]]
        # Eliminate all other rows
        for r in range(n_rows):
            if r != rank and M_work[r, col]:
                M_work[r, :] ^= M_work[rank, :]
                T[r, :] ^= T[rank, :]
        pivot_cols.append(col)
        basis_row_indices.append(rank)  # This is the row index in the final RREF, not original
        rank += 1
    
    # Now M_work[:rank, :] is the RREF basis
    # T @ M = M_work, so M = T^{-1} @ M_work
    # But we want to express each original row as a combo of the basis rows.
    # The basis rows are M_work[0..rank-1, :]
    # For each original row i, we want coeff c such that M[i,:] = sum_j c[j] * M_work[j,:]
    # Since M_work = T @ M, we have M = T^{-1} @ M_work
    # So M[i,:] = (T^{-1})[i,:] @ M_work
    # The coefficients are (T^{-1})[i, 0:rank]
    
    # Compute T^{-1} over GF(2)
    # T is invertible if rank == n_rows, but if rank < n_rows, T is not square invertible in the usual sense.
    # Actually, T is always invertible because it's a product of swaps and row additions (all invertible).
    # So we can compute T^{-1} by augmenting [T | I] and reducing T to I.
    
    # Augment T with identity
    aug = np.hstack([T, np.eye(n_rows, dtype=np.uint8)])
    # Reduce left half to identity
    for col in range(n_rows):
        if col >= n_rows:
            break
        # Find pivot in column col
        pivot = -1
        for r in range(col, n_rows):
            if aug[r, col]:
                pivot = r
                break
        if pivot == -1:
            # This shouldn't happen if T is invertible
            continue
        aug[[col, pivot]] = aug[[pivot, col]]
        for r in range(n_rows):
            if r != col and aug[r, col]:
                aug[r, :] ^= aug[col, :]
    
    T_inv = aug[:, n_rows:]  # Right half is T^{-1}
    
    # Coefficients for each original row i are T_inv[i, 0:rank]
    coeff_matrix = T_inv[:, 0:rank]
    
    return M_work[:rank, :], rank, pivot_cols, coeff_matrix

def _build_xor_tree(sets, wire_map, next_wire):
    """
    Build XOR tree for a list of frozensets.
    Returns (final_wire, gates_list, new_next_wire)
    """
    if len(sets) == 0:
        return None, [], next_wire
    if len(sets) == 1:
        return wire_map[sets[0]], [], next_wire
    
    level = list(sets)
    gates = []
    
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
    
    return wire_map[level[0]], gates, next_wire

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0
    
    M = np.array(matrix, dtype=np.uint8)
    n_rows, n_cols = M.shape
    
    # Step 1: Compute RREF and coefficients
    basis, rank, pivot_cols, coeff_matrix = _rref_gf2_with_basis(M)
    
    # Step 2: Initialize wire map for input wires
    wire_map = {}
    for i in range(n_cols):
        wire_map[frozenset([i])] = i
    wire_map[frozenset()] = n_cols  # Zero wire
    
    next_wire = n_cols + 1
    all_gates = []
    
    # Step 3: Synthesize each basis vector
    # Basis vectors are rows of 'basis' matrix (rank x n_cols)
    for b_idx in range(rank):
        row = basis[b_idx, :]
        ones = [j for j in range(n_cols) if row[j] == 1]
        if len(ones) == 0:
            wire_map[frozenset(range(b_idx, b_idx+1))] = n_cols  # This is wrong, let's use a different key
            # Actually, we need to assign a wire to each basis vector.
            # Let's use a separate mapping for basis vectors.
            pass
        
    # Let's restructure: 
    # We'll assign wires to basis vectors explicitly.
    # basis_wire[j] = wire for basis vector j
    basis_wires = []
    
    for b_idx in range(rank):
        row = basis[b_idx, :]
        ones = [j for j in range(n_cols) if row[j] == 1]
        if len(ones) == 0:
            basis_wires.append(n_cols)  # Zero wire
            continue
        if len(ones) == 1:
            basis_wires.append(ones[0])
            continue
        
        # Build XOR tree for this basis vector
        sets = [frozenset([x]) for x in ones]
        out_wire, gates, next_wire = _build_xor_tree(sets, wire_map, next_wire)
        basis_wires.append(out_wire)
        all_gates.extend(gates)
        
        if time.time() > deadline:
            # Fallback: use simple per-row synthesis for remaining
            break
    
    # Step 4: Synthesize each output row
    outputs = []
    for row_idx in range(n_rows):
        if time.time() > deadline:
            # Fallback: simple XOR tree for this row
            row = M[row_idx, :]
            ones = [j for j in range(n_cols) if row[j] == 1]
            if len(ones) == 0:
                outputs.append(n_cols)
            elif len(ones) == 1:
                outputs.append(ones[0])
            else:
                sets = [frozenset([x]) for x in ones]
                out_wire, gates, next_wire = _build_xor_tree(sets, wire_map, next_wire)
                outputs.append(out_wire)
                all_gates.extend(gates)
            continue
        
        # Get coefficients for this row
        coeff = coeff_matrix[row_idx, :]
        # Find which basis vectors are needed
        needed_basis = [j for j in range(rank) if coeff[j] == 1]
        
        if len(needed_basis) == 0:
            outputs.append(n_cols)
            continue
        if len(needed_basis) == 1:
            outputs.append(basis_wires[needed_basis[0]])
            continue
        
        # Build XOR tree over the basis wires
        sets = [frozenset([basis_wires[j]]) for j in needed_basis]
        # But wait, these are wires, not input indices. We need to XOR wires.
        # The wire_map keys are frozensets of input indices, not wire numbers.
        # This is a problem. We need a different approach for XORing arbitrary wires.
        
        # Alternative: Just XOR the basis wires directly.
        # For two wires w1, w2, we create a new gate [w1, w2] -> new_wire
        # This is simpler and correct.
        
        if len(needed_basis) == 1:
            outputs.append(basis_wires[needed_basis[0]])
        else:
            # Build a balanced XOR tree over the basis wires
            wires_to_xor = [basis_wires[j] for j in needed_basis]
            # Use a simple iterative XOR
            if len(wires_to_xor) == 1:
                outputs.append(wires_to_xor[0])
            else:
                # Balanced tree
                level = list(wires_to_xor)
                while len(level) > 1:
                    new_level = []
                    for i in range(0, len(level) - 1, 2):
                        w1 = level[i]
                        w2 = level[i + 1]
                        all_gates.append([w1, w2])
                        new_level.append(next_wire)
                        next_wire += 1
                    if len(level) % 2 == 1:
                        new_level.append(level[-1])
                    level = new_level
                outputs.append(level[0])
    
    return {'gates': all_gates, 'outputs': outputs}
