import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 54.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert to list of bitmasks for faster manipulation
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    # Step 1: Gaussian Elimination to find a basis and coordinates
    # We want to express each row as a linear combination of a set of basis vectors.
    # Basis vectors will be the rows of the RREF (or similar).
    
    # To keep it simple and robust within time limits:
    # 1. Identify pivot rows.
    # 2. The basis is the set of pivot rows (in their original form or reduced form? 
    #    Using reduced form (RREF) makes the coordinates trivial (the row itself in RREF space), 
    #    but the basis vectors themselves might be more complex (higher popcount) than original pivot rows.
    #    However, RREF basis vectors have a 1 in the pivot column and 0 in other pivot columns.
    #    Let's use the original pivot rows as the basis to keep them "simple" (low popcount usually).
    #    Actually, let's just use the standard algorithm:
    
    # We will construct the circuit for the basis vectors first.
    # Then we construct the circuit for the combinations.
    
    # Let's stick to the memoized `get_wire` approach but apply it to the Basis vectors,
    # and then express original rows as XORs of Basis wires.
    
    # 1. Find Basis and Coordinates
    # We perform elimination on the masks.
    # basis[i] will be the mask of the i-th basis vector.
    # coords[i] will be the bitmask of basis indices that sum to row i.
    
    basis_masks = []
    # pivot_col_to_basis_index = {}
    
    # We process columns to find pivots.
    # Standard Gaussian Elimination over GF(2).
    # We maintain a list of basis vectors.
    
    # To optimize, we can just use the rows that become pivots in the elimination process.
    # Let's create a list of (mask, original_index) and reduce.
    
    # Actually, a simpler structural change:
    # Just use the current `get_wire` logic but ensure we process rows in an order that maximizes sharing?
    # No, the directive asks for a structural change.
    
    # Let's implement the Basis Decomposition properly.
    
    # 1. Compute RREF-like basis
    # We'll store the basis vectors as masks.
    # We'll also store the transformation: original_row = XOR of basis vectors.
    
    # Let's use an array `basis` where basis[c] is the mask of the basis vector with pivot at column c.
    # If no basis vector has pivot at c, basis[c] is None.
    
    basis = [None] * n
    # coords[i] will be a bitmask of indices in the `basis_vectors` list (not column indices directly, 
    # but let's map pivot columns to basis indices).
    
    basis_indices = [] # List of column indices that have a basis vector
    col_to_basis_idx = {}
    
    # We need to track the linear combination for each original row.
    # Let `current_masks` be the rows being reduced.
    # We want to find $x$ such that $A x = 0$? No, we want to express rows.
    
    # Let's do:
    # For each row r:
    #   current = r
    #   combo = 0 (bitmask of basis indices used)
    #   For each pivot col c (in reverse order of elimination? or forward?):
    #     If current has bit c set:
    #       current = current XOR basis[c]
    #       combo = combo XOR (1 << basis_idx[c])
    #   If current == 0:
    #      row r is sum of basis vectors in combo.
    #   Else:
    #      current is a new basis vector. Add to basis.
    
    # This standard algorithm gives us:
    # 1. A set of basis vectors (the `current` that didn't reduce to 0).
    # 2. The coordinates for each row (the `combo`).
    
    # Let's implement this.
    
    # We need to be careful about the order of pivots to ensure we can reduce.
    # Standard Gaussian elimination processes columns 0 to n-1.
    
    # `basis_by_col[c]` = mask of basis vector with leading 1 at column c.
    # `basis_id_by_col[c]` = index of this basis vector in our `basis_list`.
    
    basis_by_col = [None] * n
    basis_list = [] # List of masks
    col_to_basis_id = {}
    
    row_coords = [] # List of bitmasks, where bit k is 1 if basis_list[k] is used.
    
    for i in range(m):
        current = row_masks[i]
        combo = 0
        # Reduce current using existing basis
        # We should iterate through pivots in descending order of column index? 
        # Or ascending? 
        # In standard GE, we eliminate from top to bottom. 
        # Here, we are reducing a new row against the current basis.
        # To ensure we can clear bits, we should use the basis vector with the highest set bit?
        # No, standard GE: for col in 0..n-1: if bit set, XOR with basis[col].
        # This works if basis[col] has leading 1 at col.
        
        for c in range(n):
            if (current >> c) & 1:
                if basis_by_col[c] is not None:
                    current ^= basis_by_col[c]
                    combo ^= (1 << col_to_basis_id[c])
                else:
                    # This is a new basis vector
                    # But wait, if we stop here, we haven't reduced the higher bits?
                    # No, we iterate c from 0 to n-1. 
                    # If we encounter a 1 at c and no basis exists, we make it a basis.
                    # However, the new basis vector might have 1s in higher columns.
                    # This is fine for the "leading 1" property if we define leading 1 as the lowest set bit?
                    # Usually leading 1 is the highest. 
                    # Let's iterate c from n-1 down to 0 to handle "highest set bit" pivots.
                    pass
                    
        # Let's restart the logic with c from n-1 down to 0.
        # This ensures that when we pick a pivot at c, all higher bits are 0.
        
        current = row_masks[i]
        combo = 0
        for c in range(n-1, -1, -1):
            if (current >> c) & 1:
                if basis_by_col[c] is not None:
                    current ^= basis_by_col[c]
                    combo ^= (1 << col_to_basis_id[c])
                else:
                    # New basis vector
                    basis_id = len(basis_list)
                    basis_list.append(current)
                    basis_by_col[c] = current
                    col_to_basis_id[c] = basis_id
                    combo ^= (1 << basis_id)
                    break # The rest of the bits (lower than c) are handled by the new basis vector's definition?
                    # Actually, we just added it. We don't need to reduce further for this row.
                    # The row IS the new basis vector (plus the previous combo? No, the combo was for higher pivots).
                    # Wait, if we break, `current` is the new basis vector.
                    # The original row was: (XOR of higher basis vectors) XOR (new basis vector).
                    # So `combo` is correct.
                    pass
        
        row_coords.append(combo)

    # Now we have `basis_list` (masks) and `row_coords` (bitmasks of basis indices).
    
    # Step 2: Synthesize
    # We need to create wires for each basis vector.
    # Then for each row, we XOR the corresponding basis wires.
    
    gates = []
    wire_cache = {}
    zero_wire = n
    
    def get_wire(mask):
        if mask == 0:
            return zero_wire
        if mask in wire_cache:
            return wire_cache[mask]
        
        # Find lowest set bit to split? Or highest?
        # To maximize sharing, we should split in a way that sub-masks are likely to be reused.
        # Using the lowest set bit:
        # mask = bit0 + rest.
        # Wire = XOR(wire(bit0), wire(rest)).
        # bit0 is an input wire.
        # This creates a tree.
        
        # Let's use the recursive split on sorted indices again, but now we are only calling it
        # for the basis vectors and the final combinations.
        
        # Convert mask to sorted list of indices
        indices = []
        mm = mask
        idx = 0
        while mm:
            if mm & 1:
                indices.append(idx)
            mm >>= 1
            idx += 1
        
        if len(indices) == 1:
            wire_id = indices[0]
            wire_cache[mask] = wire_id
            return wire_id
            
        # Split
        mid = len(indices) // 2
        left_indices = indices[:mid]
        right_indices = indices[mid:]
        
        left_mask = 0
        for i in left_indices:
            left_mask |= (1 << i)
        right_mask = 0
        for i in right_indices:
            right_mask |= (1 << i)
        
        left_wire = get_wire(left_mask)
        right_wire = get_wire(right_mask)
        
        gate_id = len(gates)
        gates.append([left_wire, right_wire])
        wire_id = n + 1 + gate_id
        
        wire_cache[mask] = wire_id
        return wire_id

    # Synthesize Basis Wires
    basis_wires = []
    for b_mask in basis_list:
        basis_wires.append(get_wire(b_mask))
        
    # Synthesize Output Wires
    outputs = []
    for i in range(m):
        combo = row_coords[i]
        if combo == 0:
            outputs.append(zero_wire)
        else:
            # XOR all basis wires where combo has a bit set
            # We can use get_wire on a "virtual" mask? No, these are intermediate wires.
            # We need to XOR wires.
            
            # Collect wires
            wires_to_xor = []
            c = combo
            idx = 0
            while c:
                if c & 1:
                    wires_to_xor.append(basis_wires[idx])
                c >>= 1
                idx += 1
            
            if len(wires_to_xor) == 1:
                outputs.append(wires_to_xor[0])
            else:
                # Chain XOR or Tree XOR
                # Tree XOR is better for depth
                # We can build a small tree here
                # Or just chain, since number of basis vectors is small (<= n)
                
                # Let's use a helper to XOR a list of wires
                def xor_wires(wlist):
                    if not wlist:
                        return zero_wire
                    if len(wlist) == 1:
                        return wlist[0]
                    
                    # Balanced split
                    mid = len(wlist) // 2
                    left_w = xor_wires(wlist[:mid])
                    right_w = xor_wires(wlist[mid:])
                    
                    gate_id = len(gates)
                    gates.append([left_w, right_w])
                    return n + 1 + gate_id
                    
                outputs.append(xor_wires(wires_to_xor))

    return {'gates': gates, 'outputs': outputs}
