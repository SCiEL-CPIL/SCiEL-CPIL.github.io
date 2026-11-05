"""XOR-circuit synthesis: Basis decomposition with CSE and pruning."""
import time
import numpy as np


def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    A = np.array(matrix, dtype=np.uint8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    n_rows, n_cols = A.shape

    if n_rows == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to integer masks
    row_masks = []
    for i in range(n_rows):
        m = 0
        for j in range(n_cols):
            if A[i, j]:
                m |= (1 << j)
        row_masks.append(m)

    # --- Step 1: Gaussian Elimination to find a basis ---
    # We want a basis that is "sparse" (low popcount) to minimize synthesis cost.
    basis = []
    basis_indices = [] # Indices of rows that form the basis
    
    # Work with a copy of the masks
    rows_to_process = list(row_masks)
    
    # We will perform GE on the set of unique non-zero masks
    # Actually, we just need a basis for the row space.
    # Standard GE:
    current_basis = []
    for i in range(n_rows):
        m = row_masks[i]
        if m == 0:
            continue
        
        # Try to reduce m using existing basis vectors
        reduced = m
        for b in current_basis:
            # If the highest set bit of b is set in reduced, XOR
            # We maintain basis in row-echelon form (pivot bits unique)
            # To make it simple, we just check if XOR reduces popcount
            # Or better, maintain pivots.
            
            # Let's use a simpler approach:
            # Just keep a list of independent vectors.
            # If m is in the span of current_basis, skip.
            # Else, add m to basis.
            pass
        
        # Simple Independence Check:
        # Add m to a list, then check rank.
        # For small n, we can just do this:
        new_basis = current_basis + [m]
        # Check rank of new_basis vs current_basis
        # If rank increased, m is independent.
        
        # Efficient check: Try to reduce m to 0 using current_basis.
        temp = m
        for b in current_basis:
            # Find pivot of b
            # This requires basis to be in echelon form.
            # Let's maintain echelon form.
            pass

    # Let's implement a proper GE with echelon form.
    basis_vecs = []
    pivot_cols = []
    
    for i in range(n_rows):
        m = row_masks[i]
        if m == 0:
            continue
        
        # Reduce m by existing basis vectors
        # We assume basis_vecs are in row-echelon form (pivot of basis_vecs[k] is the highest set bit, and unique)
        # Actually, standard GE processes columns.
        # Let's do:
        # For each column j from n_cols-1 down to 0:
        #   If there is a row with bit j set, use it as pivot.
        
        # Simpler:
        # Iterate through rows.
        # For each row, try to eliminate the highest set bit using existing basis.
        
        reduced = m
        # Sort basis by pivot (highest set bit) descending
        # We need to know the pivot of each basis vector.
        
        # Let's restart with a cleaner GE implementation.
        pass

    # --- Revised GE ---
    # We want a basis B such that any row is a linear combination of B.
    # We will store basis as a list of (mask, pivot_col).
    basis = []
    
    for i in range(n_rows):
        m = row_masks[i]
        if m == 0:
            continue
        
        # Find pivot: highest set bit
        # Try to eliminate this bit using existing basis
        # We process bits from high to low?
        # Standard GE:
        # for col in range(n_cols-1, -1, -1):
        #    find row with bit col set.
        
        # Let's do forward elimination.
        # We maintain a list of basis vectors.
        # For new vector m:
        #   for b in basis:
        #       if (m & b) and (m.bit_length() > b.bit_length()): # Not strictly correct
        #           pass
        
        # Correct GE:
        # Maintain basis in row-echelon form.
        # basis[k] has pivot at column p[k]. p[0] < p[1] < ...
        
        # Let's just use a library-free GE.
        # Since n_cols <= 16, we can use integers.
        
        current_basis = []
        for i in range(n_rows):
            m = row_masks[i]
            if m == 0: continue
            
            # Reduce m by current_basis
            # We need to eliminate the highest set bit of m.
            # Find the basis vector with the highest pivot that is <= m's highest bit?
            # No, standard GE eliminates specific pivots.
            
            # Let's do this:
            # For each bit position j from n_cols-1 down to 0:
            #   If m has bit j set:
            #      Look for a basis vector with pivot j.
            #      If found, m ^= basis_vec.
            #      If not found, m becomes a new basis vector with pivot j.
            #      Break.
            
            new_basis_vec = None
            for j in range(n_cols - 1, -1, -1):
                if (m >> j) & 1:
                    # Check if we have a basis vector with pivot j
                    found = False
                    for k in range(len(current_basis)):
                        if current_basis[k][1] == j:
                            m ^= current_basis[k][0]
                            found = True
                            break
                    if not found:
                        new_basis_vec = (m, j)
                        break
            
            if new_basis_vec:
                current_basis.append(new_basis_vec)
        
    # Now current_basis is a row-echelon basis.
    # Synthesize these basis vectors.
    
    gates = []
    mask_to_wire = {0: n_cols}
    wire_mask = [0] * (n_cols + 1)
    wire_mask[n_cols] = 0
    for i in range(n_cols):
        mask_to_wire[1 << i] = i
        wire_mask[i] = 1 << i

    def create_gate(w1, w2):
        m = wire_mask[w1] ^ wire_mask[w2]
        if m in mask_to_wire:
            return mask_to_wire[m]
        gid = n_cols + 1 + len(gates)
        gates.append([w1, w2])
        wire_mask.append(m)
        mask_to_wire[m] = gid
        return gid

    def build_mask(mask):
        if mask in mask_to_wire:
            return mask_to_wire[mask]
        
        # Try to find a decomposition
        # 1. Check if mask ^ w is an input
        for w, m in mask_to_wire.items():
            diff = mask ^ m
            if diff == 0:
                return w
            if diff & (diff - 1) == 0: # Single bit
                j = diff.bit_length() - 1
                return create_gate(w, j)
        
        # 2. Check if mask ^ w1 ^ w2 is 0? No, that's 2 gates.
        # We want 1 gate: mask = w1 ^ w2.
        # This is equivalent to w1 ^ mask = w2.
        # So if w1 is in mask_to_wire, and (w1 ^ mask) is in mask_to_wire.
        for w1, m1 in mask_to_wire.items():
            diff = m1 ^ mask
            if diff in mask_to_wire:
                w2 = mask_to_wire[diff]
                return create_gate(w1, w2)
        
        # 3. Split mask into two parts
        # Find a bit to split on
        # Balanced split by popcount
        bits = [j for j in range(n_cols) if (mask >> j) & 1]
        if not bits:
            return n_cols # Should not happen for non-zero
        
        if len(bits) == 1:
            return bits[0]
        
        # Split in half
        mid = len(bits) // 2
        m1 = 0
        m2 = 0
        for j in bits[:mid]:
            m1 |= (1 << j)
        for j in bits[mid:]:
            m2 |= (1 << j)
        
        w1 = build_mask(m1)
        w2 = build_mask(m2)
        return create_gate(w1, w2)

    # Synthesize Basis
    for b_mask, _ in current_basis:
        build_mask(b_mask)
        
    # Synthesize Rows
    outputs = []
    for i in range(n_rows):
        m = row_masks[i]
        if m == 0:
            outputs.append(n_cols)
            continue
            
        # Find representation in basis
        # Reduce m using basis
        # We need to track which basis vectors were used.
        # Since basis is in echelon form, we can do this:
        combo = 0
        temp = m
        for b_mask, pivot in current_basis:
            if (temp >> pivot) & 1:
                combo |= (1 << len([b for b, p in current_basis if p == pivot])) # This index logic is messy
                temp ^= b_mask
        
        # Better: Store basis with indices.
        # Let's re-do the basis storage to include an index.
        
    # Let's refine the basis generation to store indices.
    
    # RESTART BASIS GENERATION WITH INDICES
    basis = [] # List of (mask, pivot, index)
    index_counter = 0
    for i in range(n_rows):
        m = row_masks[i]
        if m == 0: continue
        
        new_vec = None
        for j in range(n_cols - 1, -1, -1):
            if (m >> j) & 1:
                found = False
                for k in range(len(basis)):
                    if basis[k][1] == j:
                        m ^= basis[k][0]
                        found = True
                        break
                if not found:
                    new_vec = (m, j, index_counter)
                    index_counter += 1
                    break
        if new_vec:
            basis.append(new_vec)
            
    # Synthesize Basis
    for b_mask, _, _ in basis:
        build_mask(b_mask)
        
    # Synthesize Rows
    outputs = []
    for i in range(n_rows):
        m = row_masks[i]
        if m == 0:
            outputs.append(n_cols)
            continue
            
        # Find combo
        combo_mask = 0
        temp = m
        for b_mask, pivot, idx in basis:
            if (temp >> pivot) & 1:
                combo_mask |= (1 << idx)
                temp ^= b_mask
        
        if combo_mask == 0:
            outputs.append(n_cols)
        else:
            # Build combo_mask from basis wires
            # We need a wire for each basis vector.
            # basis_wire[idx]
            basis_wires = [0] * index_counter
            for b_mask, pivot, idx in basis:
                basis_wires[idx] = mask_to_wire[b_mask]
                
            # XOR all selected basis wires
            w = None
            for idx in range(index_counter):
                if (combo_mask >> idx) & 1:
                    if w is None:
                        w = basis_wires[idx]
                    else:
                        w = create_gate(w, basis_wires[idx])
            outputs.append(w)
            
    # --- Post-Processing: CSE and Pruning ---
    
    # 1. CSE: Merge gates with same output mask
    # We need to map wire -> mask.
    # We have wire_mask list.
    
    # 2. Pruning: Remove unused gates.
    # This requires a topological sort or reverse pass.
    
    # Let's implement a simple reverse pass.
    # Create a list of gate indices.
    # Count usage of each wire.
    
    wire_usage = [0] * len(wire_mask)
    for ow in outputs:
        wire_usage[ow] += 1
        
    for g in gates:
        w1, w2 = g
        wire_usage[w1] += 1
        wire_usage[w2] += 1
        
    # This doesn't help delete gates because a gate's output might be used by another gate.
    # We need to process gates in reverse order of creation.
    
    # Re-indexing is complex. Let's just do a CSE pass.
    
    # CSE:
    # Map mask -> wire.
    # If two gates produce the same mask, they are redundant.
    # But `create_gate` already prevents duplicate masks in `mask_to_wire`.
    # So CSE is already handled by the cache!
    
    # Pruning:
    # If a gate's output is not used, it can be removed.
    # But `create_gate` only adds a gate if the mask is new.
    # So all gates are "used" by definition of being in the cache?
    # No, a gate might be created to build an intermediate, but if the intermediate is never used by an output or another gate, it's waste.
    # However, `build_mask` only creates gates to build the target mask.
    # So all gates are on the path to some basis vector or row.
    # Thus, no pruning needed?
    # Wait, if `build_mask(m1)` creates a gate for `m1`, and `m1` is only used to build `m`, and `m` is a basis vector, then it's used.
    # So the circuit is a tree/DAG with no dead ends.
    
    result = {'gates': gates, 'outputs': outputs}
    
    if not _verify(A, result):
        # Fallback
        gates = []
        outputs = []
        for i in range(n_rows):
            idx = [j for j in range(n_cols) if A[i, j] == 1]
            if not idx:
                outputs.append(n_cols)
            elif len(idx) == 1:
                outputs.append(idx[0])
            else:
                cur = idx[0]
                for w in idx[1:]:
                    gid = n_cols + 1 + len(gates)
                    gates.append([cur, w])
                    cur = gid
                outputs.append(cur)
        result = {'gates': gates, 'outputs': outputs}
        
    return result


def _verify(A, result):
    n_cols = A.shape[1]
    gates = result['gates']
    outputs = result['outputs']

    if n_cols > 64:
        return True

    wire_values = [1 << i for i in range(n_cols)]
    wire_values.append(0)

    for gate in gates:
        w1, w2 = gate
        if w1 >= len(wire_values) or w2 >= len(wire_values):
            return False
        wire_values.append(wire_values[w1] ^ wire_values[w2])

    for i, ow in enumerate(outputs):
        if ow >= len(wire_values):
            return False
        expected = 0
        for j in range(n_cols):
            if A[i, j] == 1:
                expected ^= (1 << j)
        if wire_values[ow] != expected:
            return False

    return True
