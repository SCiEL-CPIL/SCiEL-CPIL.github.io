import time
import numpy as np
from collections import deque

def synthesize(matrix, depth_limit):
    """
    Exact XOR-circuit synthesis using Global Algebraic Basis Decomposition.
    
    Strategy:
    1. Compute the rank of the matrix over GF(2).
    2. Find a basis for the row space using Gaussian elimination.
    3. Synthesize the basis vectors using a recursive balanced tree approach to minimize depth and maximize sharing.
    4. Express each original row as a linear combination of the basis vectors.
    5. Synthesize the outputs by XORing the corresponding basis wires.
    
    This approach leverages the linearity of XOR to share intermediate computations globally,
    which is more effective than local greedy methods for dense matrices.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    mat = np.array(matrix, dtype=np.int8)
    m, n = mat.shape
    
    zero_wire = n
    
    # Compute row weights and baseline
    row_weights = np.sum(mat, axis=1)
    baseline_gates = int(np.sum(np.maximum(0, row_weights - 1)))
    
    # Handle trivial cases
    if baseline_gates == 0:
        gates = []
        outputs = []
        for i in range(m):
            if row_weights[i] == 0:
                outputs.append(zero_wire)
            else:
                idx = np.where(mat[i] == 1)[0][0]
                outputs.append(int(idx))
        return {'gates': gates, 'outputs': outputs}
    
    # Convert rows to bitmasks for efficient GF(2) operations
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if mat[i, j] == 1:
                mask |= (1 << j)
        target_masks.append(mask)
    
    # Step 1: Find a basis for the row space
    basis = []
    basis_indices = []  # Track which original rows form the basis
    
    # Gaussian elimination to find basis
    rows = target_masks[:]
    for i in range(m):
        # Skip if row is zero
        if rows[i] == 0:
            continue
        
        # Try to eliminate using existing basis
        reduced = rows[i]
        for b_mask, b_idx in basis:
            # If the highest set bit of reduced is in b_mask, XOR them
            if b_mask & (1 << (n - 1)):  # This is a simplification, proper pivot needed
                pass
        
        # Proper Gaussian Elimination:
        # We want a basis where each basis vector has a unique pivot bit
        # Let's restart the basis finding with a cleaner algorithm
        
    # Cleaner Basis Finding
    basis = []
    for i in range(m):
        r = target_masks[i]
        if r == 0:
            continue
        # Try to reduce r by existing basis
        for j, (b_mask, b_idx) in enumerate(basis):
            # Find the highest set bit in b_mask
            # If r has that bit set, XOR b_mask into r
            # To make this efficient, we maintain basis in reduced row echelon form
            # Let's just do standard elimination
            # Find pivot of b_mask
            # For simplicity in this implementation, we'll just collect linearly independent rows
            # and then build a tree for each.
            # A more sophisticated approach would use RREF.
            # Given time constraints, let's use a simpler heuristic:
            # If r is not a linear combination of previous rows, add it to basis.
            # We can check independence by trying to reduce r to 0 using previous basis.
            
            # Actually, let's just use numpy to find the rank and a basis?
            # No, let's stick to bitmasks.
            
            # Let's implement a simple RREF-like basis construction
            # This is O(m * rank * n) which is fine for small n.
            
            # Let's restart the basis construction properly
            break
            
    # Let's use a dedicated function for basis construction
    basis, basis_combinations = _build_basis(target_masks, n)
    
    # Step 2: Synthesize basis vectors
    # We have a set of basis vectors. We want to compute them with shared sub-expressions.
    # Since they are linearly independent, we can't share much between them directly 
    # unless we decompose them further.
    # However, the "balanced tree" approach for each basis vector ensures minimal depth.
    
    # Let's try a different sharing strategy:
    # Instead of just basis, look for common sub-expressions in the target masks.
    # But for now, let's implement the basis decomposition correctly.
    
    # The key insight: If the matrix has low rank, the number of basis vectors is small.
    # Each output is a XOR of a subset of basis vectors.
    # If we compute the basis vectors first, then each output only needs 
    # (number of basis vectors in its combination - 1) gates.
    
    # To minimize total gates, we should:
    # 1. Compute basis vectors efficiently.
    # 2. For each output, XOR the required basis wires.
    
    # Let's refine _build_basis to return the combination coefficients
    # i.e., for each original row i, which basis vectors are XORed to form it.
    
    if not basis:
        # All rows are zero
        return {'gates': [], 'outputs': [zero_wire] * m}
        
    # Synthesize basis vectors
    # We will assign wire indices to the basis vectors.
    # To maximize sharing, we can use a global pool of intermediate wires.
    # But since basis vectors are independent, there is no sharing *between* them 
    # in the linear algebra sense. The sharing comes from the fact that 
    # we don't re-compute the same sub-expression within a single vector.
    
    # Wait, the current program's weakness is that it doesn't share *between* outputs 
    # effectively. The basis approach helps if the rank is low.
    
    # Let's implement a "Common Subexpression Elimination" (CSE) approach 
    # on the set of all target masks.
    
    # New Strategy:
    # 1. Identify all unique masks in the target set.
    # 2. For each unique mask, try to find a decomposition into two available masks 
    #    (inputs or previously computed) that minimizes cost.
    # 3. This is similar to the current program but applied globally to unique masks.
    
    # Actually, the current program *does* this, but it processes targets one by one.
    # If we process all unique masks together, we might find better sharing.
    
    # Let's try a "BFS from inputs" approach to find the cheapest way to compute 
    # each unique target mask.
    
    unique_masks = list(set(target_masks))
    
    # Map mask -> wire
    mask_to_wire = {}
    for i in range(n):
        mask_to_wire[1 << i] = i
    mask_to_wire[0] = zero_wire
    
    gates = []
    next_wire = n + 1
    
    # We want to compute all unique_masks.
    # Use a priority queue of "potential" decompositions?
    # Or just iterate and for each mask, find the best split.
    
    # To ensure we find good splits, we can do a multi-source BFS in the mask space?
    # No, that's too slow for large n.
    
    # Let's stick to the Basis approach but implement it correctly.
    
    # 1. Get Basis and Coefficients
    basis_masks, coefficients = _get_rref_basis(target_masks, n)
    
    # basis_masks: list of masks
    # coefficients: list of lists, coeff[i][j] is 1 if basis[j] is in row i
    
    # 2. Synthesize Basis Masks
    # We will compute each basis mask. Since they are independent, we compute them 
    # using a balanced tree structure to minimize depth and gate count per vector.
    
    basis_wires = []
    for b_mask in basis_masks:
        wire = _synthesize_single_mask(b_mask, mask_to_wire, gates, next_wire)
        basis_wires.append(wire)
        # Update mask_to_wire for the new wire
        mask_to_wire[b_mask] = wire
        # Note: next_wire is updated inside _synthesize_single_mask? 
        # No, it's passed by value. We need to return the new next_wire.
        
    # Let's rewrite _synthesize_single_mask to return (wire, new_next_wire)
    
    # Let's restart the synthesis part with correct state management.
    
    mask_to_wire = {}
    for i in range(n):
        mask_to_wire[1 << i] = i
    mask_to_wire[0] = zero_wire
    
    gates = []
    next_wire = n + 1
    
    basis_masks, coefficients = _get_rref_basis(target_masks, n)
    
    basis_wires = []
    for b_mask in basis_masks:
        wire, next_wire = _synthesize_single_mask(b_mask, mask_to_wire, gates, next_wire)
        basis_wires.append(wire)
        mask_to_wire[b_mask] = wire
        
    # 3. Synthesize Outputs
    outputs = []
    for i in range(m):
        # Find which basis vectors are needed
        needed = [j for j in range(len(basis_wires)) if coefficients[i][j] == 1]
        
        if not needed:
            outputs.append(zero_wire)
        elif len(needed) == 1:
            outputs.append(basis_wires[needed[0]])
        else:
            # XOR the needed basis wires
            # To minimize gates, we can just chain them.
            # Or use a balanced tree if there are many.
            # Since rank is usually small, chaining is fine.
            current = basis_wires[needed[0]]
            for j in range(1, len(needed)):
                other = basis_wires[needed[j]]
                gates.append([current, other])
                current = next_wire
                next_wire += 1
            outputs.append(current)
            
    return {'gates': gates, 'outputs': outputs}

def _get_rref_basis(masks, n):
    """
    Compute RREF basis for the set of masks and return coefficients.
    Returns:
    basis_masks: list of masks in the basis
    coefficients: list of lists, coeff[i][j] indicates if basis[j] is in mask[i]
    """
    # We'll perform Gaussian elimination on the rows (masks)
    # We want to transform the matrix into RREF.
    
    # Copy masks
    rows = masks[:]
    m = len(rows)
    
    # coeff[i][j] will track the linear combination
    # Initially, row i is just itself.
    # We'll maintain a list of "expressions" for each row.
    # expr[i] is a bitmask of length m (or rank) indicating which original rows are used.
    # But we want coefficients relative to the *basis* vectors.
    
    # Alternative: 
    # 1. Find pivot rows.
    # 2. For each pivot row, record its mask.
    # 3. For each original row, determine which pivot rows sum to it.
    
    # Let's do standard elimination and track the linear combinations.
    
    # expr[i] is an integer bitmask of length m, where bit j is 1 if original row j is in the current row i.
    exprs = [1 << i for i in range(m)]
    
    pivot_cols = []
    pivot_rows = []
    
    for col in range(n):
        # Find a row with a 1 in this column
        pivot_row = -1
        for i in range(m):
            if (rows[i] >> col) & 1:
                pivot_row = i
                break
        
        if pivot_row == -1:
            continue
            
        # Swap pivot_row to the top of the remaining rows? 
        # No, we just use it to eliminate other rows.
        
        # Record this as a basis vector
        pivot_cols.append(col)
        pivot_rows.append(pivot_row)
        
        # Eliminate this column from all other rows
        for i in range(m):
            if i != pivot_row and ((rows[i] >> col) & 1):
                rows[i] ^= rows[pivot_row]
                exprs[i] ^= exprs[pivot_row]
                
    # Now, rows[i] are in RREF.
    # The basis vectors are the rows that have a pivot.
    # Actually, the non-zero rows in RREF form the basis.
    
    basis_indices = []
    for i in range(m):
        if rows[i] != 0:
            basis_indices.append(i)
            
    # The basis vectors are rows[basis_indices]
    # And their "original" composition is exprs[basis_indices]
    
    # Now, for each original row j, we need to find which basis vectors sum to it.
    # In RREF, the original row j is a linear combination of the basis vectors.
    # Specifically, if we look at the final state of the elimination, 
    # the row j in the final matrix (which is 0 for non-pivot rows? No, RREF keeps them)
    # Wait, in the loop above, we modified `rows` in place.
    # The final `rows` are the RREF of the original matrix.
    # The non-zero rows of RREF are the basis vectors.
    
    # Let's extract the basis vectors and their positions.
    basis_masks = []
    basis_pos_in_rows = []
    for i in range(m):
        if rows[i] != 0:
            basis_masks.append(rows[i])
            basis_pos_in_rows.append(i)
            
    # Now, for each original row j, we need to find the coefficients c_k such that 
    # original_row_j = XOR(c_k * basis_k)
    
    # This is equivalent to solving the system.
    # Since we have the RREF, we can just look at the "history" or re-derive.
    
    # A simpler way: 
    # The expression `exprs[i]` tells us how `rows[i]` (which is a basis vector or zero) 
    # is formed from the *original* rows.
    # We want the reverse: how each *original* row is formed from the *basis* vectors.
    
    # Let's build a matrix B where B[i][j] is 1 if basis vector j is in original row i.
    # We can compute this by solving the linear system.
    
    # Let's create a list of basis vectors as numpy arrays of length n.
    # And for each original row, we solve for the coefficients.
    
    # Since n is small (<=16), we can do this efficiently.
    
    coefficients = []
    for j in range(m):
        # original row j
        target = masks[j]
        # Find combination of basis_masks that equals target
        # This is a linear system: sum(c_k * basis_k) = target
        # We can use the fact that the basis is in RREF (almost).
        
        # Let's just do a simple search for small rank.
        # Rank is usually small.
        rank = len(basis_masks)
        
        # If rank is 0, target must be 0
        if rank == 0:
            if target == 0:
                coefficients.append([0] * 0)
            else:
                # Should not happen if basis is correct
                coefficients.append([0] * 0)
            continue
            
        # Solve for coefficients
        # We can use Gaussian elimination on the basis vectors to represent target.
        # Or, since we have the RREF of the *original* matrix, 
        # the row j in the RREF matrix `rows` is actually the representation of 
        # original row j in terms of the basis?
        
        # No, `rows` was modified. `rows[j]` is the RREF of original row j.
        # The non-zero rows of RREF are the basis vectors.
        # So, `rows[j]` is a linear combination of the basis vectors.
        # Specifically, if `rows[j]` has a 1 in the pivot column of basis vector k, 
        # then basis vector k is in the combination.
        
        # Let's map pivot columns to basis indices.
        pivot_to_basis_idx = {}
        for idx, b_mask in enumerate(basis_masks):
            # Find the pivot column of this basis vector
            # In RREF, each basis vector has a unique pivot column.
            for col in range(n):
                if (b_mask >> col) & 1:
                    pivot_to_basis_idx[col] = idx
                    break
                    
        coeff = [0] * rank
        for col in range(n):
            if (rows[j] >> col) & 1:
                # This bit in rows[j] comes from a basis vector
                if col in pivot_to_basis_idx:
                    coeff[pivot_to_basis_idx[col]] = 1
                # If it's not a pivot column, it must be 0 in RREF for that row?
                # No, RREF means 1s in pivot cols, 0s above and below.
                # So if col is not a pivot col, rows[j] should have 0 there?
                # Actually, RREF has 1s in pivot positions and 0s elsewhere in that column.
                # So yes, only pivot columns can be 1.
        
        coefficients.append(coeff)
        
    return basis_masks, coefficients

def _synthesize_single_mask(mask, mask_to_wire, gates, next_wire):
    """
    Synthesize a single mask using a balanced tree approach.
    Returns (wire, new_next_wire).
    """
    if mask in mask_to_wire:
        return mask_to_wire[mask], next_wire
        
    # Find the set bits
    bits = [i for i in range(64) if (mask >> i) & 1]
    
    if not bits:
        return 0, next_wire # Should be zero_wire, but 0 is not a wire index usually. 
                            # Wait, zero_wire is n. 
                            # If mask is 0, it should be in mask_to_wire already.
                            
    if len(bits) == 1:
        # It's an input
        return bits[0], next_wire
        
    # Recursive balanced split
    # Split bits into two roughly equal halves
    mid = len(bits) // 2
    left_bits = bits[:mid]
    right_bits = bits[mid:]
    
    left_mask = 0
    for b in left_bits:
        left_mask |= (1 << b)
        
    right_mask = 0
    for b in right_bits:
        right_mask |= (1 << b)
        
    left_wire, next_wire = _synthesize_single_mask(left_mask, mask_to_wire, gates, next_wire)
    right_wire, next_wire = _synthesize_single_mask(right_mask, mask_to_wire, gates, next_wire)
    
    gates.append([left_wire, right_wire])
    new_wire = next_wire
    next_wire += 1
    
    return new_wire, next_wire

# Fallback for safety
def _fallback(matrix, depth_limit):
    mat = np.array(matrix, dtype=np.int8)
    m, n = mat.shape
    zero_wire = n
    gates = []
    outputs = []
    next_wire = n + 1
    for i in range(m):
        ones = np.where(mat[i] == 1)[0]
        if len(ones) == 0:
            outputs.append(zero_wire)
        elif len(ones) == 1:
            outputs.append(int(ones[0]))
        else:
            current = int(ones[0])
            for j in range(1, len(ones)):
                other = int(ones[j])
                gates.append([current, other])
                current = next_wire
                next_wire += 1
            outputs.append(current)
    return {'gates': gates, 'outputs': outputs}

# Wrap the main synthesize with error handling and time check
def synthesize(matrix, depth_limit):
    start_time = time.time()
    try:
        result = _synthesize_impl(matrix, depth_limit)
        return result
    except Exception as e:
        # Fallback if something goes wrong
        return _fallback(matrix, depth_limit)

def _synthesize_impl(matrix, depth_limit):
    # Use the logic from above
    mat = np.array(matrix, dtype=np.int8)
    m, n = mat.shape
    zero_wire = n
    
    row_weights = np.sum(mat, axis=1)
    baseline_gates = int(np.sum(np.maximum(0, row_weights - 1)))
    
    if baseline_gates == 0:
        gates = []
        outputs = []
        for i in range(m):
            if row_weights[i] == 0:
                outputs.append(zero_wire)
            else:
                idx = np.where(mat[i] == 1)[0][0]
                outputs.append(int(idx))
        return {'gates': gates, 'outputs': outputs}
    
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if mat[i, j] == 1:
                mask |= (1 << j)
        target_masks.append(mask)
        
    basis_masks, coefficients = _get_rref_basis(target_masks, n)
    
    mask_to_wire = {}
    for i in range(n):
        mask_to_wire[1 << i] = i
    mask_to_wire[0] = zero_wire
    
    gates = []
    next_wire = n + 1
    
    basis_wires = []
    for b_mask in basis_masks:
        wire, next_wire = _synthesize_single_mask(b_mask, mask_to_wire, gates, next_wire)
        basis_wires.append(wire)
        mask_to_wire[b_mask] = wire
        
    outputs = []
    for i in range(m):
        needed = [j for j in range(len(basis_wires)) if coefficients[i][j] == 1]
        
        if not needed:
            outputs.append(zero_wire)
        elif len(needed) == 1:
            outputs.append(basis_wires[needed[0]])
        else:
            current = basis_wires[needed[0]]
            for j in range(1, len(needed)):
                other = basis_wires[needed[j]]
                gates.append([current, other])
                current = next_wire
                next_wire += 1
            outputs.append(current)
            
    return {'gates': gates, 'outputs': outputs}

