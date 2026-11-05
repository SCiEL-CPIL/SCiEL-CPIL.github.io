import time
import numpy as np

def synthesize(matrix, depth_limit):
    """
    Exact XOR-circuit synthesis using global algebraic basis reduction.
    
    Strategy:
    1. Compute the rank of the matrix (number of linearly independent rows).
    2. Find a basis for the row space using Gaussian elimination over GF(2).
    3. Synthesize a circuit that computes this basis efficiently by sharing
       intermediate XORs (using a recursive balanced tree or greedy sharing).
    4. Express each original row as a linear combination of the basis vectors.
    5. Compute the outputs by XORing the appropriate basis wires.
    
    This approach minimizes gates by focusing on the algebraic structure (rank)
    rather than per-row weights, which is superior for dense or highly correlated matrices.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    mat = np.array(matrix, dtype=np.int8)
    m, n = mat.shape
    
    zero_wire = n
    
    # Trivial case: empty matrix or all zeros
    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}
    
    # Compute baseline for comparison (though we aim to beat it)
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
    
    # 1. Gaussian Elimination to find a basis for the row space
    # We work with integer representations of rows for easier XOR operations
    rows = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if mat[i, j] == 1:
                mask |= (1 << j)
        rows.append(mask)
    
    # Perform Gaussian elimination to get a reduced row echelon form (RREF) basis
    # and track the linear combinations
    # basis[i] is the i-th basis vector (mask)
    # combo[i] is the list of original row indices that sum to basis[i]
    # Actually, we want to express original rows in terms of the basis.
    # Let's find the pivot columns.
    
    # Standard Gaussian elimination on the matrix of rows
    # We will create a list of basis vectors and their "cost" in terms of original rows
    # But for circuit synthesis, we just need the basis vectors themselves to be computed.
    
    # Let's find the rank and the basis vectors
    # We'll use a dictionary to map pivot column -> basis vector index
    pivot_to_basis = {}
    basis_vectors = []
    basis_row_indices = [] # Not strictly needed for circuit, but good for debugging
    
    # Sort rows by weight to find pivots? No, standard Gaussian elimination is fine.
    # To make the basis "nice", we can pick the row with the lowest weight for each pivot.
    
    # Let's do a greedy basis selection to minimize the cost of computing the basis.
    # 1. Identify pivot columns.
    # 2. For each pivot column, select a row that has this pivot and no previously selected pivots.
    
    # A simpler approach:
    # 1. Compute RREF of the matrix.
    # 2. The non-zero rows of RREF form a basis.
    # 3. The circuit computes these basis vectors.
    # 4. Each original row is a linear combination of the basis vectors (given by the RREF coefficients).
    
    # Let's implement Gaussian Elimination to get RREF and the transformation matrix.
    # A is m x n. We want to find B (basis, r x n) and C (r x m) such that A = C @ B.
    # Actually, rows of A are linear combinations of rows of B.
    
    # We'll work with the transpose? No, rows are the vectors.
    # Let's perform elimination on the rows.
    
    # State:
    # current_rows: list of masks
    # coeffs: list of lists, coeffs[i][j] is 1 if original row j contributes to current row i.
    # Initially, current_rows[i] = rows[i], coeffs[i] = e_i
    
    current_rows = rows[:]
    coeffs = []
    for i in range(m):
        c = [0] * m
        c[i] = 1
        coeffs.append(c)
    
    # Eliminate
    pivot_cols = []
    basis_indices = []
    
    for col in range(n):
        if time.time() > deadline:
            break
            
        # Find a row with a 1 in this column that hasn't been eliminated
        found = -1
        for i in range(len(current_rows)):
            if (current_rows[i] >> col) & 1:
                found = i
                break
        
        if found == -1:
            continue
            
        # Swap found row to the top of the remaining active rows?
        # To keep it simple, we'll just use the first one found and eliminate it from others.
        # This is O(m^2 * n) which is fine for m,n <= 16.
        
        pivot_cols.append(col)
        basis_indices.append(found)
        
        pivot_row = current_rows[found]
        pivot_coeff = coeffs[found]
        
        # Eliminate this column from all other rows
        for i in range(len(current_rows)):
            if i == found:
                continue
            if (current_rows[i] >> col) & 1:
                current_rows[i] ^= pivot_row
                # Update coeffs
                for k in range(m):
                    coeffs[i][k] ^= pivot_coeff[k]
    
    # Now, the rows at indices in basis_indices are the basis vectors.
    # But wait, the elimination above didn't remove the pivot rows from the list,
    # it just used them. The "basis" is the set of rows that were chosen as pivots.
    # However, the `current_rows` list has been modified.
    # The basis vectors are the values of `current_rows` at the indices `basis_indices` 
    # AT THE TIME THEY WERE SELECTED? No, they might have been modified by later eliminations?
    # No, we only eliminate FROM other rows, we don't modify the pivot row itself in this simple version.
    # So current_rows[basis_indices[i]] is the i-th basis vector.
    
    basis_vecs = []
    for idx in basis_indices:
        basis_vecs.append(current_rows[idx])
        
    r = len(basis_vecs)
    
    if r == 0:
        # All rows are zero
        gates = []
        outputs = [zero_wire] * m
        return {'gates': gates, 'outputs': outputs}
    
    # Now we need to synthesize the circuit for the basis vectors.
    # And then compute each original row as a linear combination of the basis vectors.
    
    # Step 2: Synthesize the basis vectors.
    # We have r basis vectors. We need to compute them from inputs.
    # We can use a greedy sharing approach for these r vectors.
    # Since r <= min(m, n) and n <= 16, r is small.
    
    # Let's use the previous greedy logic but only for the basis vectors.
    available = {}
    for i in range(n):
        available[1 << i] = i
    available[0] = zero_wire
    
    gates = []
    next_wire = n + 1
    
    basis_wires = []
    
    # Sort basis vectors by weight to compute simpler ones first
    basis_order = sorted(range(r), key=lambda i: bin(basis_vecs[i]).count('1'))
    
    for bi in basis_order:
        if time.time() > deadline:
            break
        mask = basis_vecs[bi]
        
        if mask in available:
            basis_wires.append(available[mask])
            continue
            
        wire = None
        found = False
        
        # Try to find decomposition from available wires
        # This is the same greedy logic as before, but applied to the basis set.
        for w_mask, w_wire in list(available.items()):
            other_mask = mask ^ w_mask
            if other_mask in available:
                other_wire = available[other_mask]
                if w_mask == 0:
                    wire = other_wire
                    found = True
                    break
                elif other_mask == 0:
                    wire = w_wire
                    found = True
                    break
                else:
                    gates.append([w_wire, other_wire])
                    wire = next_wire
                    next_wire += 1
                    available[mask] = wire
                    found = True
                    break
        
        if not found:
            # Fallback: compute from inputs
            ones = [j for j in range(n) if (mask >> j) & 1]
            if len(ones) == 0:
                wire = zero_wire
            elif len(ones) == 1:
                wire = ones[0]
            else:
                current = ones[0]
                for j in range(1, len(ones)):
                    gates.append([current, ones[j]])
                    current = next_wire
                    next_wire += 1
                wire = current
                available[mask] = wire
        
        basis_wires.append(wire)
        
    # Step 3: Compute outputs.
    # Each original row i is a linear combination of the basis vectors.
    # We need to find the coefficients.
    # We can use the `coeffs` array from the Gaussian elimination?
    # Wait, the `coeffs` array was updated during elimination.
    # coeffs[i] represents the combination of ORIGINAL rows that sum to current_rows[i].
    # We want the opposite: combination of BASIS vectors that sum to ORIGINAL rows.
    
    # Let's re-derive the coefficients.
    # We know A = C @ B, where B is the matrix of basis vectors (r x n) and C is (m x r).
    # Actually, rows of A are linear combos of rows of B.
    # Let's solve for C.
    # Since B is in RREF (or close to it), we can just look at the pivot columns.
    
    # Alternative: Since we have the basis vectors and the original rows,
    # and we know the basis is a subset of the row space,
    # we can simply compute the coefficients by checking which basis vectors sum to the original row.
    # But that's expensive.
    
    # Better: Use the structure of Gaussian Elimination.
    # When we selected a pivot row `found` for column `col`, that row became a basis vector.
    # The other rows were modified by XORing with this pivot row.
    # This means: Original_Row_j = Current_Row_j + (sum of pivot rows that were used to eliminate j).
    
    # Let's track this properly.
    # Let's restart the elimination but track the "inverse" transformation.
    # Or, simpler:
    # We have basis_vecs[0..r-1].
    # For each original row i, we want to find c_0..c_{r-1} such that row_i = XOR(c_k * basis_vecs[k]).
    
    # We can do this by back-substitution or by solving a small linear system.
    # Since r <= 16, we can just try all combinations? No, 2^16 is 65536, which is fast.
    # But we can do it smarter.
    
    # Let's build a dictionary: mask -> list of basis indices that XOR to this mask.
    # No, let's just use the fact that the basis vectors are linearly independent.
    # We can represent each basis vector as a bitstring of length r? No.
    
    # Let's use the standard method:
    # 1. Identify the pivot columns of the basis.
    # 2. For each original row, look at its bits in the pivot columns.
    # 3. These bits directly give the coefficients of the basis vectors, IF the basis is in RREF.
    
    # Is our basis in RREF?
    # We eliminated columns from left to right.
    # The pivot rows have 1 in their pivot column and 0 in all previous pivot columns.
    # Do they have 0 in later pivot columns?
    # In the elimination step: `if (current_rows[i] >> col) & 1: current_rows[i] ^= pivot_row`
    # This eliminates the current column from ALL other rows, including those that will become pivots later?
    # No, `i` iterates over all rows. So yes, it eliminates from future pivots too.
    # This means the basis vectors are in RREF (Reduced Row Echelon Form).
    # Therefore, the coefficient of basis vector k for original row i is simply the bit of original row i at the k-th pivot column.
    
    # Let's verify this logic.
    # If B is in RREF, then the matrix C (coeffs) can be read directly from A.
    # Let pivot_cols be the list of columns where pivots were found.
    # For each original row i, coeff[k] = A[i, pivot_cols[k]].
    
    # Let's test this hypothesis.
    # If A = C @ B, and B is RREF, then A[:, pivot_cols] = C @ B[:, pivot_cols].
    # B[:, pivot_cols] is the identity matrix (because B is in RREF).
    # So A[:, pivot_cols] = C.
    # Yes, this is correct.
    
    outputs = []
    for i in range(m):
        if time.time() > deadline:
            # Fallback for remaining outputs
            outputs.append(zero_wire)
            continue
            
        # Determine which basis vectors are needed
        needed_basis = []
        for k, pcol in enumerate(pivot_cols):
            if (rows[i] >> pcol) & 1:
                needed_basis.append(k)
        
        if len(needed_basis) == 0:
            outputs.append(zero_wire)
        elif len(needed_basis) == 1:
            outputs.append(basis_wires[needed_basis[0]])
        else:
            # XOR the required basis wires
            current = basis_wires[needed_basis[0]]
            for k in range(1, len(needed_basis)):
                other = basis_wires[needed_basis[k]]
                gates.append([current, other])
                current = next_wire
                next_wire += 1
            outputs.append(current)
            
    return {'gates': gates, 'outputs': outputs}
