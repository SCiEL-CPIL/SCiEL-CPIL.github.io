"""XOR circuit synthesis with greedy sharing."""
import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    # Convert to numpy array over GF(2)
    A = np.array(matrix, dtype=np.uint8) % 2
    m, n = A.shape
    
    # Handle trivial cases
    if m == 0:
        return {'gates': [], 'outputs': []}
    if n == 0:
        # All outputs are 0 (constant-zero wire)
        return {'gates': [], 'outputs': [n] * m}
    
    # Gaussian elimination to find rank and express rows in terms of basis
    # Work with rows as bit vectors
    rows = [list(row) for row in A]
    
    # Find pivot columns and basis
    basis = []  # list of (pivot_col, row_vector)
    row_to_basis = []  # for each original row, list of basis indices it's XOR of
    
    # Perform Gaussian elimination on rows
    working = [list(row) for row in rows]
    pivot_cols = []
    pivot_row_indices = []
    
    for col in range(n):
        # Find a row with 1 in this column that hasn't been used
        found = -1
        for i in range(len(pivot_cols), len(working)):
            if working[i][col] == 1:
                found = i
                break
        if found == -1:
            continue
        
        # Swap to pivot position
        pivot_pos = len(pivot_cols)
        working[pivot_pos], working[found] = working[found], working[pivot_pos]
        
        # Eliminate this column from other rows
        for i in range(len(working)):
            if i != pivot_pos and working[i][col] == 1:
                for j in range(n):
                    working[i][j] ^= working[pivot_pos][j]
        
        pivot_cols.append(col)
        pivot_row_indices.append(found)
    
    r = len(pivot_cols)  # rank
    
    if r == 0:
        return {'gates': [], 'outputs': [n] * m}
    
    # Now working[0:r] are the pivot rows (basis vectors)
    # Each original row can be expressed as XOR of some basis vectors
    # We need to track how each original row maps to the basis
    
    # Actually, let's redo this more carefully to track the linear combinations
    # Use a different approach: build the circuit directly
    
    # Strategy:
    # 1. Compute basis vectors (r of them) using XOR gates
    # 2. Express each output as XOR of basis vectors
    # 3. Share intermediate wires where possible
    
    # Wire numbering:
    # 0..n-1: inputs
    # n: constant zero
    # n+1, n+2, ...: gate outputs
    
    gates = []
    wire_count = n + 1  # next wire to assign
    
    # Step 1: Compute basis vectors
    # Each basis vector is a subset of inputs. XOR them together.
    basis_wires = []
    for i in range(r):
        vec = working[i]
        # Find which inputs are 1
        inputs_needed = [j for j in range(n) if vec[j] == 1]
        if not inputs_needed:
            basis_wires.append(n)  # zero
        elif len(inputs_needed) == 1:
            basis_wires.append(inputs_needed[0])
        else:
            # XOR all inputs together
            w = inputs_needed[0]
            for inp in inputs_needed[1:]:
                gates.append([w, inp])
                wire_count += 1
                w = wire_count - 1  # the new wire
            basis_wires.append(w)
    
    # Step 2: Express each original row as XOR of basis vectors
    # We need to find, for each original row, which basis vectors XOR to give it
    # This is equivalent to solving: A_row = sum of some basis vectors
    
    # The pivot columns give us a direct mapping. Let's use the reduced row echelon form.
    # In RREF, row i has a 1 in pivot_cols[i] and 0s in all other pivot columns.
    # So to express an original row, we look at its values in the pivot columns.
    
    # But we need to track how the original rows relate to the RREF rows.
    # Let me redo the elimination tracking the combinations.
    
    # Alternative: just solve the system for each output row
    # For output row y (length n), find x (length r) such that B @ x = y (mod 2)
    # where B is the r x n matrix of basis vectors
    
    # Build B
    B = np.array(working[:r], dtype=np.uint8) % 2
    
    # For each output row, solve B @ x = y
    outputs = []
    for i in range(m):
        y = A[i]
        # Solve B @ x = y mod 2
        # Since B has full row rank r, we can use the pivot structure
        # x[j] = y[pivot_cols[j]] in RREF... but we need to be careful
        
        # Actually in our elimination, working[:r] is in row echelon form but not necessarily RREF
        # Let's make it RREF
        B_rref = [list(row) for row in working[:r]]
        for i in range(r):
            # Make the pivot 1 (it already is) and eliminate above
            pivot_col = pivot_cols[i]
            for j in range(i):
                if B_rref[j][pivot_col] == 1:
                    for k in range(n):
                        B_rref[j][k] ^= B_rref[i][k]
        
        # Now in RREF, x[i] = y[pivot_cols[i]]
        x = [int(y[pivot_cols[i]]) for i in range(r)]
        
        # Check if solution is valid (it should be if y is in row space)
        # Compute XOR of selected basis wires
        selected = [basis_wires[i] for i in range(r) if x[i] == 1]
        
        if not selected:
            outputs.append(n)
        elif len(selected) == 1:
            outputs.append(selected[0])
        else:
            w = selected[0]
            for sw in selected[1:]:
                gates.append([w, sw])
                wire_count += 1
                w = wire_count - 1
            outputs.append(w)
    
    return {'gates': gates, 'outputs': outputs}
