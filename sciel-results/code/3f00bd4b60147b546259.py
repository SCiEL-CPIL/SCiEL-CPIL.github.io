"""Rewritten program for exact XOR-circuit synthesis."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 50.0  # Safety margin for 60s limit
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert to numpy array
    A = np.array(matrix, dtype=np.int8) % 2
    
    # Compute RREF over GF(2) to find independent rows and dependencies
    # We work with the row space: each output row is a linear combination of basis rows
    
    # Step 1: Find basis via Gaussian elimination over GF(2)
    # We track which original rows form the basis and the transformation
    
    # Work on augmented matrix: [A | I_m] to track row operations
    # But for large m, this is expensive. Let's use a smarter approach.
    
    # Actually, let's think about this differently.
    # The key insight: if we compute a set of "basis" linear combinations first,
    # then each output row can be expressed as XOR of some basis vectors.
    # The total cost = cost(basis vectors) + cost(combinations of basis vectors).
    
    # A good strategy:
    # 1. Find a basis for the row space using RREF.
    # 2. Compute each basis vector as an XOR tree of its input columns.
    # 3. For each output row, find which basis vectors XOR together to form it,
    #    then compute that as an XOR tree of the basis wires.
    
    # Let's implement RREF over GF(2)
    def rref_gf2(mat):
        """Compute RREF of a binary matrix. Returns (rref_matrix, pivot_cols, row_ops).
        row_ops[i] tells which original rows were used to form rref row i."""
        m, n = mat.shape
        # We'll track the transformation: each rref row = XOR of certain original rows
        # Start with identity transformation
        transform = np.eye(m, dtype=np.int8)
        rref = mat.copy()
        pivot_cols = []
        pivot_row = 0
        
        for col in range(n):
            if pivot_row >= m:
                break
            # Find pivot
            pivot = -1
            for row in range(pivot_row, m):
                if rref[row, col] == 1:
                    pivot = row
                    break
            if pivot == -1:
                continue
            
            # Swap rows
            if pivot != pivot_row:
                rref[[pivot_row, pivot]] = rref[[pivot, pivot_row]]
                transform[[pivot_row, pivot]] = transform[[pivot, pivot_row]]
            
            pivot_cols.append(col)
            
            # Eliminate all other rows in this column
            for row in range(m):
                if row != pivot_row and rref[row, col] == 1:
                    rref[row] = (rref[row] + rref[pivot_row]) % 2
                    transform[row] = (transform[row] + transform[pivot_row]) % 2
            
            pivot_row += 1
        
        return rref, pivot_cols, transform, pivot_row
    
    rref_mat, pivot_cols, transform, rank = rref_gf2(A)
    
    # Now, for each output row i, we need to express it in terms of the basis.
    # The RREF gives us: row i of A = XOR of basis rows j where rref[i,j] == 1
    # Wait, actually the RREF transformation works differently.
    # Let me reconsider.
    
    # Actually, let's use a different approach:
    # 1. Compute basis vectors from RREF.
    # 2. For each output, determine which basis vectors to XOR.
    
    # The RREF matrix has rank `rank` non-zero rows.
    # Each non-zero row of RREF corresponds to a basis vector.
    # The pivot columns tell us which input variables are in each basis vector.
    
    # Basis vectors: for each pivot row r (0 to rank-1),
    # the basis vector has 1s at pivot_cols[r] and at other columns where rref[r, c] == 1.
    
    # Now, to express original row i in terms of basis:
    # We need to solve: A[i] = sum_j (coeff[i,j] * basis[j])
    # This is equivalent to: in the RREF, row i of the original matrix expressed in RREF basis.
    
    # Actually, let's just compute it directly:
    # For each output row i, find which basis vectors (rows of RREF) XOR to give row i.
    
    # The basis vectors are the non-zero rows of RREF.
    # Let basis_vecs = [rref[j] for j in range(rank)]
    # For output i, we need coefficients c_0, ..., c_{rank-1} such that:
    #   sum_j c_j * basis_vecs[j] = A[i]
    # This is a system of n equations in rank unknowns.
    # Since the basis spans the row space, a solution exists.
    
    # We can solve this by projecting A[i] onto the basis.
    # Using the RREF structure: if we look at the pivot columns,
    # the coefficient for basis vector j is A[i, pivot_cols[j]].
    
    # Wait, that's not quite right either. Let me think more carefully.
    
    # In RREF, the pivot columns form an identity submatrix.
    # So if we take the values of A[i] at the pivot columns, those are exactly
    # the coefficients of the basis vectors.
    
    # Let's verify: basis vector j has 1 at pivot_cols[j] and possibly at other positions.
    # If we XOR together basis vectors j where A[i, pivot_cols[j]] == 1,
    # does that give us A[i]?
    
    # At pivot column p = pivot_cols[k]:
    #   The XOR gives A[i, p] (since only basis vector k has a 1 at pivot p, 
    #   and it's selected iff A[i,p]==1). ✓
    # At non-pivot column c:
    #   The XOR gives sum_j A[i, pivot_cols[j]] * rref[j, c]
    #   We need this to equal A[i, c].
    #   This is true because RREF is the reduced form, and the row space is preserved.
    
    # So: coefficients for output i are [A[i, pivot_cols[j]] for j in range(rank)]
    
    # Now let's build the circuit:
    next_wire = n + 1
    gates = []
    
    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w
    
    # Compute basis vectors
    basis_wires = []
    for j in range(rank):
        # Basis vector j: columns where rref[j, :] == 1
        cols = [c for c in range(n) if rref[j, c] == 1]
        if not cols:
            basis_wires.append(n)  # zero wire
            continue
        if len(cols) == 1:
            basis_wires.append(cols[0])
            continue
        # XOR tree
        wires = list(cols)
        while len(wires) > 1:
            new_wires = []
            for k in range(0, len(wires), 2):
                if k + 1 < len(wires):
                    new_wires.append(add_gate(wires[k], wires[k+1]))
                else:
                    new_wires.append(wires[k])
            wires = new_wires
        basis_wires.append(wires[0])
    
    # Compute outputs
    outputs = []
    for i in range(m):
        if time.time() > DEADLINE:
            outputs.append(n)
            continue
        
        # Get coefficients
        coeffs = [A[i, pivot_cols[j]] for j in range(rank)]
        selected = [basis_wires[j] for j in range(rank) if coeffs[j] == 1]
        
        if not selected:
            outputs.append(n)
        elif len(selected) == 1:
            outputs.append(selected[0])
        else:
            # XOR tree of selected basis wires
            wires = list(selected)
            while len(wires) > 1:
                new_wires = []
                for k in range(0, len(wires), 2):
                    if k + 1 < len(wires):
                        new_wires.append(add_gate(wires[k], wires[k+1]))
                    else:
                        new_wires.append(wires[k])
                wires = new_wires
            outputs.append(wires[0])
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
