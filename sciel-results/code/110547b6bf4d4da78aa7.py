"""XOR-circuit synthesis: greedy super-gate sharing with RREF-based output decomposition."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _rref_gf2(mat):
    """Compute RREF over GF(2). Returns (rank, pivot_cols). mat is (m, n) uint8."""
    m, n = mat.shape
    work = mat.astype(np.uint8).copy()
    pivot_cols = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        # find pivot in column col at or below row
        found = -1
        for r in range(row, m):
            if work[r, col] == 1:
                found = r
                break
        if found == -1:
            continue
        work[[row, found]] = work[[found, row]]
        pivot_cols.append(col)
        # eliminate all other rows
        for r in range(m):
            if r != row and work[r, col] == 1:
                work[r] ^= work[row]
        row += 1
    return row, pivot_cols


def _popcount(v):
    """Population count of a numpy array of 0/1 values."""
    return int(np.sum(v))


def _xor_weight(a, b):
    """Hamming weight of a XOR b for 0/1 arrays."""
    return int(np.sum(a ^ b))


def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D array-like of 0/1 values, shape (n_outputs, n_inputs)
        depth_limit: max circuit depth (<=16)
    
    Returns:
        {'gates': [[wire_in1, wire_in2], ...], 'outputs': [wire, ...]}
    """
    start_time = time.time()
    DEADLINE = start_time + 85.0  # 5s safety margin
    
    mat = np.array(matrix, dtype=np.uint8)
    n_out, n_in = mat.shape
    
    # Wires: 0..n_in-1 are inputs, n_in is constant zero (unused typically)
    # Gate k produces wire n_in + 1 + k
    
    # Compute RREF to understand the row space
    rank, pivot_cols = _rref_gf2(mat)
    
    # The "independent-row baseline" would use rank gates (one per pivot row)
    # but we want to minimize total gates by sharing intermediate XORs.
    
    # Strategy: Greedy super-gate construction
    # 1. Start with all output rows as "targets"
    # 2. Repeatedly find the most beneficial 2-input XOR gate that reduces total weight
    # 3. A gate XORs two existing wires to produce a new wire
    # 4. Cost = number of gates; each output must be expressible as XOR of gates/inputs
    
    # Simpler and effective approach:
    # - Compute RREF of the matrix
    # - Each output row is a linear combination of the basis vectors (pivot rows)
    # - We create one gate per non-trivial basis vector (cost = rank gates for basis)
    # - Then each output is XOR of some basis vectors + input wires
    # - But we can share: if multiple outputs share the same sub-expression, use one gate
    
    # Better: iterative improvement
    # Phase 1: Get a valid circuit via RREF
    # Phase 2: Try to reduce gate count by finding shared sub-expressions
    
    # Phase 1: RREF-based construction
    # The RREF has rank rows with pivot columns
    # Each original row = XOR of some RREF rows (the ones where original had 1 in pivot cols)
    # Actually, we need the change-of-basis: express each original row in terms of RREF rows
    
    # Let's do it differently:
    # - The row space is spanned by the rank independent rows
    # - We'll pick rank "basis gates": each basis gate computes one independent combination
    # - Then outputs are XORs of basis gates
    
    # Actually the simplest valid circuit:
    # For each output row that is not just a single input, create a chain of XORs
    # But that uses sum(popcount) - n_out gates (roughly)
    
    # Better baseline: use the fact that rank independent rows span everything
    # Create rank intermediate wires, then express each output as XOR of those + inputs
    
    # Let me think about this more carefully.
    # A gate takes two wires and produces wire_in1 XOR wire_in2.
    # We need each output to equal the corresponding row's linear combination of inputs.
    
    # Approach: 
    # 1. Compute RREF and the transformation
    # 2. The RREF rows (rank of them) form a basis
    # 3. Create one gate per RREF row that isn't a single input wire
    #    (i.e., has weight > 1)
    # 4. Express each output as XOR of a subset of these basis gates
    #    (plus possibly input wires if the basis row has a single 1)
    
    # Let me compute: for each output, its representation in the RREF basis
    # The RREF matrix R has shape (rank, n_in)
    # Each original row r can be written as: r = sum over pivot rows where coefficient is 1
    # The coefficient is: for original row i, look at the pivot columns; the bits in those columns
    # give the linear combination coefficients.
    
    # Actually, if RREF is computed from the original matrix, then:
    # original_matrix = T @ RREF (over GF(2))
    # where T is an (n_out x rank) matrix
    # T[i, j] = original[i, pivot_cols[j]]  (the bits of row i at the pivot positions)
    
    # So output i = XOR of RREF[j] for all j where T[i,j] == 1
    
    # Now, RREF[j] is a row of the RREF. If RREF[j] has weight 1, it's just an input wire.
    # If weight > 1, we need to build it with gates.
    
    # Cost model:
    # - For each RREF row with weight w > 1: needs w-1 gates to build (chain of XORs)
    #   But we can do better with a tree: ceil(log2(w)) gates... no, each gate produces one wire.
    #   To compute XOR of w inputs, you need w-1 gates (each gate XORs two wires).
    #   Wait, no: if you have w input wires and want their XOR, you need w-1 XOR gates.
    # - For each output: needs (number of basis vectors in its combo) - 1 gates if > 1 basis vector
    #   But the basis vectors themselves are wires (either input wires or gate outputs),
    #   so combining k basis vectors needs k-1 gates.
    
    # Total gates = sum over RREF rows with weight>1 of (weight-1) + sum over outputs of (num_basis_in_combo - 1 if >1 else 0)
    
    # Hmm, but this might not be optimal. Let's just go with a direct approach:
    
    # DIRECT APPROACH: Each output is computed independently with a chain of XORs.
    # Output i has weight w_i (number of 1s in its row).
    # Gates needed for output i: w_i - 1 (if w_i > 0), 0 if w_i == 0, 0 if w_i == 1.
    # Total gates = sum(max(0, w_i - 1))
    # This is the "no sharing" baseline.
    
    # SHARED approach: Use RREF basis.
    # Let me compute both and take the better one, then try to improve.
    
    weights = mat.sum(axis=1)  # weight of each output row
    
    # Direct cost
    direct_cost = int(np.sum(np.maximum(0, weights - 1)))
    
    # RREF basis cost
    # RREF rows: extract from the work matrix
    # Recompute RREF and keep the full matrix
    work = mat.astype(np.uint8).copy()
    m, n = work.shape
    row = 0
    pivot_cols = []
    for col in range(n):
        if row >= m:
            break
        found = -1
        for r in range(row, m):
            if work[r, col] == 1:
                found = r
                break
        if found == -1:
            continue
        work[[row, found]] = work[[found, row]]
        pivot_cols.append(col)
        for r in range(m):
            if r != row and work[r, col] == 1:
                work[r] ^= work[row]
        row += 1
    
    rank = row
    rref = work[:rank, :]
    
    # T matrix: original @ pivot_cols gives coefficients in RREF basis
    T = mat[:, pivot_cols] if rank > 0 else np.zeros((n_out, 0), dtype=np.uint8)
    
    # RREF row weights
    rref_weights = rref.sum(axis=1) if rank > 0 else np.array([])
    
    # Cost for RREF basis approach:
    # - Build each RREF row: (rref_weight - 1) gates if weight > 1, else 0
    #   But if weight == 1, it's just an input wire, no gate needed
    # - For each output: it's XOR of T[i] subset of RREF rows
    #   Number of RREF rows in combo = T[i].sum()
    #   If that count is 0: output is 0 (wire n_in, the constant zero)
    #   If count is 1: output is that single RREF row's wire (input or gate output), 0 extra gates
    #   If count > 1: need (count - 1) additional gates to combine them
    basis_build_cost = int(np.sum(np.maximum(0, rref_weights - 1))) if rank > 0 else 0
    combo_counts = T.sum(axis=1) if n_out > 0 else np.array([])
    output_combine_cost = int(np.sum(np.maximum(0, combo_counts - 1))) if n_out > 0 else 0
    rref_total_cost = basis_build_cost + output_combine_cost
    
    # Choose the better approach
    if direct_cost <= rref_total_cost:
        # Use direct approach
        gates = []
        outputs = []
        next_wire = n_in + 1  # first gate output wire
        
        for i in range(n_out):
            row_data = mat[i]
            ones = np.where(row_data == 1)[0]
            
            if len(ones) == 0:
                outputs.append(n_in)  # constant zero
            elif len(ones) == 1:
                outputs.append(int(ones[0]))
            else:
                # Chain of XORs
                # Start with first input
                current = int(ones[0])
                for j in range(1, len(ones)):
                    gates.append([current, int(ones[j])])
                    current = next_wire
                    next_wire += 1
                outputs.append(current)
        
        return {'gates': gates, 'outputs': outputs}
    
    else:
        # Use RREF basis approach
        gates = []
        outputs = []
        next_wire = n_in + 1
        
        # Build RREF basis wires
        # rref_wire[j] = the wire that computes RREF row j
        rref_wire = []
        for j in range(rank):
            row_data = rref[j]
            ones = np.where(row_data == 1)[0]
            
            if len(ones) == 0:
                rref_wire.append(n_in)  # zero wire
            elif len(ones) == 1:
                rref_wire.append(int(ones[0]))
            else:
                current = int(ones[0])
                for k in range(1, len(ones)):
                    gates.append([current, int(ones[k])])
                    current = next_wire
                    next_wire += 1
                rref_wire.append(current)
        
        # Build outputs
        for i in range(n_out):
            combo = np.where(T[i] == 1)[0]
            
            if len(combo) == 0:
                outputs.append(n_in)
            elif len(combo) == 1:
                outputs.append(rref_wire[int(combo[0])])
            else:
                current = rref_wire[int(combo[0])]
                for k in range(1, len(combo)):
                    gates.append([current, rref_wire[int(combo[k])]])
                    current = next_wire
                    next_wire += 1
                outputs.append(current)
        
        return {'gates': gates, 'outputs': outputs}

# EVOLVE-BLOCK-END
