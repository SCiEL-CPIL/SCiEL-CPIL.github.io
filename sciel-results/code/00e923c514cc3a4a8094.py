"""Exact XOR-circuit synthesis: minimize mean gate-count / independent-row baseline over 8x8,12x16,16x16 maps."""
import time
import numpy as np
import itertools
from collections import defaultdict

def _rref_gf2_with_pivots(M, pivot_order):
    """
    Compute RREF of M using a specific pivot column order.
    Returns:
        basis_indices: list of row indices in the original matrix that form the basis.
        rref_matrix: The reduced row echelon form (rank x cols).
        rank: The rank of the matrix.
    """
    rows, cols = M.shape
    M_work = M.copy()
    rank = 0
    basis_indices = []
    pivot_cols = []
    
    # We need to pick pivots in the order specified by pivot_order.
    # pivot_order is a permutation of range(cols).
    # For each column in pivot_order, if it can be a pivot (i.e., there's a 1 in that column 
    # below the current rank that hasn't been eliminated), we make it a pivot.
    
    # Standard RREF algorithm but iterating columns in pivot_order.
    # Note: RREF is unique regardless of pivot order, but the *basis rows* we pick 
    # (which original rows correspond to the pivot rows in the reduced form) 
    # depend on the elimination path.
    
    # Actually, a more direct way to get different bases:
    # Perform Gaussian elimination, but when we have a choice of pivot row, 
    # the standard algorithm picks the first one. 
    # To get different bases, we can permute the columns before doing RREF.
    # The RREF of M with columns permuted by P is equivalent to RREF of M * P.
    # The pivot columns in M*P correspond to pivot rows in the original M?
    # No, the basis vectors are linear combinations of the original rows.
    
    # Let's stick to the strategy from Inspiration 1:
    # Enumerate pivot sets. 
    # A simpler and effective approach for small matrices:
    # 1. Compute the standard RREF.
    # 2. The rank is r.
    # 3. The "independent rows" are the pivot rows in the RREF process.
    # However, different choices of pivot rows (from the set of rows with 1s in the pivot column)
    # lead to different intermediate matrices and potentially different "costs" if we view the 
    # synthesis as building the basis vectors first.
    
    # Actually, the most impactful structural change is:
    # Instead of just using the rows of the matrix directly, we compute a basis B.
    # The cost is:
    # 1. Cost to synthesize each basis vector from inputs.
    # 2. Cost to synthesize each original row from the basis vectors.
    
    # If we use the standard RREF, the basis vectors are the pivot rows of the RREF.
    # But we can choose *any* set of r independent rows from the original matrix as the basis.
    # Let's try a few random subsets of independent rows as the basis.
    
    pass # This function signature was a placeholder for the thought process. 
         # The actual implementation below uses a different approach.

def _cost_of_basis_and_outputs(M, basis_indices):
    """
    Calculate the total gate count if we use `basis_indices` as the basis.
    1. Synthesize each basis vector from inputs (using a tree).
    2. Determine the coefficients of each original row in terms of the basis.
    3. Synthesize each original row from the basis wires (using a tree).
    
    Returns total gate count.
    """
    rows, cols = M.shape
    n_inputs = cols
    
    # Step 1: Synthesize basis vectors
    # basis_indices is a list of row indices.
    # We need to build a circuit for each basis vector.
    # We can share subexpressions between basis vectors? 
    # For simplicity and speed in search, let's assume independent trees for basis vectors first,
    # then see if we can improve.
    # Actually, the "independent-row baseline" in the task description implies we are comparing 
    # against the cost of just XORing the inputs for each row independently? 
    # "Minimize mean gate-count / independent-row baseline"
    # Baseline: For each row, popcount(row) - 1 gates (if popcount > 1).
    # Or is it the cost of a naive tree for each row?
    
    # Let's assume the baseline is the sum of (popcount(row) - 1) for all rows.
    # We want to minimize the actual gate count.
    
    # To get the coefficients of original rows in terms of the chosen basis:
    # We need to solve M * x = 0? No.
    # We need to express each row m_i as a linear combination of basis rows b_j.
    # Let B be the matrix formed by rows basis_indices.
    # We need coefficients c_i such that m_i = sum(c_i[j] * b_j).
    # This is equivalent to solving B^T * c_i^T = m_i^T? No, B * c_i = m_i.
    # Since B is square (r x r) and invertible (full rank), we can solve for c_i.
    
    # However, doing this for every candidate basis in a search loop is expensive.
    # Let's optimize:
    # 1. Compute the RREF of M. Let the pivot rows be P.
    # 2. The standard basis is the rows of RREF corresponding to pivots.
    # 3. Any other basis can be obtained by changing the pivot rows.
    
    # Alternative Strategy:
    # Use the RREF to identify the rank r.
    # The cost of a circuit is roughly:
    # - Gates to create the "basis" wires.
    # - Gates to combine basis wires to form outputs.
    
    # A very effective heuristic for XOR synthesis is to use the RREF rows as the basis.
    # The "cost" of the basis rows is the sum of their popcounts (minus 1 for each).
    # The "cost" of the outputs is the sum of the popcounts of the coefficient vectors 
    # (which are the rows of the transformation matrix from RREF to original M).
    
    # By permuting columns before RREF, we change which columns become pivots, 
    # which changes the sparsity of the RREF and the transformation matrix.
    # We can try several column permutations (e.g., random shuffles, or sorted by column weight)
    # and pick the one that minimizes the estimated total cost.
    
    def _estimate_cost(M_perm):
        # Compute RREF of M_perm
        M_rref, rank, pivot_cols_in_perm = _rref_gf2_simple(M_perm)
        if rank == 0:
            return 0
        
        # The basis vectors are the rows of M_rref.
        # Cost of basis: sum(popcount(row) - 1 for row in M_rref if popcount > 0)
        # Note: If a row is all zeros, it's not a basis vector.
        basis_cost = 0
        for i in range(rank):
            pc = np.sum(M_rref[i])
            if pc > 1:
                basis_cost += pc - 1
            elif pc == 1:
                basis_cost += 0 # Just an input wire
        
        # Now we need the cost to express original rows in terms of the RREF basis.
        # The RREF process effectively computes:
        # M * P = E * R
        # Where E is an invertible matrix (product of elementary row ops) and R is RREF.
        # So M * P = E * R => M = E * R * P^-1.
        # The rows of M are linear combinations of rows of R.
        # The coefficients are given by the matrix E.
        # Specifically, row i of M is sum_j (E[i,j] * row_j of R).
        
        # We need to reconstruct E.
        # During RREF, we apply row operations to M. We can apply the same operations to I.
        # Let's re-run RREF but track E.
        
        M_work = M_perm.copy()
        I_work = np.eye(M_perm.shape[0], dtype=np.uint8)
        rows, cols = M_work.shape
        rank = 0
        for col in range(cols):
            if rank >= rows:
                break
            pivot = -1
            for r in range(rank, rows):
                if M_work[r, col]:
                    pivot = r
                    break
            if pivot == -1:
                continue
            M_work[[rank, pivot]] = M_work[[pivot, rank]]
            I_work[[rank, pivot]] = I_work[[pivot, rank]]
            for r in range(rows):
                if r != rank and M_work[r, col]:
                    M_work[r, :] ^= M_work[rank, :]
                    I_work[r, :] ^= I_work[rank, :]
            rank += 1
        
        # Now I_work contains E (the transformation matrix).
        # The first `rank` rows of I_work correspond to the basis vectors (RREF rows).
        # The rest of the rows express the non-pivot original rows in terms of the basis?
        # No, I_work is the full transformation matrix such that M_perm = I_work * RREF.
        # Wait, standard RREF does:
        # 1. Swap rows
        # 2. Eliminate below
        # 3. Eliminate above (for reduced form)
        
        # If we do full RREF (eliminate above and below), then:
        # M_perm = I_work * RREF
        # The rows of RREF are the basis.
        # The rows of I_work tell us how to combine the RREF rows to get the original rows (in permuted order).
        # Specifically, row i of M_perm is the linear combination of RREF rows given by row i of I_work.
        
        # Cost of outputs:
        # For each row i in I_work, the number of 1s in that row indicates how many basis wires need to be XORed.
        # If a row in I_work has k 1s, it costs k-1 gates (if k>0).
        # But we only care about the original `rows` rows of M.
        # Since M_perm is just M with columns permuted, the row space is the same, 
        # but the "rows" of M are fixed. 
        # Wait, M_perm has the same rows as M, just columns shuffled.
        # So row i of M_perm is row i of M (with columns shuffled).
        # The output i must correspond to row i of the original M.
        # So we need to express row i of M in terms of the basis.
        # The coefficients are given by row i of I_work.
        
        output_cost = 0
        for i in range(rows):
            pc = np.sum(I_work[i])
            if pc > 1:
                output_cost += pc - 1
            elif pc == 1:
                output_cost += 0
            else:
                output_cost += 0
        
        return basis_cost + output_cost

    # We need to implement _rref_gf2_simple that returns just the cost or the matrices.
    # Let's combine the logic.
    
    M_perm = M.copy()
    # Apply column permutation
    for i, idx in enumerate(pivot_order):
        M_perm[:, i] = M[:, idx]
        
    # Compute RREF and Transformation Matrix E
    rows, cols = M_perm.shape
    M_work = M_perm.copy()
    I_work = np.eye(rows, dtype=np.uint8)
    rank = 0
    for col in range(cols):
        if rank >= rows:
            break
        pivot = -1
        for r in range(rank, rows):
            if M_work[r, col]:
                pivot = r
                break
        if pivot == -1:
            continue
        M_work[[rank, pivot]] = M_work[[pivot, rank]]
        I_work[[rank, pivot]] = I_work[[pivot, rank]]
        for r in range(rows):
            if r != rank and M_work[r, col]:
                M_work[r, :] ^= M_work[rank, :]
                I_work[r, :] ^= I_work[rank, :]
        rank += 1
    
    # M_work is now RREF.
    # I_work is the transformation matrix.
    
    # Cost 1: Basis (rows of M_work)
    basis_cost = 0
    for i in range(rank):
        pc = np.sum(M_work[i])
        if pc > 1:
            basis_cost += pc - 1
            
    # Cost 2: Outputs (rows of I_work)
    # Note: I_work has `rows` rows. Each corresponds to an original row of M.
    output_cost = 0
    for i in range(rows):
        pc = np.sum(I_work[i])
        if pc > 1:
            output_cost += pc - 1
            
    return basis_cost + output_cost

def _rref_gf2_simple(M):
    """Placeholder, logic moved to _cost_of_basis_and_outputs for efficiency."""
    pass

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0
    
    M = np.array(matrix, dtype=np.uint8)
    n_rows, n_cols = M.shape
    
    if n_rows == 0:
        return {'gates': [], 'outputs': []}
        
    # Baseline: Independent synthesis of each row
    # Cost = sum(max(0, popcount(row) - 1))
    baseline_cost = 0
    for i in range(n_rows):
        pc = np.sum(M[i])
        if pc > 1:
            baseline_cost += pc - 1
            
    if baseline_cost == 0:
        # All rows are 0 or unit vectors. No gates needed.
        outputs = []
        for i in range(n_rows):
            ones = np.nonzero(M[i])[0]
            if len(ones) == 0:
                outputs.append(n_cols) # Zero wire
            else:
                outputs.append(ones[0])
        return {'gates': [], 'outputs': outputs}

    # Strategy: Try several column permutations to find a RREF that minimizes total cost.
    # Permutations to try:
    # 1. Natural order (0..n-1)
    # 2. Sorted by column weight (ascending)
    # 3. Sorted by column weight (descending)
    # 4. A few random permutations
    
    col_weights = np.sum(M, axis=0)
    
    perms_to_try = []
    perms_to_try.append(list(range(n_cols)))
    perms_to_try.append(list(np.argsort(col_weights)))
    perms_to_try.append(list(np.argsort(-col_weights)))
    
    # Add some random permutations
    rng = np.random.default_rng(42)
    for _ in range(10):
        p = list(range(n_cols))
        rng.shuffle(p)
        perms_to_try.append(p)
        
    best_perm = None
    best_cost = float('inf')
    
    for p in perms_to_try:
        if time.time() > deadline - 5.0:
            break
        cost = _cost_of_basis_and_outputs(M, p)
        if cost < best_cost:
            best_cost = cost
            best_perm = p
            
    if best_perm is None:
        best_perm = list(range(n_cols))
        
    # Now, actually build the circuit using the best_perm.
    # We need to:
    # 1. Compute RREF of M with columns permuted by best_perm.
    # 2. Track the transformation matrix I_work.
    # 3. Build wires for the basis rows (rows of RREF).
    # 4. Build wires for the outputs (combinations of basis wires given by I_work).
    
    M_perm = M.copy()
    for i, idx in enumerate(best_perm):
        M_perm[:, i] = M[:, idx]
        
    rows, cols = M_perm.shape
    M_work = M_perm.copy()
    I_work = np.eye(rows, dtype=np.uint8)
    rank = 0
    
    # We also need to know which columns are pivots to build the basis wires correctly.
    # The basis wires correspond to the rows of M_work (RREF).
    # Let's store the RREF rows.
    rref_rows = []
    
    for col in range(cols):
        if rank >= rows:
            break
        pivot = -1
        for r in range(rank, rows):
            if M_work[r, col]:
                pivot = r
                break
        if pivot == -1:
            continue
        M_work[[rank, pivot]] = M_work[[pivot, rank]]
        I_work[[rank, pivot]] = I_work[[pivot, rank]]
        for r in range(rows):
            if r != rank and M_work[r, col]:
                M_work[r, :] ^= M_work[rank, :]
                I_work[r, :] ^= I_work[rank, :]
        rank += 1
        rref_rows.append(M_work[rank-1].copy())
        
    # Now M_work is RREF, I_work is transformation.
    # rref_rows contains the basis vectors (in permuted column space).
    
    # Wire Mapping:
    # 0..n_cols-1: Inputs (in original order)
    # n_cols: Zero
    # n_cols+1..: Gates
    
    # We need to build the basis wires.
    # Each basis vector is a set of inputs (in permuted space).
    # We need to map permuted input indices back to original input indices.
    # perm[j] = original_col_index
    # So if basis vector has 1 at permuted col j, it uses original input perm[j].
    
    wire_map = {} # frozenset of original input indices -> wire index
    for j in range(n_cols):
        wire_map[frozenset([j])] = j
    wire_map[frozenset()] = n_cols
    
    next_wire = n_cols + 1
    gates = []
    
    # Build basis wires
    basis_wires = []
    for i in range(rank):
        # Get the set of original input indices for this basis vector
        ones_in_perm = np.nonzero(rref_rows[i])[0]
        orig_indices = set()
        for j in ones_in_perm:
            orig_indices.add(best_perm[j])
        s = frozenset(orig_indices)
        
        if s in wire_map:
            basis_wires.append(wire_map[s])
        else:
            # Build a tree for this set
            # For simplicity, use a balanced tree. 
            # We can optimize by checking if subsets are in wire_map.
            # Given the small size, a simple tree is fine.
            level = [frozenset([x]) for x in s]
            while len(level) > 1:
                new_level = []
                for k in range(0, len(level) - 1, 2):
                    s1 = level[k]
                    s2 = level[k + 1]
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
            basis_wires.append(wire_map[level[0]])
            
    # Build output wires
    outputs = []
    for i in range(rows):
        # Get the set of basis indices that are XORed
        # I_work[i] is the coefficient vector.
        # 1s in I_work[i] indicate which basis rows are used.
        ones_in_basis = np.nonzero(I_work[i])[0]
        
        if len(ones_in_basis) == 0:
            outputs.append(n_cols)
        elif len(ones_in_basis) == 1:
            outputs.append(basis_wires[ones_in_basis[0]])
        else:
            # Build tree for these basis wires
            # We can share sub-trees if we map frozenset of basis_wires?
            # Since basis_wires are unique wires, we can just build a tree.
            wires_to_xor = [basis_wires[j] for j in ones_in_basis]
            level = wires_to_xor
            while len(level) > 1:
                new_level = []
                for k in range(0, len(level) - 1, 2):
                    w1 = level[k]
                    w2 = level[k + 1]
                    gates.append([w1, w2])
                    new_level.append(next_wire)
                    next_wire += 1
                if len(level) % 2 == 1:
                    new_level.append(level[-1])
                level = new_level
            outputs.append(level[0])
            
    return {'gates': gates, 'outputs': outputs}

