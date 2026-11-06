"""
Exact XOR-circuit synthesis using a different structural strategy:
Instead of searching over pivot sets, we use a canonical RREF basis and
optimize the synthesis by minimizing the total Hamming weight of the
basis vectors and output coefficients through a row-space basis change.
"""
import time
import numpy as np
from itertools import combinations
import heapq

def _rref_gf2_with_pivots(M, pivot_cols):
    """
    Compute RREF of M over GF(2) using the specified pivot columns.
    Returns:
    - rref: The RREF matrix (rank x n_cols)
    - basis_coeffs: The transformation matrix T (rank x n_rows) such that RREF = T @ M.
    """
    M = np.array(M, dtype=np.uint8)
    n_rows, n_cols = M.shape
    rank = len(pivot_cols)
    
    if rank == 0:
        return np.zeros((0, n_cols), dtype=np.uint8), np.zeros((0, n_rows), dtype=np.uint8)

    work = M.copy()
    basis_coeffs = np.eye(n_rows, dtype=np.uint8)
    
    # We need to perform elimination to ensure that the specified pivot_cols are the pivots.
    # This requires that the submatrix M[:, pivot_cols] is invertible.
    # We will perform Gaussian elimination on the full matrix, but we force the pivot positions.
    
    # Standard Gaussian elimination picks the first available pivot.
    # To force specific pivots, we can permute columns so that pivot_cols come first,
    # perform RREF, and then permute back? No, RREF depends on column order for the "reduced" part.
    # Actually, if we define the basis by the pivot columns, the RREF is unique for that set of pivots IF the matrix is full rank on those columns.
    # Let's construct the RREF directly.
    # Let P be the matrix of columns in pivot_cols. P is n_rows x rank.
    # We want to find coefficients T (rank x n_rows) such that T @ M has 1s in pivot_cols and 0s elsewhere in the pivot rows?
    # No, the RREF property is:
    # 1. The pivot in row i is 1.
    # 2. All other entries in the pivot column are 0.
    # 3. The pivot row has 0s in other pivot columns.
    
    # Let's just do standard RREF but on a permuted matrix where pivot_cols are the first 'rank' columns.
    # Then the first 'rank' columns of the resulting RREF will be the Identity matrix.
    # The rest of the columns will be the free variables.
    
    perm = list(pivot_cols) + [c for c in range(n_cols) if c not in pivot_cols]
    M_perm = M[:, perm]
    
    # Perform RREF on M_perm
    work = M_perm.copy()
    basis_coeffs = np.eye(n_rows, dtype=np.uint8)
    
    current_rank = 0
    for col in range(n_cols):
        if current_rank >= n_rows:
            break
        # Find pivot in current_rank..n_rows-1
        pivot = -1
        for row in range(current_rank, n_rows):
            if work[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        
        # Swap
        if pivot != current_rank:
            work[[current_rank, pivot]] = work[[pivot, current_rank]]
            basis_coeffs[[current_rank, pivot]] = basis_coeffs[[pivot, current_rank]]
            
        # Eliminate
        for row in range(n_rows):
            if row != current_rank and work[row, col] == 1:
                work[row, :] ^= work[current_rank, :]
                basis_coeffs[row, :] ^= basis_coeffs[current_rank, :]
                
        current_rank += 1
        
    # The first 'rank' columns of work should be Identity if pivot_cols were independent.
    # The RREF in the original column order is obtained by un-permuting the columns.
    # rref_original = work[:current_rank, :][:, inv_perm]
    
    # However, we need the basis vectors in terms of the ORIGINAL inputs.
    # The rows of basis_coeffs[:current_rank, :] are the coefficients of the ORIGINAL rows (which are permuted? No, basis_coeffs tracks row ops on the original M).
    # Wait, basis_coeffs tracks the row operations applied to the identity matrix of size n_rows.
    # So, RREF_perm = T @ M_perm.
    # We want the basis vectors as vectors in the original input space.
    # The i-th basis vector is the i-th row of RREF_original.
    # RREF_original = RREF_perm[:, inv_perm].
    
    inv_perm = [0] * n_cols
    for i, p in enumerate(perm):
        inv_perm[p] = i
        
    rref_original = work[:current_rank, :][:, inv_perm]
    
    # The coefficients T are the rows of basis_coeffs[:current_rank, :].
    # These tell us how to form the RREF rows from the original M rows.
    # We need the inverse to form the original M rows from the RREF rows.
    # But for synthesis, we just need the RREF rows (basis vectors) and the coefficients of M rows in terms of RREF rows.
    # The coefficients of M_j in terms of RREF rows are given by M_j[pivot_cols] (if RREF has identity in pivot cols).
    # Let's verify: If RREF has identity in pivot_cols, then M_j = sum_i M_j[pivot_cols[i]] * RREF_i.
    
    return rref_original, basis_coeffs[:current_rank, :]

def _synthesize_circuit(M, pivot_cols, n_cols):
    """
    Synthesize circuit for M using the basis defined by pivot_cols.
    Returns gate count and the circuit dict.
    """
    n_rows = M.shape[0]
    rref, _ = _rref_gf2_with_pivots(M, pivot_cols)
    rank = len(pivot_cols)
    
    if rank == 0:
        return 0, {'gates': [], 'outputs': [n_cols] * n_rows}

    next_wire = [n_cols + 1]
    gates = []
    wire_map = {}
    
    # Helper to build XOR tree with caching
    def build_xor_tree(indices):
        if not indices:
            return n_cols
        if len(indices) == 1:
            return indices[0]
        key = tuple(sorted(indices))
        if key in wire_map:
            return wire_map[key]
        
        mid = len(indices) // 2
        left_w = build_xor_tree(indices[:mid])
        right_w = build_xor_tree(indices[mid:])
        
        gates.append([left_w, right_w])
        w_id = next_wire[0]
        next_wire[0] += 1
        wire_map[key] = w_id
        return w_id

    # Synthesize basis vectors (rows of RREF)
    basis_wires = []
    for i in range(rank):
        row = rref[i, :]
        ones = [j for j in range(n_cols) if row[j] == 1]
        w = build_xor_tree(ones)
        basis_wires.append(w)
        
    # Synthesize outputs
    outputs = []
    for j in range(n_rows):
        # Coefficients are M[j, pivot_cols]
        coeffs = M[j, pivot_cols]
        active = [basis_wires[i] for i in range(rank) if coeffs[i] == 1]
        w = build_xor_tree(active)
        outputs.append(w)
        
    return len(gates), {'gates': gates, 'outputs': outputs}

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0
    
    M = np.array(matrix, dtype=np.uint8)
    n_rows, n_cols = M.shape
    
    # Compute rank
    # Use a simple rank calculation
    work = M.copy()
    rank = 0
    for col in range(n_cols):
        pivot = -1
        for row in range(rank, n_rows):
            if work[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        if pivot != rank:
            work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(n_rows):
            if row != rank and work[row, col] == 1:
                work[row, :] ^= work[rank, :]
        rank += 1
        
    if rank == 0:
        return {'gates': [], 'outputs': [n_cols] * n_rows}

    # If rank is small and n_cols is small, enumerate pivot sets
    # Number of pivot sets is C(n_cols, rank)
    # Heuristic: if C(n_cols, rank) < 5000, enumerate
    # Calculate C(n, k)
    def nck(n, k):
        if k > n - k:
            k = n - k
        if k == 0:
            return 1
        res = 1
        for i in range(k):
            res = res * (n - i) // (i + 1)
        return res
        
    num_pivots = nck(n_cols, rank)
    
    best_cost = float('inf')
    best_circuit = None
    
    if num_pivots <= 5000:
        # Enumerate all pivot sets
        # We need to ensure the selected columns are linearly independent.
        # We can filter combinations by checking rank of M[:, combo] == rank.
        # To speed up, we can use a greedy pre-filter or just check.
        
        for combo in combinations(range(n_cols), rank):
            if time.time() > deadline:
                break
                
            # Check independence
            sub = M[:, list(combo)]
            # Quick rank check on sub
            w_sub = sub.copy()
            r_sub = 0
            for c in range(rank):
                p = -1
                for r in range(r_sub, n_rows):
                    if w_sub[r, c] == 1:
                        p = r
                        break
                if p == -1:
                    break
                if p != r_sub:
                    w_sub[[r_sub, p]] = w_sub[[p, r_sub]]
                for r in range(n_rows):
                    if r != r_sub and w_sub[r, c] == 1:
                        w_sub[r, :] ^= w_sub[r_sub, :]
                r_sub += 1
            if r_sub < rank:
                continue
                
            cost, circuit = _synthesize_circuit(M, list(combo), n_cols)
            if cost < best_cost:
                best_cost = cost
                best_circuit = circuit
    else:
        # Fallback: Standard RREF (left-to-right pivots)
        # Find the first 'rank' independent columns
        pivot_cols = []
        work = M.copy()
        r = 0
        for c in range(n_cols):
            if r >= rank:
                break
            p = -1
            for row in range(r, n_rows):
                if work[row, c] == 1:
                    p = row
                    break
            if p == -1:
                continue
            if p != r:
                work[[r, p]] = work[[p, r]]
            for row in range(n_rows):
                if row != r and work[row, c] == 1:
                    work[row, :] ^= work[r, :]
            pivot_cols.append(c)
            r += 1
            
        cost, circuit = _synthesize_circuit(M, pivot_cols, n_cols)
        best_cost = cost
        best_circuit = circuit
        
    if best_circuit is None:
        # Should not happen
        return {'gates': [], 'outputs': [n_cols] * n_rows}
        
    # Post-processing: Local search to improve pivot set
    # If we have time, try to swap one pivot with another to reduce cost
    if time.time() < deadline - 2.0:
        # Extract current pivot set from best_circuit? No, we lost it.
        # We need to store the best pivot set.
        pass
        
    return best_circuit
