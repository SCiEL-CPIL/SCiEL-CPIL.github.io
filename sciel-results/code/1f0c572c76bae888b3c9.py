"""
Exact XOR-circuit synthesis using a different structural strategy:
Instead of searching over pivot sets, we use a canonical RREF basis and
optimize the synthesis by minimizing the total Hamming weight of the
basis vectors and output coefficients through a row-space basis change.

Key insight: The cost of the circuit is dominated by the sum of Hamming weights
of the basis vectors (RREF rows) and the output coefficient vectors. By
performing a change of basis within the row space (using the transformation
matrix from RREF), we can find a basis that minimizes the total synthesis cost.

We use a greedy local search over the row space basis: start with RREF, then
try to reduce the total weight by adding rows to other rows (elementary
row operations) and keep the change if it reduces cost.
"""
import time
import numpy as np

def _rref_gf2(M):
    """
    Compute RREF of M over GF(2).
    Returns:
      rref: reduced matrix (rank x n_cols)
      T: transformation matrix (rank x n_rows) such that rref = T @ M
      pivot_cols: list of pivot column indices
    """
    M = np.array(M, dtype=np.uint8)
    n_rows, n_cols = M.shape
    work = M.copy()
    T = np.eye(n_rows, dtype=np.uint8)
    rank = 0
    pivot_cols = []
    
    for col in range(n_cols):
        if rank >= n_rows:
            break
        # Find pivot
        pivot = -1
        for row in range(rank, n_rows):
            if work[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        if pivot != rank:
            work[[rank, pivot]] = work[[pivot, rank]]
            T[[rank, pivot]] = T[[pivot, rank]]
        for row in range(n_rows):
            if row != rank and work[row, col] == 1:
                work[row, :] ^= work[rank, :]
                T[row, :] ^= T[rank, :]
        rank += 1
        pivot_cols.append(col)
    
    return work[:rank, :], T[:rank, :], pivot_cols


def _synthesize_with_basis(basis, coeff_matrix, n_cols):
    """
    Synthesize XOR circuit given a basis (list of vectors) and coefficient matrix.
    Each output is a linear combination of basis vectors given by coeff_matrix.
    Returns gate count and circuit dict.
    """
    n_outputs = coeff_matrix.shape[0]
    rank = len(basis)
    
    if rank == 0:
        return 0, {'gates': [], 'outputs': [n_cols] * n_outputs}
    
    next_wire = [n_cols + 1]
    gates = []
    wire_cache = {}
    
    def build_xor(indices):
        """Build XOR of wires in indices, with caching."""
        if not indices:
            return n_cols
        if len(indices) == 1:
            return indices[0]
        key = tuple(sorted(indices))
        if key in wire_cache:
            return wire_cache[key]
        mid = len(indices) // 2
        left = build_xor(indices[:mid])
        right = build_xor(indices[mid:])
        gates.append([left, right])
        w = next_wire[0]
        next_wire[0] += 1
        wire_cache[key] = w
        return w
    
    # Synthesize each basis vector
    basis_wires = []
    for i in range(rank):
        vec = basis[i]
        ones = [j for j in range(n_cols) if vec[j] == 1]
        w = build_xor(ones)
        basis_wires.append(w)
    
    # Synthesize outputs
    outputs = []
    for j in range(n_outputs):
        coeffs = coeff_matrix[j]
        active = [basis_wires[i] for i in range(rank) if coeffs[i] == 1]
        w = build_xor(active)
        outputs.append(w)
    
    return len(gates), {'gates': gates, 'outputs': outputs}


def _total_weight(basis, coeff_matrix):
    """Compute total Hamming weight of basis vectors and coefficient vectors."""
    w = 0
    for i in range(len(basis)):
        w += np.sum(basis[i])
    w += np.sum(coeff_matrix)
    return w


def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 80.0
    
    M = np.array(matrix, dtype=np.uint8)
    n_rows, n_cols = M.shape
    
    # Compute RREF
    rref, T, pivot_cols = _rref_gf2(M)
    rank = len(pivot_cols)
    
    if rank == 0:
        return {'gates': [], 'outputs': [n_cols] * n_rows}
    
    # Initial basis: rows of RREF
    basis = [rref[i].copy() for i in range(rank)]
    
    # Coefficient matrix: how each original row is expressed in terms of RREF rows
    # Since RREF has identity in pivot_cols, the coefficients are M[:, pivot_cols]
    coeff = M[:, pivot_cols].copy()
    
    # Greedy local search: try elementary row operations to reduce total weight
    # An elementary operation: basis[i] ^= basis[j] for i != j
    # This changes the basis but we need to update coeff accordingly.
    # If we do basis[i] <- basis[i] + basis[j], then:
    #   new_coeff[:, i] = old_coeff[:, i] + old_coeff[:, j]
    #   new_coeff[:, j] = old_coeff[:, j]
    
    improved = True
    max_iterations = 500
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        if time.time() > deadline:
            break
        
        best_delta = 0
        best_op = None
        
        current_weight = _total_weight(basis, coeff)
        
        for i in range(rank):
            if time.time() > deadline:
                break
            for j in range(rank):
                if i == j:
                    continue
                # Try basis[i] ^= basis[j]
                new_basis_i = basis[i] ^ basis[j]
                new_coeff_i = coeff[:, i] ^ coeff[:, j]
                
                # Compute new total weight
                # Weight change: 
                #   basis[i] changes from weight(basis[i]) to weight(new_basis_i)
                #   coeff[:,i] changes from weight(coeff[:,i]) to weight(new_coeff_i)
                old_w = np.sum(basis[i]) + np.sum(coeff[:, i])
                new_w = np.sum(new_basis_i) + np.sum(new_coeff_i)
                delta = new_w - old_w
                
                if delta < best_delta:
                    best_delta = delta
                    best_op = (i, j)
        
        if best_op is not None and best_delta < 0:
            i, j = best_op
            basis[i] = basis[i] ^ basis[j]
            coeff[:, i] = coeff[:, i] ^ coeff[:, j]
            improved = True
    
    # Final synthesis
    gate_count, circuit = _synthesize_with_basis(basis, coeff, n_cols)
    
    return circuit
