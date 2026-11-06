import time
import numpy as np
from itertools import combinations

def _compute_rref(M):
    """Compute RREF over GF(2) and track row operations."""
    M = np.array(M, dtype=np.uint8)
    n_rows, n_cols = M.shape
    work = M.copy()
    basis = np.eye(n_rows, dtype=np.uint8)
    
    pivot_cols = []
    for col in range(n_cols):
        if not pivot_cols or len(pivot_cols) >= n_rows:
            pass
        # Find pivot in this column
        pivot_row = -1
        for row in range(len(pivot_cols), n_rows):
            if work[row, col] == 1:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        
        # Swap to pivot position
        pivot_pos = len(pivot_cols)
        if pivot_row != pivot_pos:
            work[[pivot_row, pivot_pos]] = work[[pivot_pos, pivot_row]]
            basis[[pivot_row, pivot_pos]] = basis[[pivot_pos, pivot_row]]
        
        # Eliminate all other rows in this column
        for row in range(n_rows):
            if row != pivot_pos and work[row, col] == 1:
                work[row, :] ^= work[pivot_pos, :]
                basis[row, :] ^= basis[pivot_pos, :]
        
        pivot_cols.append(col)
    
    rank = len(pivot_cols)
    rref = work[:rank, :]
    # basis[:rank, :] gives the transformation from original rows to RREF rows
    # Actually, RREF = T @ M where T = basis[:rank, :]
    # We want to express original rows in terms of RREF rows
    # M = T^{-1} @ RREF, so original_row_j = sum_i (T^{-1})[j,i] * RREF_row_i
    
    # For synthesis, we need:
    # 1. The RREF rows (basis vectors)
    # 2. For each original row, which RREF rows to XOR together
    # The coefficients are given by M[j, pivot_cols] since RREF has identity in pivot_cols
    
    return rref, pivot_cols

def _synthesize_with_basis(M, rref, pivot_cols, n_cols, deadline):
    """Synthesize circuit using given RREF basis. Returns (gate_count, circuit)."""
    n_rows = M.shape[0]
    rank = len(pivot_cols)
    
    if rank == 0:
        return 0, {'gates': [], 'outputs': [n_cols] * n_rows}
    
    gates = []
    wire_cache = {}  # frozenset of wire indices -> wire id
    next_wire = [n_cols + 1]
    
    def get_wire(indices):
        """Get or create wire for XOR of given indices."""
        if not indices:
            return n_cols
        if len(indices) == 1:
            return indices[0]
        
        key = frozenset(indices)
        if key in wire_cache:
            return wire_cache[key]
        
        # Build balanced XOR tree
        indices = list(indices)
        if len(indices) == 2:
            w = [indices[0], indices[1]]
        else:
            mid = len(indices) // 2
            left = get_wire(indices[:mid])
            right = get_wire(indices[mid:])
            w = [left, right]
        
        gates.append(w)
        wire_id = next_wire[0]
        next_wire[0] += 1
        wire_cache[key] = wire_id
        return wire_id
    
    # Build basis wires
    basis_wires = []
    for i in range(rank):
        row = rref[i, :]
        ones = [j for j in range(n_cols) if row[j] == 1]
        w = get_wire(ones)
        basis_wires.append(w)
    
    # Build output wires
    outputs = []
    for j in range(n_rows):
        coeffs = M[j, pivot_cols]
        active = [basis_wires[i] for i in range(rank) if coeffs[i] == 1]
        w = get_wire(active)
        outputs.append(w)
    
    return len(gates), {'gates': gates, 'outputs': outputs}

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    M = np.array(matrix, dtype=np.uint8)
    n_rows, n_cols = M.shape
    
    # Compute RREF
    rref, pivot_cols = _compute_rref(M)
    rank = len(pivot_cols)
    
    if rank == 0:
        return {'gates': [], 'outputs': [n_cols] * n_rows}
    
    # Strategy 1: Standard RREF basis
    cost1, circuit1 = _synthesize_with_basis(M, rref, pivot_cols, n_cols, deadline)
    
    best_cost = cost1
    best_circuit = circuit1
    
    # If time allows, try alternative pivot strategies
    if time.time() < deadline - 5.0:
        # Try different column orderings to get different RREF bases
        # For small matrices, we can try a few permutations
        
        if n_cols <= 12:
            # Try reversing column order
            M_rev = M[:, ::-1]
            rref_rev, pivot_cols_rev = _compute_rref(M_rev)
            # Map pivot columns back
            pivot_cols_rev_orig = [n_cols - 1 - c for c in pivot_cols_rev]
            cost2, circuit2 = _synthesize_with_basis(M, rref_rev[:, ::-1], pivot_cols_rev_orig, n_cols, deadline)
            if cost2 < best_cost:
                best_cost = cost2
                best_circuit = circuit2
        
        # Try greedy: pick columns with lowest weight first
        if time.time() < deadline - 3.0:
            col_weights = M.sum(axis=0)
            # Sort columns by weight
            sorted_cols = np.argsort(col_weights)
            
            # Build a new RREF using only these columns in this order
            # We need to find independent columns in this order
            work = M.copy()
            basis = np.eye(n_rows, dtype=np.uint8)
            new_pivot_cols = []
            new_rref_rows = []
            
            for col in sorted_cols:
                if len(new_pivot_cols) >= rank:
                    break
                pivot_row = -1
                for row in range(len(new_pivot_cols), n_rows):
                    if work[row, col] == 1:
                        pivot_row = row
                        break
                if pivot_row == -1:
                    continue
                
                pivot_pos = len(new_pivot_cols)
                if pivot_row != pivot_pos:
                    work[[pivot_row, pivot_pos]] = work[[pivot_pos, pivot_row]]
                    basis[[pivot_row, pivot_pos]] = basis[[pivot_pos, pivot_row]]
                
                for row in range(n_rows):
                    if row != pivot_pos and work[row, col] == 1:
                        work[row, :] ^= work[pivot_pos, :]
                        basis[row, :] ^= basis[pivot_pos, :]
                
                new_pivot_cols.append(col)
                new_rref_rows.append(work[pivot_pos, :])
            
            if len(new_pivot_cols) == rank:
                rref_new = np.array(new_rref_rows, dtype=np.uint8)
                cost3, circuit3 = _synthesize_with_basis(M, rref_new, new_pivot_cols, n_cols, deadline)
                if cost3 < best_cost:
                    best_cost = cost3
                    best_circuit = circuit3
    
    return best_circuit
