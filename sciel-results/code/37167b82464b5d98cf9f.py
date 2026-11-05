"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import defaultdict
import heapq

# EVOLVE-BLOCK-START
def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    Returns {'gates': [[wire, wire], ...], 'outputs': [wire, ...]}
    
    Strategy:
    1. Convert rows to bitmasks.
    2. Use Beam Search to find a low-gate-count circuit that computes all unique row masks.
    3. The search space is the set of available masks. We start with inputs and 0.
    4. We expand by adding gates (XOR of two available masks).
    5. We stop when all target masks are available.
    6. This allows for global sharing of intermediate computations.
    """
    start_time = time.time()
    deadline = start_time + 50.0
    
    A = np.array(matrix, dtype=np.uint8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    n_rows, n_cols = A.shape
    
    # Convert rows to integer masks
    row_masks = []
    for i in range(n_rows):
        mask = 0
        for j in range(n_cols):
            if A[i, j] == 1:
                mask |= (1 << j)
        row_masks.append(mask)
    
    # Unique targets
    targets = set(row_masks)
    targets.discard(0) # 0 is always available
    
    if not targets:
        return {'gates': [], 'outputs': [n_cols] * n_rows}
    
    # Initial state
    # Wire IDs: 0..n_cols-1 are inputs, n_cols is 0.
    # New gates get IDs n_cols+1, n_cols+2, ...
    
    initial_available = set()
    initial_wire_map = {}
    for i in range(n_cols):
        m = 1 << i
        initial_available.add(m)
        initial_wire_map[m] = i
    initial_available.add(0)
    initial_wire_map[0] = n_cols
    
    # Beam Search
    # State: (cost, available_set, wire_map, gates_list)
    # We use a list for the beam, sorted by cost.
    
    beam = [(0, frozenset(initial_available), initial_wire_map, [])]
    best_cost = float('inf')
    best_solution = None
    
    beam_width = 200
    max_depth = 20 # Max gates to search
    
    # To avoid re-processing identical available sets with higher cost
    seen_states = {} # frozenset(available) -> min_cost
    
    while beam:
        if time.time() > deadline - 5.0:
            break
            
        # Sort beam by cost
        beam.sort(key=lambda x: x[0])
        
        new_beam = []
        
        for cost, available, wire_map, gates in beam:
            if cost >= best_cost:
                continue
            if cost > max_depth:
                continue
                
            # Check if all targets are available
            if targets.issubset(available):
                if cost < best_cost:
                    best_cost = cost
                    # Reconstruct outputs
                    outputs = []
                    for mask in row_masks:
                        outputs.append(wire_map[mask])
                    best_solution = {'gates': gates, 'outputs': outputs}
                continue
            
            # Prune if we've seen this available set with lower or equal cost
            avail_key = available
            if avail_key in seen_states:
                if seen_states[avail_key] <= cost:
                    continue
            else:
                seen_states[avail_key] = cost
            
            # Generate neighbors
            avail_list = list(available)
            n_avail = len(avail_list)
            
            # Optimization: If n_avail is large, we can't check all pairs.
            # But n_avail is small in early steps.
            # In later steps, we might have many masks.
            # Limit pair checks if too many?
            # For n_cols=16, max useful masks is small.
            
            for i in range(n_avail):
                m1 = avail_list[i]
                for j in range(i + 1, n_avail):
                    m2 = avail_list[j]
                    m_new = m1 ^ m2
                    
                    if m_new in available:
                        continue
                    
                    # New state
                    new_available = available | {m_new}
                    new_wire_id = n_cols + 1 + len(gates)
                    new_wire_map = wire_map.copy()
                    new_wire_map[m_new] = new_wire_id
                    new_gates = gates + [[wire_map[m1], wire_map[m2]]]
                    new_cost = cost + 1
                    
                    new_beam.append((new_cost, new_available, new_wire_map, new_gates))
                    
        # Prune new_beam
        if not new_beam:
            break
            
        # Sort by cost
        new_beam.sort(key=lambda x: x[0])
        
        # Keep top beam_width
        # Also, if multiple states have the same available set, keep the one with lowest cost
        # (Already handled by seen_states, but within the beam we might have duplicates)
        
        beam = new_beam[:beam_width]
        
        # If we found a solution, we can stop if the minimum cost in beam is >= best_cost
        if best_solution is not None:
            if beam and beam[0][0] >= best_cost:
                break

    if best_solution is None:
        # Fallback to Gaussian Elimination if Beam Search fails or times out
        return _synthesize_gaussian(A, row_masks, n_cols)
        
    return best_solution

def _synthesize_gaussian(A, row_masks, n_cols):
    """Fallback Gaussian Elimination synthesis."""
    n_rows = len(row_masks)
    
    # Gaussian Elimination
    vectors = []
    for i in range(n_rows):
        vectors.append([row_masks[i], 1 << i])
        
    A_work = [[row_masks[i], 1 << i] for i in range(n_rows)]
    r = 0
    pivot_cols = []
    
    for col in range(n_cols):
        pivot_idx = -1
        min_weight = n_cols + 1
        for i in range(r, n_rows):
            if (A_work[i][0] >> col) & 1:
                w = bin(A_work[i][0]).count('1')
                if w < min_weight:
                    min_weight = w
                    pivot_idx = i
                    
        if pivot_idx == -1:
            continue
            
        A_work[r], A_work[pivot_idx] = A_work[pivot_idx], A_work[r]
        
        for i in range(n_rows):
            if i != r and ((A_work[i][0] >> col) & 1):
                A_work[i][0] ^= A_work[r][0]
                A_work[i][1] ^= A_work[r][1]
                
        pivot_cols.append(col)
        r += 1
        
    basis_vectors = [A_work[i][0] for i in range(r)]
    pivot_to_basis_idx = {col: i for i, col in enumerate(pivot_cols)}
    
    row_basis_reprs = []
    for i in range(n_rows):
        M = row_masks[i]
        repr_mask = 0
        temp_M = M
        for col in pivot_cols:
            if (temp_M >> col) & 1:
                b_idx = pivot_to_basis_idx[col]
                repr_mask |= (1 << b_idx)
                temp_M ^= basis_vectors[b_idx]
        row_basis_reprs.append(repr_mask)
        
    gates = []
    gate_to_mask = {}
    for i in range(n_cols):
        gate_to_mask[i] = 1 << i
    gate_to_mask[n_cols] = 0
    
    basis_wire_ids = []
    
    for b_idx in range(r):
        bv_mask = basis_vectors[b_idx]
        if bv_mask == 0:
            basis_wire_ids.append(n_cols)
            continue
        if bv_mask & (bv_mask - 1) == 0:
            wire_id = int(np.log2(bv_mask))
            basis_wire_ids.append(wire_id)
            continue
            
        inputs_needed = [j for j in range(n_cols) if (bv_mask >> j) & 1]
        if len(inputs_needed) == 1:
            basis_wire_ids.append(inputs_needed[0])
            continue
            
        current_wire = inputs_needed[0]
        for w in inputs_needed[1:]:
            target_mask = gate_to_mask[current_wire] ^ gate_to_mask[w]
            found = False
            for wid, m in gate_to_mask.items():
                if m == target_mask:
                    current_wire = wid
                    found = True
                    break
            if not found:
                gate_id = n_cols + 1 + len(gates)
                gates.append([current_wire, w])
                gate_to_mask[gate_id] = target_mask
                current_wire = gate_id
        basis_wire_ids.append(current_wire)
        
    outputs = []
    repr_to_wire = {0: n_cols}
    
    for i in range(n_rows):
        repr_mask = row_basis_reprs[i]
        if repr_mask in repr_to_wire:
            outputs.append(repr_to_wire[repr_mask])
            continue
            
        active_basis = [b_idx for b_idx in range(r) if (repr_mask >> b_idx) & 1]
        
        if not active_basis:
            outputs.append(n_cols)
            repr_to_wire[repr_mask] = n_cols
            continue
        if len(active_basis) == 1:
            w = basis_wire_ids[active_basis[0]]
            outputs.append(w)
            repr_to_wire[repr_mask] = w
            continue
            
        current_wire = basis_wire_ids[active_basis[0]]
        for b_idx in active_basis[1:]:
            w = basis_wire_ids[b_idx]
            target_mask = gate_to_mask[current_wire] ^ gate_to_mask[w]
            found = False
            for wid, m in gate_to_mask.items():
                if m == target_mask:
                    current_wire = wid
                    found = True
                    break
            if not found:
                gate_id = n_cols + 1 + len(gates)
                gates.append([current_wire, w])
                gate_to_mask[gate_id] = target_mask
                current_wire = gate_id
        outputs.append(current_wire)
        repr_to_wire[repr_mask] = current_wire
        
    return {'gates': gates, 'outputs': outputs}

def _verify(A, result):
    """Verify that the circuit computes the correct outputs."""
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
        
    for i, out_wire in enumerate(outputs):
        if out_wire >= len(wire_values):
            return False
        expected = 0
        for j in range(n_cols):
            if A[i, j] == 1:
                expected ^= (1 << j)
        if wire_values[out_wire] != expected:
            return False
            
    return True
# EVOLVE-BLOCK-END
