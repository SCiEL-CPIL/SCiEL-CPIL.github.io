"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    Returns {'gates': [[wire, wire], ...], 'outputs': [wire, ...]}
    
    Strategy:
    1. Convert rows to bitmasks for fast XOR operations.
    2. Perform Gaussian elimination over GF(2) with minimum-weight pivot selection.
    3. Express each output row as a linear combination of basis rows.
    4. Synthesize gates for basis vectors using a greedy sub-expression sharing approach.
    5. Synthesize gates for output combinations, sharing identical basis-representation masks.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit

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

    # Gaussian Elimination to find basis and representations
    # We want a basis with minimum weight vectors to reduce gate count for basis synthesis.
    
    A_work = [[row_masks[i], 1 << i] for i in range(n_rows)]
    
    r = 0
    pivot_cols = []
    
    for col in range(n_cols):
        # Find pivot with minimum weight among rows that have 1 in this column
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
        
        # Swap
        A_work[r], A_work[pivot_idx] = A_work[pivot_idx], A_work[r]
        
        # Eliminate all other rows
        for i in range(n_rows):
            if i != r and ((A_work[i][0] >> col) & 1):
                A_work[i][0] ^= A_work[r][0]
                A_work[i][1] ^= A_work[r][1]
        
        pivot_cols.append(col)
        r += 1
    
    # r is the rank
    # The first r rows of A_work are the basis vectors (in RREF form)
    basis_vectors = [A_work[i][0] for i in range(r)]
    
    # Map pivot col to basis index
    pivot_to_basis_idx = {col: i for i, col in enumerate(pivot_cols)}
    
    # For each original row, find its basis representation
    # A row M can be written as XOR of basis vectors B_i.
    # Since basis is in RREF, the pivot column of B_i is unique.
    # If M has a 1 in pivot_col_i, then B_i is in the combination.
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
        # temp_M should be 0
        row_basis_reprs.append(repr_mask)

    # Synthesize gates
    gates = []
    # wire_to_mask maps wire_id to the input mask it represents
    # Wire 0..n_cols-1 are inputs. Wire n_cols is 0.
    # Wire n_cols+1+k is the k-th gate output.
    
    wire_to_mask = {}
    for i in range(n_cols):
        wire_to_mask[i] = 1 << i
    wire_to_mask[n_cols] = 0
    
    # Helper to find a wire with a specific mask, or create a gate if not found
    # This is used to share sub-expressions.
    def get_or_create_wire(mask):
        if mask in wire_to_mask:
            return wire_to_mask[mask]
        
        # Check if we can create this mask from two existing wires
        # This is a simple heuristic: try to split mask into two parts that exist.
        # For small n_cols, we can just check all pairs? No, that's O(N^2) per gate.
        # Given n_cols <= 16, and number of gates likely < 100, this might be okay?
        # But we can be smarter.
        
        # If mask is a single bit, it's an input.
        if mask & (mask - 1) == 0:
            # Find which input
            idx = int(np.log2(mask))
            return idx
            
        # If mask is 0, return zero wire
        if mask == 0:
            return n_cols
            
        # Try to find two existing wires whose XOR is mask
        # We can iterate through existing wires and check if mask ^ w1 is also in wire_to_mask
        # This is O(N_wires).
        for w1, m1 in list(wire_to_mask.items()):
            m2 = mask ^ m1
            if m2 in wire_to_mask:
                w2 = wire_to_mask[m2]
                # Create gate
                gate_id = n_cols + 1 + len(gates)
                gates.append([w1, w2])
                wire_to_mask[gate_id] = mask
                return gate_id
        
        # If no pair found, we need to build it from inputs.
        # This case is rare if we start with inputs and build up.
        # Fallback: build tree from inputs.
        # Find set bits
        bits = []
        for j in range(n_cols):
            if (mask >> j) & 1:
                bits.append(j)
                
        if not bits:
            return n_cols
        if len(bits) == 1:
            return bits[0]
            
        # Build tree
        current = bits[0]
        for b in bits[1:]:
            target = wire_to_mask[current] ^ (1 << b)
            # Check if target exists
            if target in wire_to_mask:
                current = wire_to_mask[target]
            else:
                gate_id = n_cols + 1 + len(gates)
                gates.append([current, b])
                wire_to_mask[gate_id] = target
                current = gate_id
        return current

    # 1. Compute basis wires
    basis_wire_ids = []
    for b_idx in range(r):
        bv_mask = basis_vectors[b_idx]
        w = get_or_create_wire(bv_mask)
        basis_wire_ids.append(w)

    # 2. Compute output wires
    outputs = []
    # Map basis-repr mask to wire id to share combinations
    repr_to_wire = {0: n_cols}
    
    for i in range(n_rows):
        repr_mask = row_basis_reprs[i]
        
        if repr_mask in repr_to_wire:
            outputs.append(repr_to_wire[repr_mask])
            continue
        
        # Need to compute this combination of basis wires
        # Find active basis indices
        active_basis = []
        for b_idx in range(r):
            if (repr_mask >> b_idx) & 1:
                active_basis.append(b_idx)
        
        if len(active_basis) == 0:
            outputs.append(n_cols)
            repr_to_wire[repr_mask] = n_cols
            continue
            
        if len(active_basis) == 1:
            w = basis_wire_ids[active_basis[0]]
            outputs.append(w)
            repr_to_wire[repr_mask] = w
            continue
            
        # Multiple basis vectors
        # We need XOR of basis_wire_ids[active_basis]
        # We can use get_or_create_wire on the resulting input mask?
        # The resulting input mask is XOR of basis_vectors[active_basis]
        # Which is just row_masks[i]!
        # So we can just call get_or_create_wire(row_masks[i])
        # This will find if the combination was already created as a sub-expression.
        
        # However, get_or_create_wire might create a gate that is "deeper" or different structure.
        # But since it shares any existing wire that produces the same mask, it is optimal for sharing.
        
        # Let's verify:
        # If we have computed basis wires, and now we need XOR of them.
        # The mask is row_masks[i].
        # If a previous output or basis vector had this mask, we reuse it.
        # If not, get_or_create_wire will try to find two existing wires that XOR to this mask.
        # Since the basis wires are existing wires, and their XOR is this mask,
        # it might find them (if it iterates to one of them).
        
        w = get_or_create_wire(row_masks[i])
        outputs.append(w)
        repr_to_wire[repr_mask] = w

    result = {'gates': gates, 'outputs': outputs}
    
    # Verify
    if not _verify(A, result):
        # Fallback to simple direct synthesis
        gates = []
        outputs = []
        for i in range(n_rows):
            input_indices = [j for j in range(n_cols) if A[i, j] == 1]
            if len(input_indices) == 0:
                outputs.append(n_cols)
            elif len(input_indices) == 1:
                outputs.append(input_indices[0])
            else:
                wires = input_indices
                current_wire = wires[0]
                for w in wires[1:]:
                    gate_id = n_cols + 1 + len(gates)
                    gates.append([current_wire, w])
                    current_wire = gate_id
                outputs.append(current_wire)
        result = {'gates': gates, 'outputs': outputs}
        
        # Final check
        if not _verify(A, result):
             # Should not happen, but raise error if it does
             raise ValueError("Synthesis failed verification")

    return result


def _verify(A, result):
    """Verify that the circuit computes the correct outputs."""
    n_cols = A.shape[1]
    gates = result['gates']
    outputs = result['outputs']
    
    # Use bitmasks for verification
    if n_cols > 64:
        return True
    
    wire_values = [1 << i for i in range(n_cols)]
    wire_values.append(0)  # zero wire
    
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
