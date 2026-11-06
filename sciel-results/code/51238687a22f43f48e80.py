import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    # Safety margin: 90s limit, use 50s
    deadline = start_time + 50.0
    
    if not matrix or not matrix[0]:
        return {'gates': [], 'outputs': []}
        
    m = len(matrix)
    n = len(matrix[0])
    
    # Convert rows to masks
    rows = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j]:
                mask |= (1 << j)
        rows.append(mask)
        
    # Wire IDs:
    # 0 .. n-1 : Inputs
    # n       : Zero
    # n+1 ..  : Gates
    
    # available_wires: list of masks
    # wire_depth: list of depths
    # wire_repr: list of sets of wire IDs that form this wire
    
    available_wires = [1 << j for j in range(n)] + [0]
    wire_depth = [0] * (n + 1)
    # repr[i] is the set of wire IDs that XOR to wire i
    # For inputs, repr[i] = {i}
    # For zero, repr[n] = set()
    wire_repr = [set([i]) for i in range(n)] + [set()]
    
    gates = []
    outputs = []
    
    # Basis for finding representations
    # basis: dict mapping pivot_bit_index -> wire_id
    # This basis is built from available_wires
    basis = {}
    
    # Initialize basis with inputs and zero
    # Zero doesn't help with reduction, but inputs do.
    for i in range(n):
        w = available_wires[i]
        if w == 0: continue
        # Find pivot (highest set bit)
        pivot = w.bit_length() - 1
        # If pivot already exists, we might want to reduce?
        # For initial inputs, they are independent.
        basis[pivot] = i
        
    # Helper to reduce a target mask using the current basis
    # Returns the set of wire IDs used, or None if not possible (should not happen)
    def get_representation(target):
        if target == 0:
            return set()
            
        current = target
        used_wires = set()
        
        # We need to reduce 'current' to 0
        # Iterate through bits from high to low
        for bit in range(n - 1, -1, -1):
            if not (current & (1 << bit)):
                continue
            if bit in basis:
                wire_id = basis[bit]
                w = available_wires[wire_id]
                current ^= w
                used_wires.add(wire_id)
            else:
                # Should not happen if target is in span
                # If it does, we can't represent it with existing wires
                return None
        
        if current != 0:
            return None
        return used_wires

    for r_idx, target in enumerate(rows):
        if time.time() > deadline:
            break
            
        # Check if target is directly available
        found_id = -1
        for i, w in enumerate(available_wires):
            if w == target:
                found_id = i
                break
        
        if found_id != -1:
            outputs.append(found_id)
            continue
            
        # Find representation using basis
        repr_set = get_representation(target)
        
        if repr_set is None or len(repr_set) == 0:
            # If target is 0, output is n (zero wire)
            if target == 0:
                outputs.append(n)
                continue
            # If not in span, something is wrong. 
            # But inputs span the space. So this shouldn't happen.
            # Fallback: use inputs directly
            # Find inputs that match
            inputs_used = []
            t = target
            for j in range(n):
                if t & (1 << j):
                    inputs_used.append(j)
                    t ^= (1 << j)
            if not inputs_used:
                outputs.append(n)
                continue
            if len(inputs_used) == 1:
                outputs.append(inputs_used[0])
                continue
            # Combine inputs
            prev = inputs_used[0]
            for j in range(1, len(inputs_used)):
                new_id = len(available_wires)
                gates.append([prev, inputs_used[j]])
                available_wires.append(available_wires[prev] ^ available_wires[inputs_used[j]])
                wire_depth.append(max(wire_depth[prev], wire_depth[inputs_used[j]]) + 1)
                wire_repr.append(wire_repr[prev] | wire_repr[inputs_used[j]])
                # Update basis with new wire?
                # For now, skip basis update in fallback
                prev = new_id
            outputs.append(prev)
            continue

        # We have a representation using existing wires
        if len(repr_set) == 1:
            # Just use that wire
            outputs.append(list(repr_set)[0])
            continue
            
        # Create new gates to combine these wires
        # To minimize depth, we can pair them up
        # But we also want to keep the new wire in the basis for future reuse
        # Let's just chain them: w1 ^ w2 -> t1, t1 ^ w3 -> t2 ...
        # This might increase depth. 
        # For small N, depth is not a constraint.
        
        wires_to_combine = list(repr_set)
        prev = wires_to_combine[0]
        for j in range(1, len(wires_to_combine)):
            new_id = len(available_wires)
            gates.append([prev, wires_to_combine[j]])
            new_mask = available_wires[prev] ^ available_wires[wires_to_combine[j]]
            available_wires.append(new_mask)
            wire_depth.append(max(wire_depth[prev], wire_depth[wires_to_combine[j]]) + 1)
            wire_repr.append(wire_repr[prev] | wire_repr[wires_to_combine[j]])
            
            # Update basis with the new wire
            if new_mask != 0:
                pivot = new_mask.bit_length() - 1
                # If pivot exists, we could reduce, but for simplicity we just add
                # If we add multiple wires with same pivot, the basis is not reduced.
                # get_representation assumes a reduced basis (unique pivot).
                # So we must reduce the new wire against the existing basis before adding.
                
                # Reduce new_mask using basis
                reduced = new_mask
                # We need to track the representation of the reduced wire
                # This is getting complicated.
                
                # Simple approach: Only add to basis if pivot is free.
                # If pivot is taken, skip adding to basis (or replace if better?).
                # Skipping is safer for correctness of get_representation.
                if pivot not in basis:
                    basis[pivot] = new_id
                else:
                    # Optional: Try to reduce new_mask against basis[pivot]
                    # If it becomes 0, it's a dependency.
                    # If not, it has a different pivot? No, same pivot means they share highest bit.
                    # XORing them will clear that bit.
                    # Let's just not add to basis if occupied to keep basis valid.
                    pass
                    
            prev = new_id
            
        outputs.append(prev)

    return {'gates': gates, 'outputs': outputs}
