import time
import random

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 85.0  # 90s limit, leave 5s margin

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Parse rows as bitmasks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    def elapsed():
        return time.time() - start_time

    def popcount(x):
        return bin(x).count('1')

    best_result = None
    best_score = float('inf')

    # Run multiple restarts
    num_restarts = 100 # Adjust based on speed
    
    for restart in range(num_restarts):
        if elapsed() > DEADLINE - 5.0:
            break
        
        # State
        wire_masks = {}
        mask_to_wire = {}
        
        # Initialize inputs
        for j in range(n):
            wire_id = j
            mask = 1 << j
            wire_masks[wire_id] = mask
            mask_to_wire[mask] = wire_id
            
        # Initialize zero
        zero_wire = n
        wire_masks[zero_wire] = 0
        mask_to_wire[0] = zero_wire
        
        next_wire_id = n + 1
        
        # row_bits[r] is an integer bitmask where bit w is set if wire w is in row r
        row_bits = [0] * m
        
        # Initialize row_bits based on input wires
        for i in range(m):
            mask = row_masks[i]
            if mask == 0:
                row_bits[i] = (1 << zero_wire)
            else:
                bits = 0
                for j in range(n):
                    if (mask >> j) & 1:
                        bits |= (1 << j)
                row_bits[i] = bits

        gates = []
        
        # Calculate initial cost
        def get_cost():
            num_gates = next_wire_id - n - 1
            total_wires = 0
            for r in range(m):
                total_wires += popcount(row_bits[r])
            return num_gates + total_wires

        current_cost = get_cost()
        
        # Greedy Loop
        while elapsed() < DEADLINE - 5.0:
            best_delta = 0
            best_u = -1
            best_v = -1
            best_w_exist = -1
            best_new_mask = -1
            
            wire_ids = [w for w in wire_masks.keys() if w != zero_wire]
            k = len(wire_ids)
            
            # Optimization: If k is large, we might need to be careful.
            # But k is small (<= n + gates).
            for i in range(k):
                u = wire_ids[i]
                for j in range(i + 1, k):
                    v = wire_ids[j]
                    
                    mask_u = wire_masks[u]
                    mask_v = wire_masks[v]
                    new_mask = mask_u ^ mask_v
                    
                    if new_mask == 0:
                        # u and v are the same mask? Should not happen if distinct wires
                        # unless we have duplicates. We assume distinct masks for distinct wires.
                        continue

                    if new_mask in mask_to_wire:
                        w_exist = mask_to_wire[new_mask]
                        if w_exist == u or w_exist == v:
                            continue
                    else:
                        w_exist = -1
                    
                    # Calculate delta cost
                    delta_gates = 0 if w_exist != -1 else 1
                    delta_wires = 0
                    
                    for r in range(m):
                        u_in = (row_bits[r] >> u) & 1
                        v_in = (row_bits[r] >> v) & 1
                        if u_in and v_in:
                            # u and v are in this row
                            if w_exist != -1:
                                w_in = (row_bits[r] >> w_exist) & 1
                                if w_in:
                                    # Remove u, v, w. Add nothing.
                                    # Old: u, v, w (3 wires). New: nothing (0 wires).
                                    # Delta: -3
                                    delta_wires -= 3
                                else:
                                    # Remove u, v. Add w.
                                    # Old: u, v (2 wires). New: w (1 wire).
                                    # Delta: -1
                                    delta_wires -= 1
                            else:
                                # Remove u, v. Add new w.
                                # Old: u, v (2 wires). New: w (1 wire).
                                # Delta: -1
                                delta_wires -= 1
                    
                    delta_cost = delta_gates + delta_wires
                    
                    if delta_cost < best_delta:
                        best_delta = delta_cost
                        best_u = u
                        best_v = v
                        best_w_exist = w_exist
                        best_new_mask = new_mask

            if best_delta >= 0:
                break
            
            # Apply best move
            u = best_u
            v = best_v
            new_mask = best_new_mask
            
            if best_w_exist != -1:
                w = best_w_exist
            else:
                w = next_wire_id
                next_wire_id += 1
                gates.append([u, v])
                wire_masks[w] = new_mask
                mask_to_wire[new_mask] = w
            
            # Update row_bits
            for r in range(m):
                if (row_bits[r] >> u) & 1 and (row_bits[r] >> v) & 1:
                    row_bits[r] &= ~(1 << u)
                    row_bits[r] &= ~(1 << v)
                    
                    if best_w_exist != -1:
                        if (row_bits[r] >> w) & 1:
                            row_bits[r] &= ~(1 << w)
                        else:
                            row_bits[r] |= (1 << w)
                    else:
                        row_bits[r] |= (1 << w)
                        
            current_cost += best_delta

        # Final Output Construction
        outputs = [0] * m
        for i in range(m):
            bits = row_bits[i]
            if bits == 0:
                outputs[i] = zero_wire
            elif bits & (bits - 1) == 0:
                outputs[i] = bits.bit_length() - 1
            else:
                wires = []
                temp_bits = bits
                while temp_bits:
                    lsb = temp_bits & -temp_bits
                    w_id = lsb.bit_length() - 1
                    wires.append(w_id)
                    temp_bits ^= lsb
                
                current_wires = wires
                while len(current_wires) > 1:
                    new_current_wires = []
                    for k in range(0, len(current_wires), 2):
                        if k + 1 < len(current_wires):
                            w1 = current_wires[k]
                            w2 = current_wires[k + 1]
                            m1 = wire_masks[w1]
                            m2 = wire_masks[w2]
                            nm = m1 ^ m2
                            if nm in mask_to_wire:
                                new_current_wires.append(mask_to_wire[nm])
                            else:
                                new_w = next_wire_id
                                next_wire_id += 1
                                gates.append([w1, w2])
                                wire_masks[new_w] = nm
                                mask_to_wire[nm] = new_w
                                new_current_wires.append(new_w)
                        else:
                            new_current_wires.append(current_wires[k])
                    current_wires = new_current_wires
                outputs[i] = current_wires[0]

        # Calculate final score for this restart
        final_gates = len(gates)
        # The "value" is likely related to final_gates.
        # We want to minimize final_gates.
        if best_score > final_gates:
            best_score = final_gates
            best_result = {'gates': gates, 'outputs': outputs}
            
            # Optional: Check if we can improve further with a different strategy?
            # For now, just keep the best.

    if best_result is None:
        # Fallback: return trivial circuit
        gates = []
        outputs = []
        for i in range(m):
            # Just output the input wires directly?
            # But we need to combine them if there are multiple.
            # This is the baseline.
            bits = row_masks[i]
            if bits == 0:
                outputs.append(n)
            elif bits & (bits - 1) == 0:
                outputs.append(bits.bit_length() - 1)
            else:
                # Need to build a tree
                wires = []
                temp_bits = bits
                while temp_bits:
                    lsb = temp_bits & -temp_bits
                    w_id = lsb.bit_length() - 1
                    wires.append(w_id)
                    temp_bits ^= lsb
                
                current_wires = wires
                while len(current_wires) > 1:
                    new_current_wires = []
                    for k in range(0, len(current_wires), 2):
                        if k + 1 < len(current_wires):
                            w1 = current_wires[k]
                            w2 = current_wires[k + 1]
                            new_w = n + 1 + len(gates)
                            gates.append([w1, w2])
                            new_current_wires.append(new_w)
                        else:
                            new_current_wires.append(current_wires[k])
                    current_wires = new_current_wires
                outputs.append(current_wires[0])
        best_result = {'gates': gates, 'outputs': outputs}

    return best_result
