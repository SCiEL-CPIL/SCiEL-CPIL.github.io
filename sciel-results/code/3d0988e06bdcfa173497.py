import time
import random

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 87.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    def elapsed():
        return time.time() - start_time

    best_result = None
    best_score = float('inf')

    max_restarts = 1000 # Increased restarts

    for restart in range(max_restarts):
        if elapsed() > DEADLINE - 3.0:
            break
        
        wire_masks = {}
        mask_to_wire = {}
        wire_depth = {}
        
        for j in range(n):
            wire_id = j
            mask = 1 << j
            wire_masks[wire_id] = mask
            mask_to_wire[mask] = wire_id
            wire_depth[wire_id] = 0
            
        zero_wire = n
        wire_masks[zero_wire] = 0
        mask_to_wire[0] = zero_wire
        wire_depth[zero_wire] = 0
        
        next_wire_id = n + 1
        
        row_bits = [0] * m
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
        
        wire_usage = {}
        for w in wire_masks:
            usage = 0
            for r in range(m):
                if (row_bits[r] >> w) & 1:
                    usage |= (1 << r)
            wire_usage[w] = usage

        while elapsed() < DEADLINE - 3.0:
            best_delta = 0
            best_u = -1
            best_v = -1
            best_w_exist = -1
            best_new_mask = -1
            best_new_depth = 0
            
            wire_ids = [w for w in wire_masks.keys() if w != zero_wire]
            k = len(wire_ids)
            
            # Randomize order for tie-breaking diversity
            random.shuffle(wire_ids)

            for i in range(k):
                u = wire_ids[i]
                usage_u = wire_usage[u]
                du = wire_depth[u]
                for j in range(i + 1, k):
                    v = wire_ids[j]
                    usage_v = wire_usage[v]
                    
                    common_rows = usage_u & usage_v
                    if common_rows == 0:
                        continue
                    
                    mask_u = wire_masks[u]
                    mask_v = wire_masks[v]
                    new_mask = mask_u ^ mask_v
                    
                    if new_mask == 0:
                        continue

                    if new_mask in mask_to_wire:
                        w_exist = mask_to_wire[new_mask]
                        if w_exist == u or w_exist == v:
                            continue
                    else:
                        w_exist = -1
                    
                    delta_gates = 0 if w_exist != -1 else 1
                    
                    if w_exist != -1:
                        w = w_exist
                        usage_w = wire_usage[w]
                        rows_with_uvw = common_rows & usage_w
                        rows_with_uv_not_w = common_rows & ~usage_w
                        delta_wires = -3 * (rows_with_uvw.bit_count()) - 1 * (rows_with_uv_not_w.bit_count())
                        new_depth = wire_depth[w]
                    else:
                        delta_wires = -1 * (common_rows.bit_count())
                        dv = wire_depth[v]
                        new_depth = max(du, dv) + 1
                        if new_depth > depth_limit:
                            continue

                    # Depth penalty
                    current_max_depth = max(wire_depth.values()) if wire_depth else 0
                    depth_penalty = 0
                    if new_depth > current_max_depth:
                        depth_penalty = (new_depth - current_max_depth) * 0.1 # Small penalty

                    delta_cost = delta_gates + delta_wires + depth_penalty
                    
                    if delta_cost < best_delta:
                        best_delta = delta_cost
                        best_u = u
                        best_v = v
                        best_w_exist = w_exist
                        best_new_mask = new_mask
                        best_new_depth = new_depth

            if best_delta >= 0:
                break
            
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
                wire_depth[w] = best_new_depth
            
            for r in range(m):
                if (row_bits[r] >> u) & 1 and (row_bits[r] >> v) & 1:
                    row_bits[r] &= ~(1 << u)
                    row_bits[r] &= ~(1 << v)
                    
                    if best_w_exist != -1:
                        if (row_bits[r] >> w) & 1:
                            row_bits[r] &= ~(1 << w)
                            wire_usage[w] &= ~(1 << r)
                        else:
                            row_bits[r] |= (1 << w)
                            wire_usage[w] |= (1 << r)
                    else:
                        row_bits[r] |= (1 << w)
            
            wire_usage[u] = 0
            wire_usage[v] = 0
            if best_w_exist == -1:
                wire_usage[w] = 0
                for r in range(m):
                    if (row_bits[r] >> w) & 1:
                        wire_usage[w] |= (1 << r)

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
                                # Check depth here too?
                                # wire_depth[new_w] = max(wire_depth[w1], wire_depth[w2]) + 1
                                new_current_wires.append(new_w)
                        else:
                            new_current_wires.append(current_wires[k])
                    current_wires = new_current_wires
                outputs[i] = current_wires[0]

        final_gates = len(gates)
        if best_score > final_gates:
            best_score = final_gates
            best_result = {'gates': gates, 'outputs': outputs}

    if best_result is None:
        # Fallback
        gates = []
        outputs = []
        for i in range(m):
            bits = row_masks[i]
            if bits == 0:
                outputs.append(n)
            elif bits & (bits - 1) == 0:
                outputs.append(bits.bit_length() - 1)
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
                            new_w = n + 1 + len(gates)
                            gates.append([w1, w2])
                            new_current_wires.append(new_w)
                        else:
                            new_current_wires.append(current_wires[k])
                    current_wires = new_current_wires
                outputs.append(current_wires[0])
        best_result = {'gates': gates, 'outputs': outputs}

    return best_result
