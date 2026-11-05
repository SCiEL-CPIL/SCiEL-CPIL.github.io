# Corrected Final Construction
    outputs = []
    for i in range(n_rows):
        wires = row_wires[i]
        if len(wires) == 0:
            outputs.append(zero_wire)
        elif len(wires) == 1:
            outputs.append(wires[0])
        else:
            # Sort wires? No, order doesn't matter for XOR.
            # We want to minimize depth and gates.
            # Greedy pairing: always XOR the two "simplest" wires?
            # Or just pair them sequentially.
            
            # To ensure depth <= 16, we should use a balanced tree if len(wires) is large.
            # But len(wires) is likely small after optimization.
            
            # Let's use a simple list-based reduction.
            remaining = list(wires)
            while len(remaining) > 1:
                if time.time() > deadline:
                    break
                # Pick two wires to XOR
                # Heuristic: Pick two wires that result in an existing wire if possible?
                # Or just pick first two.
                w1 = remaining.pop(0)
                w2 = remaining.pop(0)
                
                new_m = wire_mask[w1] ^ wire_mask[w2]
                if new_m in mask_to_wire:
                    new_w = mask_to_wire[new_m]
                else:
                    new_w = next_wire
                    next_wire += 1
                    wire_mask.append(new_m)
                    mask_to_wire[new_m] = new_w
                    wire_depth.append(max(wire_depth[w1], wire_depth[w2]) + 1)
                    gates.append([w1, w2])
                
                remaining.append(new_w)
            
            outputs.append(remaining[0])
