def synthesize(matrix, depth_limit):
    import numpy as np
    import time
    
    m, n = matrix.shape
    rows = []
    for i in range(m):
        v = 0
        for j in range(n):
            if matrix[i, j]:
                v |= (1 << j)
        rows.append(v)
    
    wire_to_val = {}
    val_to_wire = {}
    
    for j in range(n):
        v = 1 << j
        wire_to_val[j] = v
        val_to_wire[v] = j
    wire_to_val[n] = 0
    val_to_wire[0] = n
    
    next_wire = n + 1
    gates = []
    
    def xor_wires(w1, w2):
        nonlocal next_wire
        v1 = wire_to_val[w1]
        v2 = wire_to_val[w2]
        v = v1 ^ v2
        if v in val_to_wire:
            return val_to_wire[v]
        gates.append([w1, w2])
        wire_to_val[next_wire] = v
        val_to_wire[v] = next_wire
        next_wire += 1
        return next_wire
    
    def build_from_bits(v):
        """Build wire for v from input bits, using balanced tree"""
        if v in val_to_wire:
            return val_to_wire[v]
        if v == 0:
            return n
        bits = [b for b in range(n) if (v >> b) & 1]
        if not bits:
            return n
        if len(bits) == 1:
            return bits[0]
        # Balanced XOR
        wires = [b for b in bits]
        while len(wires) > 1:
            new_wires = []
            for i in range(0, len(wires), 2):
                if i + 1 < len(wires):
                    new_wires.append(xor_wires(wires[i], wires[i+1]))
                else:
                    new_wires.append(wires[i])
            wires = new_wires
        return wires[0]
    
    def get_wire(v):
        if v in val_to_wire:
            return val_to_wire[v]
        if v == 0:
            return n
        
        best_w1, best_diff = None, None
        best_cost = float('inf')
        
        for w, val in wire_to_val.items():
            if val == 0:
                continue
            diff = v ^ val
            dw = diff.bit_count()
            if dw == 0:
                cost = 1
            elif dw == 1:
                cost = 1
            elif dw == 2:
                cost = 2
            else:
                cost = dw - 1
            
            if cost < best_cost:
                best_cost = cost
                best_w1 = w
                best_diff = diff
        
        if best_w1 is not None:
            diff = best_diff
            if diff == 0:
                return xor_wires(best_w1, n)
            elif diff in val_to_wire:
                return xor_wires(best_w1, val_to_wire[diff])
            else:
                diff_wire = build_from_bits(diff)
                return xor_wires(best_w1, diff_wire)
        
        return build_from_bits(v)
    
    row_indices = list(range(m))
    row_indices.sort(key=lambda i: rows[i].bit_count())
    
    outputs = [None] * m
    for idx in row_indices:
        outputs[idx] = get_wire(rows[idx])
    
    return {'gates': gates, 'outputs': outputs}
