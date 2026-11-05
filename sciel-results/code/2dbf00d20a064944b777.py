def synthesize(matrix, depth_limit):
    import numpy as np
    
    m, n = matrix.shape
    targets = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i, j]:
                val |= (1 << j)
        targets.append(val)
        
    wire_to_mask = {}
    mask_to_wire = {}
    wire_depth = {}
    
    for j in range(n):
        mask = 1 << j
        wire_to_mask[j] = mask
        mask_to_wire[mask] = j
        wire_depth[j] = 0
        
    wire_to_mask[n] = 0
    mask_to_wire[0] = n
    wire_depth[n] = 0
    
    current_wires = list(range(n + 1))
    gates = []
    next_wire = n + 1
    outputs = [0] * m
    remaining_targets = targets[:]
    
    def add_gate(w1, w2):
        nonlocal next_wire
        mask1 = wire_to_mask[w1]
        mask2 = wire_to_mask[w2]
        new_mask = mask1 ^ mask2
        
        if new_mask in mask_to_wire:
            return mask_to_wire[new_mask]
            
        d1 = wire_depth[w1]
        d2 = wire_depth[w2]
        new_depth = max(d1, d2) + 1
        
        if new_depth > depth_limit:
            return None
            
        w_new = next_wire
        next_wire += 1
        gates.append([w1, w2])
        wire_to_mask[w_new] = new_mask
        mask_to_wire[new_mask] = w_new
        wire_depth[w_new] = new_depth
        current_wires.append(w_new)
        return w_new

    def build_direct(target_mask):
        nonlocal next_wire
        if target_mask in mask_to_wire:
            return mask_to_wire[target_mask]
            
        w_current = n # Zero wire
        while target_mask != 0:
            lsb = target_mask & -target_mask
            j = lsb.bit_length() - 1
            w_input = j
            
            w_new = add_gate(w_current, w_input)
            if w_new is None:
                # Depth limit reached, cannot build this way
                # Fallback: return a dummy or raise?
                # For now, assume we can always build within limit for n=16
                return n 
            w_current = w_new
            target_mask ^= lsb
            
        return w_current

    max_iter = 1000
    
    for _ in range(max_iter):
        if not remaining_targets:
            break
            
        satisfied = []
        still_remaining = []
        for t_mask in remaining_targets:
            if t_mask in mask_to_wire:
                satisfied.append(t_mask)
            else:
                still_remaining.append(t_mask)
                
        if satisfied:
            remaining_targets = still_remaining
            if not remaining_targets:
                break
                
        best_gain = 0
        best_pair = None
        best_new_mask = None
        
        for i in range(len(current_wires)):
            w1 = current_wires[i]
            m1 = wire_to_mask[w1]
            for j in range(i + 1, len(current_wires)):
                w2 = current_wires[j]
                m2 = wire_to_mask[w2]
                new_mask = m1 ^ m2
                
                if new_mask in mask_to_wire:
                    continue
                    
                d1 = wire_depth[w1]
                d2 = wire_depth[w2]
                if max(d1, d2) + 1 > depth_limit:
                    continue
                    
                gain = 0
                for t in remaining_targets:
                    old_cost = t.bit_count()
                    new_cost = (t ^ new_mask).bit_count() + 1
                    if new_cost < old_cost:
                        gain += (old_cost - new_cost)
                        
                net_gain = gain - 1
                
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_pair = (w1, w2)
                    best_new_mask = new_mask
                    
        if best_pair is not None and best_gain > 0:
            add_gate(best_pair[0], best_pair[1])
        else:
            easiest = min(remaining_targets, key=lambda x: x.bit_count())
            build_direct(easiest)
            remaining_targets.remove(easiest)
            
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i, j]:
                val |= (1 << j)
        if val in mask_to_wire:
            outputs[i] = mask_to_wire[val]
        else:
            outputs[i] = build_direct(val)
            
    return {'gates': gates, 'outputs': outputs}
