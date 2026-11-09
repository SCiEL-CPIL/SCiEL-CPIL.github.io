import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0
    
    matrix = np.array(matrix, dtype=np.uint8)
    if matrix.size == 0:
        return {'gates': [], 'outputs': []}
    
    n, m = matrix.shape
    
    masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j]:
                mask |= (1 << j)
        masks.append(mask)
    
    available = {0: n}
    for j in range(n):
        available[1 << j] = j
    
    gates = []
    next_wire = n + 1
    
    def create_gate(w1, w2):
        nonlocal next_wire
        gates.append([w1, w2])
        w = next_wire
        next_wire += 1
        return w
    
    def build_mask(mask):
        if mask in available:
            return available[mask]
        if mask == 0:
            return n
        
        # Find best split
        best = None
        best_cost = float('inf')
        sub = mask
        while sub:
            other = mask ^ sub
            if other != 0:
                c1 = 0 if sub in available else bin(sub).count('1') - 1
                c2 = 0 if other in available else bin(other).count('1') - 1
                cost = c1 + c2 + 1
                if cost < best_cost:
                    best_cost = cost
                    best = (sub, other)
            sub = (sub - 1) & mask
        
        if best is None:
            bits = []
            temp = mask
            while temp:
                lb = temp & (-temp)
                bits.append(lb)
                temp ^= lb
            acc = available[bits[0]]
            for b in bits[1:]:
                acc = create_gate(acc, available[b])
            available[mask] = acc
            return acc
        
        s1, s2 = best
        w1 = build_mask(s1)
        w2 = build_mask(s2)
        w = create_gate(w1, w2)
        available[mask] = w
        return w
    
    outputs = []
    for mask in masks:
        if time.time() > deadline:
            # Fallback
            if mask == 0:
                outputs.append(n)
            else:
                bits = []
                temp = mask
                while temp:
                    lb = temp & (-temp)
                    bits.append(lb)
                    temp ^= lb
                if len(bits) == 1:
                    outputs.append(available[bits[0]])
                else:
                    acc = available[bits[0]]
                    for b in bits[1:]:
                        gates.append([acc, available[b]])
                        acc = next_wire
                        next_wire += 1
                    outputs.append(acc)
        else:
            outputs.append(build_mask(mask))
    
    return {'gates': gates, 'outputs': outputs}
