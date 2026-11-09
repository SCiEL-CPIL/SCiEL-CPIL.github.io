"""XOR circuit synthesis using greedy set-cover with depth tracking."""
import time
import heapq
import random

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = 85.0  # seconds
    
    if matrix is None or len(matrix) == 0:
        return {'gates': [], 'outputs': []}
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    def row_to_int(row):
        val = 0
        for i in range(n):
            if row[i] % 2 != 0:
                val |= (1 << i)
        return val
    
    outputs = [row_to_int(row) for row in matrix]
    
    # Wire management with depth tracking
    # wire_vec[wire_id] = vector value
    # wire_depth[wire_id] = depth (max depth of inputs)
    wire_vec = {}
    wire_depth = {}
    vec_to_wire = {}  # vector -> wire_id (first occurrence)
    
    next_wire = 0
    gates = []  # list of (wire_a, wire_b)
    
    def add_wire(vec, depth):
        nonlocal next_wire
        if vec in vec_to_wire:
            return vec_to_wire[vec]
        wid = next_wire
        next_wire += 1
        wire_vec[wid] = vec
        wire_depth[wid] = depth
        vec_to_wire[vec] = wid
        return wid
    
    # Initialize inputs
    for i in range(n):
        add_wire(1 << i, 0)
    add_wire(0, 0)  # constant zero
    
    # Track output wires
    output_wires = [None] * m
    output_done = [False] * m
    
    for i in range(m):
        if outputs[i] in vec_to_wire:
            output_wires[i] = vec_to_wire[outputs[i]]
            output_done[i] = True
    
    if all(output_done):
        return {'gates': gates, 'outputs': output_wires}
    
    # Greedy: add most useful intermediate wires
    max_steps = min(5 * m + n, 500)
    
    for step in range(max_steps):
        if time.time() - start_time > DEADLINE:
            break
        
        if all(output_done):
            break
        
        # Find best pair to XOR
        best_score = -1
        best_a = None
        best_b = None
        
        wires = list(wire_vec.keys())
        num_wires = len(wires)
        
        # Limit pairs for speed
        if num_wires > 100:
            random.seed(step)
            pairs = set()
            for _ in range(3000):
                i = random.randint(0, num_wires - 1)
                j = random.randint(0, num_wires - 1)
                if i != j:
                    pairs.add((min(i, j), max(i, j)))
        else:
            pairs = [(i, j) for i in range(num_wires) for j in range(i + 1, num_wires)]
        
        for (i, j) in pairs:
            wi = wires[i]
            wj = wires[j]
            vi = wire_vec[wi]
            vj = wire_vec[wj]
            vc = vi ^ vj
            if vc in vec_to_wire:
                continue
            
            # Score
            score = 0
            for k in range(m):
                if output_done[k]:
                    continue
                o = outputs[k]
                rem = o ^ vc
                if rem in vec_to_wire:
                    score += 100
                elif bin(rem).count('1') < bin(o).count('1'):
                    score += 1
                elif rem == 0:
                    score += 50
            
            if score > best_score:
                best_score = score
                best_a = wi
                best_b = wj
        
        if best_a is None or best_score <= 0:
            break
        
        # Add the gate
        gates.append((best_a, best_b))
        vc = wire_vec[best_a] ^ wire_vec[best_b]
        new_depth = max(wire_depth[best_a], wire_depth[best_b]) + 1
        if new_depth > depth_limit:
            # Skip this gate if it exceeds depth
            gates.pop()
            continue
        add_wire(vc, new_depth)
        
        # Update output status
        for k in range(m):
            if output_done[k]:
                continue
            if outputs[k] in vec_to_wire:
                output_wires[k] = vec_to_wire[outputs[k]]
                output_done[k] = True
    
    # Fallback for unproduced outputs
    for k in range(m):
        if output_done[k]:
            continue
        o = outputs[k]
        if o in vec_to_wire:
            output_wires[k] = vec_to_wire[o]
            output_done[k] = True
            continue
        
        # Direct construction
        bits = [i for i in range(n) if (o >> i) & 1]
        if not bits:
            output_wires[k] = vec_to_wire[0]
            output_done[k] = True
            continue
        if len(bits) == 1:
            output_wires[k] = vec_to_wire[1 << bits[0]]
            output_done[k] = True
            continue
        
        acc = 1 << bits[0]
        acc_wire = vec_to_wire[acc]
        for i in range(1, len(bits)):
            b = bits[i]
            new_acc = acc | (1 << b)
            if new_acc in vec_to_wire:
                acc = new_acc
                acc_wire = vec_to_wire[acc]
            else:
                gates.append((acc_wire, vec_to_wire[1 << b]))
                new_depth = max(wire_depth[acc_wire], wire_depth[vec_to_wire[1 << b]]) + 1
                add_wire(new_acc, new_depth)
                acc = new_acc
                acc_wire = vec_to_wire[acc]
        
        output_wires[k] = acc_wire
        output_done[k] = True
    
    return {'gates': gates, 'outputs': output_wires}
