import time
import random
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
        
    # Convert to bitsets
    row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if matrix[i][j]:
                bits |= (1 << j)
        row_bits.append(bits)
        
    # If all zeros
    if not any(row_bits):
        return {'gates': [], 'outputs': [n] * m}

    def _greedy_cse(row_bits, n, m, time_budget, seed=None):
        if seed is not None:
            random.seed(seed)
            
        exprs = list(row_bits)
        gates = []
        next_wire = n + 1
        pool = list(range(n + 1)) # inputs + zero
        
        # Pre-compute counts of each wire in exprs?
        # We can just iterate.
        
        max_iter = 500
        for _ in range(max_iter):
            if time.time() > time_budget:
                break
                
            best_pair = None
            best_benefit = 1 # Must be > 1 to be useful
            
            # Optimization: Only consider pairs where both wires appear in at least 2 exprs
            # This is a heuristic to reduce search space.
            # Let's just check all pairs in pool for simplicity first, 
            # but pool size can grow.
            
            pool_size = len(pool)
            # To handle random tie-breaking, we can shuffle the pool or the pairs
            # Let's shuffle the pool indices for iteration
            indices = list(range(pool_size))
            random.shuffle(indices)
            
            for i_idx in indices:
                a = pool[i_idx]
                a_bit = 1 << a
                # Count how many exprs have a
                count_a = 0
                for e in exprs:
                    if e & a_bit:
                        count_a += 1
                if count_a < 2:
                    continue
                    
                for j_idx in indices:
                    if j_idx == i_idx:
                        continue
                    b = pool[j_idx]
                    b_bit = 1 << b
                    
                    # Count how many exprs have both a and b
                    benefit = 0
                    for e in exprs:
                        if (e & a_bit) and (e & b_bit):
                            benefit += 1
                            
                    if benefit > best_benefit:
                        best_benefit = benefit
                        best_pair = (a, b)
                        
            if best_pair is None:
                break
                
            a, b = best_pair
            g = next_wire
            next_wire += 1
            gates.append([a, b])
            pool.append(g)
            
            # Update exprs
            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g
            
            for i in range(m):
                e = exprs[i]
                if (e & a_bit) and (e & b_bit):
                    # Remove a and b, add g
                    e = (e & ~a_bit & ~b_bit) | g_bit
                    exprs[i] = e
                    
        # Final reduction
        outputs = []
        for i in range(m):
            e = exprs[i]
            if e == 0:
                outputs.append(n)
            else:
                # Extract wires
                wires = []
                w = 0
                while e:
                    if e & 1:
                        wires.append(w)
                    e >>= 1
                    w += 1
                    
                if len(wires) == 1:
                    outputs.append(wires[0])
                else:
                    prev = wires[0]
                    for w in wires[1:]:
                        gates.append([prev, w])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
                    
        return {'gates': gates, 'outputs': outputs}

    def _verify(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
        num_wires = n + 1 + len(gates)
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0
        
        for k, (a, b) in enumerate(gates):
            if a >= num_wires or b >= num_wires:
                return False
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
            
        for i in range(m):
            expected = row_bits[i]
            if outputs[i] >= num_wires:
                return False
            if wire_val[outputs[i]] != expected:
                return False
        return True

    def _compute_depth(gates, outputs):
        if not gates:
            return 0
        num_wires = n + 1 + len(gates)
        wire_depth = [0] * num_wires
        for k, (a, b) in enumerate(gates):
            wire_depth[n + 1 + k] = max(wire_depth[a], wire_depth[b]) + 1
        max_depth = 0
        for w in outputs:
            if w < num_wires:
                max_depth = max(max_depth, wire_depth[w])
        return max_depth

    best_result = None
    best_gate_count = float('inf')
    
    # Run multiple seeds
    # First run is deterministic (seed 0)
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for seed in seeds:
        if time.time() > deadline - 1.0:
            break
            
        try:
            result = _greedy_cse(row_bits, n, m, deadline - 1.0, seed)
            if _verify(result):
                d = _compute_depth(result['gates'], result['outputs'])
                if d <= depth_limit:
                    if len(result['gates']) < best_gate_count:
                        best_gate_count = len(result['gates'])
                        best_result = result
        except Exception:
            pass

    if best_result is None:
        # Fallback: Simple baseline
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            bits = row_bits[i]
            if bits == 0:
                outputs.append(n)
            else:
                indices = [j for j in range(n) if (bits >> j) & 1]
                if len(indices) == 1:
                    outputs.append(indices[0])
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        best_result = {'gates': gates, 'outputs': outputs}
        
    return best_result
