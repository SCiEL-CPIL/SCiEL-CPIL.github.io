"""
XOR-circuit synthesis using Greedy Common Subexpression Elimination (CSE) 
on bitmask representations with optimized pair scoring.

Strategy:
1. Represent each output row as a bitmask (Python int).
2. Maintain a pool of available wires (inputs 0..n-1, zero n, and generated gates).
3. Iteratively find pairs of wires (a, b) that appear together in at least two 
   current expressions. Creating a gate g = a XOR b and replacing a, b with g 
   in those expressions reduces the total gate count.
4. Use precomputed co-occurrence matrix for fast pair scoring.
5. Repeat until no more beneficial pairs are found or time limit is reached.
6. Realize the final expressions as chains of XOR gates.
7. Verify the result and return the best valid circuit found.
"""
import time
import random
import sys

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
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

    def _greedy_cse(row_bits, n, m, time_budget, seed):
        random.seed(seed)
        
        exprs = list(row_bits)
        gates = []
        next_wire = n + 1
        pool = list(range(n + 1)) # inputs + zero
        
        max_iter = 2000
        for _ in range(max_iter):
            if time.time() > time_budget:
                break
                
            best_pair = None
            best_benefit = 1 # Must be > 1 to be useful (strict improvement in gate count)
            
            pool_size = len(pool)
            if pool_size < 2:
                break
            
            # Pre-compute counts of each wire in exprs to filter candidates
            wire_counts = {}
            for w in pool:
                w_bit = 1 << w
                count = 0
                for e in exprs:
                    if e & w_bit:
                        count += 1
                if count >= 2:
                    wire_counts[w] = count
            
            if len(wire_counts) < 2:
                break
                
            candidate_wires = list(wire_counts.keys())
            num_candidates = len(candidate_wires)
            
            # Build index mapping for faster access
            wire_to_idx = {w: i for i, w in enumerate(candidate_wires)}
            
            # Precompute co-occurrence: for each pair of candidates, count exprs containing both
            # Only consider pairs where both have count >= 2
            best_benefit = 1
            best_pair = None
            
            # For efficiency, only check pairs where the smaller count limits the max benefit
            for i_idx in range(num_candidates):
                a = candidate_wires[i_idx]
                a_bit = 1 << a
                count_a = wire_counts[a]
                
                # Early skip: if count_a <= best_benefit, no pair with a can beat best
                if count_a <= best_benefit:
                    continue
                    
                for j_idx in range(i_idx + 1, num_candidates):
                    b = candidate_wires[j_idx]
                    b_bit = 1 << b
                    count_b = wire_counts[b]
                    
                    # Upper bound on benefit
                    upper_bound = min(count_a, count_b)
                    if upper_bound <= best_benefit:
                        continue
                        
                    # Count actual co-occurrence
                    benefit = 0
                    for e in exprs:
                        if (e & a_bit) and (e & b_bit):
                            benefit += 1
                            if benefit > best_benefit:
                                best_benefit = benefit
                                best_pair = (a, b)
                                break  # No need to count further for this pair
                    
                    if best_pair is not None and best_benefit >= upper_bound:
                        # Can't do better than this pair's upper bound
                        pass
                    
            if best_pair is None:
                break
                
            a, b = best_pair
            g = next_wire
            next_wire += 1
            gates.append([a, b])
            pool.append(g)
            
            # Update exprs: replace a and b with g where both are present
            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g
            
            for i in range(m):
                e = exprs[i]
                if (e & a_bit) and (e & b_bit):
                    # Remove a and b, add g
                    e = (e & ~a_bit & ~b_bit) | g_bit
                    exprs[i] = e
                    
        # Final reduction: realize remaining expressions as chains
        outputs = []
        for i in range(m):
            e = exprs[i]
            if e == 0:
                outputs.append(n)
            else:
                # Extract wires
                wires = []
                w = 0
                temp_e = e
                while temp_e:
                    if temp_e & 1:
                        wires.append(w)
                    temp_e >>= 1
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
        
        # Check wire indices
        for k, (a, b) in enumerate(gates):
            if a >= num_wires or b >= num_wires or a < 0 or b < 0:
                return False
        
        # Check output indices
        for o in outputs:
            if o >= num_wires or o < 0:
                return False

        # Simulate
        # wire_val[i] represents the bitmask of the value on wire i
        # This is a bit of a trick: we simulate the circuit on the "basis" 
        # where input j is 1<<j.
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0
        
        for k, (a, b) in enumerate(gates):
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
            
        for i in range(m):
            expected = row_bits[i]
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
    
    # Run multiple seeds to explore different CSE paths
    # First run is deterministic (seed 0)
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    
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
        # Fallback: Simple baseline (chain per row)
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            bits = row_bits[i]
            if bits == 0:
                outputs.append(n)
            else:
                indices = []
                temp_bits = bits
                w = 0
                while temp_bits:
                    if temp_bits & 1:
                        indices.append(w)
                    temp_bits >>= 1
                    w += 1
                    
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
