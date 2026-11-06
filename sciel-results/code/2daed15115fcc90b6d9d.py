import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    A = np.array(matrix, dtype=np.int8)

    if m == 0:
        return {'gates': [], 'outputs': []}

    if not np.any(A):
        return {'gates': [], 'outputs': [n] * m}

    # Convert rows to bitsets (Python ints) for efficient manipulation
    row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if A[i, j]:
                bits |= (1 << j)
        row_bits.append(bits)

    def _make_baseline():
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

    def _greedy_global_cse(row_bits, n, m, time_budget):
        """
        Global Common Subexpression Elimination using bitsets.
        1. Maintain a set of 'available' wires (inputs + created gates).
        2. Maintain 'expressions' for each output (bitset of available wires).
        3. Repeatedly find the pair of wires (a, b) in the pool that appears
           together in the most expressions.
        4. Create a new gate g = a ^ b, update expressions replacing {a, b} with {g}.
        5. Continue until no beneficial pair exists or time runs out.
        """
        expressions = list(row_bits)
        gates = []
        next_wire = n + 1
        pool = set(range(n))
        
        # Pre-compute masks for efficiency? Not really needed for small n.
        
        max_iterations = 200
        
        for _ in range(max_iterations):
            if time.time() > time_budget:
                break
                
            best_pair = None
            best_count = 1 # Must be > 1 to be beneficial (saves at least 1 gate)
            
            # Get current pool
            current_pool = list(pool)
            pool_size = len(current_pool)
            
            # Find best pair
            # Complexity: O(P^2 * M) where P is pool size, M is num outputs
            # P <= n + num_gates. n<=16, gates usually small.
            for i in range(pool_size):
                a = current_pool[i]
                a_bit = 1 << a
                for j in range(i + 1, pool_size):
                    b = current_pool[j]
                    b_bit = 1 << b
                    # Count how many expressions contain both a and b
                    count = 0
                    for expr in expressions:
                        if (expr & a_bit) and (expr & b_bit):
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_pair = (a, b)
                        
            if best_pair is None:
                break
                
            a, b = best_pair
            # Create new gate G = a ^ b
            g = next_wire
            next_wire += 1
            gates.append([a, b])
            pool.add(g)
            
            # Update expressions
            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g
            
            for idx, expr in enumerate(expressions):
                if (expr & a_bit) and (expr & b_bit):
                    # Replace a and b with g
                    # Clear a and b, set g
                    new_expr = (expr & ~a_bit & ~b_bit) | g_bit
                    expressions[idx] = new_expr
                    
        # Finalize outputs
        outputs = []
        for i in range(m):
            expr = expressions[i]
            if expr == 0:
                outputs.append(n)
            else:
                # Need to reduce expr to a single wire
                # Find set bits
                wires = []
                # Iterate through possible wire indices
                # Max wire index is next_wire - 1
                for w in range(next_wire):
                    if (expr >> w) & 1:
                        wires.append(w)
                        
                if len(wires) == 1:
                    outputs.append(wires[0])
                else:
                    # Chain XORs
                    prev = wires[0]
                    for w in wires[1:]:
                        gates.append([prev, w])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
                    
        return {'gates': gates, 'outputs': outputs}

    def _greedy_pair_sharing(row_bits, n, m, time_budget):
        """
        Alternative: Greedy pairwise sharing based on common inputs.
        This tends to create deeper circuits but might find different structures.
        """
        expressions = list(row_bits)
        gates = []
        next_wire = n + 1
        wire_depth = [0] * (n + 1 + m * m + 100)
        
        max_iterations = min(500, m * m * 3)
        
        for _ in range(max_iterations):
            if time.time() > time_budget:
                break
            
            best_pair = None
            best_subset = 0
            best_savings = 0
            
            for i in range(m):
                if expressions[i] == 0:
                    continue
                for j in range(i + 1, m):
                    if expressions[j] == 0:
                        continue
                    # Find common input wires (only inputs 0..n-1)
                    # Note: expressions might contain gate wires now.
                    # We only share if they share INPUT wires?
                    # Or any wires?
                    # If they share gate wires, that implies they already used a shared subcircuit.
                    # Sharing inputs is the primary driver.
                    common = expressions[i] & expressions[j] & ((1 << n) - 1)
                    savings = bin(common).count('1') - 1
                    if savings > best_savings:
                        best_savings = savings
                        best_pair = (i, j)
                        best_subset = common
            
            if best_pair is None:
                break
            
            i, j = best_pair
            S = best_subset
            
            # Create gates to compute XOR(S)
            S_list = [k for k in range(n) if (S >> k) & 1]
            
            if len(S_list) == 2:
                gates.append([S_list[0], S_list[1]])
                shared_wire = next_wire
                next_wire += 1
                if shared_wire < len(wire_depth):
                    wire_depth[shared_wire] = max(wire_depth[S_list[0]], wire_depth[S_list[1]]) + 1
            else:
                prev = S_list[0]
                for k in range(1, len(S_list)):
                    gates.append([prev, S_list[k]])
                    new_wire = next_wire
                    next_wire += 1
                    if new_wire < len(wire_depth):
                        wire_depth[new_wire] = max(wire_depth[prev], wire_depth[S_list[k]]) + 1
                    prev = new_wire
                shared_wire = prev
            
            # Update expressions
            S_mask = S
            shared_bit = 1 << shared_wire
            
            expressions[i] = (expressions[i] & ~S_mask) | shared_bit
            expressions[j] = (expressions[j] & ~S_mask) | shared_bit
        
        # Build final outputs
        outputs = []
        for i in range(m):
            expr = expressions[i]
            if expr == 0:
                outputs.append(n)
            else:
                expr_list = [w for w in range(n + 1 + len(gates)) if (expr >> w) & 1]
                
                if len(expr_list) == 1:
                    outputs.append(expr_list[0])
                else:
                    prev = expr_list[0]
                    for w in expr_list[1:]:
                        gates.append([prev, w])
                        new_wire = next_wire
                        next_wire += 1
                        if new_wire < len(wire_depth):
                            wire_depth[new_wire] = max(wire_depth[prev], wire_depth[w]) + 1
                        prev = new_wire
                    outputs.append(prev)
        
        return {'gates': gates, 'outputs': outputs}

    # Try strategies
    best_result = None
    best_gate_count = float('inf')
    
    # 1. Baseline
    baseline = _make_baseline()
    if _verify(baseline):
        d = _compute_depth(baseline['gates'], baseline['outputs'])
        if d <= depth_limit:
            best_result = baseline
            best_gate_count = len(baseline['gates'])
            
    # 2. Greedy Global CSE (Primary improvement)
    try:
        # Use most of the time budget here
        time_left = deadline - time.time()
        if time_left > 0.1:
            result = _greedy_global_cse(row_bits, n, m, deadline - 0.5)
            if _verify(result):
                d = _compute_depth(result['gates'], result['outputs'])
                if d <= depth_limit:
                    if len(result['gates']) < best_gate_count:
                        best_result = result
                        best_gate_count = len(result['gates'])
    except Exception:
        pass
        
    # 3. Greedy Pair Sharing (Secondary, if time remains)
    try:
        time_left = deadline - time.time()
        if time_left > 0.5:
            result = _greedy_pair_sharing(row_bits, n, m, deadline - 0.5)
            if _verify(result):
                d = _compute_depth(result['gates'], result['outputs'])
                if d <= depth_limit:
                    if len(result['gates']) < best_gate_count:
                        best_result = result
                        best_gate_count = len(result['gates'])
    except Exception:
        pass
        
    if best_result is None:
        return baseline
        
    return best_result
