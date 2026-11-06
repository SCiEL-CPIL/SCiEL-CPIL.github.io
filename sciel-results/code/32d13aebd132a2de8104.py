import time
import random
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin from 90s

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    A = np.array(matrix, dtype=np.int8)

    if m == 0:
        return {'gates': [], 'outputs': []}

    if not np.any(A):
        return {'gates': [], 'outputs': [n] * m}

    # Convert rows to bitsets (Python ints)
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

    def _finalize_circuit(expressions, gates, next_wire, n, m):
        """
        Takes the final state of expressions and gates, and constructs the full circuit
        by adding any remaining XORs needed to reduce each expression to a single wire.
        """
        outputs = []
        # We need to be careful with next_wire as we add gates here
        for i in range(m):
            expr = expressions[i]
            if expr == 0:
                outputs.append(n)
            else:
                # Find set bits in the expression
                # The expression is a bitset of wire indices
                wires = []
                # Max wire index is next_wire - 1
                for w in range(next_wire):
                    if (expr >> w) & 1:
                        wires.append(w)
                        
                if len(wires) == 1:
                    outputs.append(wires[0])
                else:
                    # Chain XORs to reduce to a single wire
                    prev = wires[0]
                    for w in wires[1:]:
                        gates.append([prev, w])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
                    
        return {'gates': gates, 'outputs': outputs}

    def _greedy_cse_with_randomness(row_bits, n, m, time_budget, seed=0):
        """
        Greedy CSE with random tie-breaking to escape local minima.
        """
        rng = random.Random(seed)
        expressions = list(row_bits)
        gates = []
        next_wire = n + 1
        pool = set(range(n))
        
        # Precompute popcount for speed? No, Python's bin(x).count('1') is fast enough for small ints.
        # But we can use int.bit_count() in Python 3.10+
        # Let's check if bit_count is available.
        if hasattr(int, 'bit_count'):
            def popcount(x):
                return x.bit_count()
        else:
            def popcount(x):
                return bin(x).count('1')

        while time.time() < time_budget:
            # Find all candidate pairs (a, b) in pool that appear together in >1 expression
            current_pool = list(pool)
            pool_size = len(current_pool)
            
            if pool_size < 2:
                break
                
            candidates = []
            
            # Optimization: If pool is large, this is O(P^2 * M).
            # n <= 16, gates usually small, so P is small.
            for i in range(pool_size):
                a = current_pool[i]
                a_bit = 1 << a
                for j in range(i + 1, pool_size):
                    b = current_pool[j]
                    b_bit = 1 << b
                    count = 0
                    for expr in expressions:
                        if (expr & a_bit) and (expr & b_bit):
                            count += 1
                    if count > 1:
                        # Score this candidate
                        # Heuristic: Higher count is better.
                        # Also consider depth?
                        # Let's stick to count for now, maybe add a small penalty for creating deep wires.
                        # Depth of new wire g = max(depth(a), depth(b)) + 1
                        # We don't track depth in state currently. Let's add it?
                        # For simplicity, let's just use count.
                        candidates.append((count, a, b))
                        
            if not candidates:
                break
                
            # Random tie-breaking among top candidates
            # Sort by count descending, then shuffle within same count
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Get the max count
            max_count = candidates[0][0]
            
            # Filter to only top count candidates
            top_candidates = [c for c in candidates if c[0] == max_count]
            
            # Randomly pick one
            if top_candidates:
                count, a, b = rng.choice(top_candidates)
            else:
                break
                
            # Create new gate
            g = next_wire
            gates.append([a, b])
            next_wire += 1
            pool.add(g)
            
            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g
            
            # Update expressions
            new_exprs = []
            for expr in expressions:
                if (expr & a_bit) and (expr & b_bit):
                    new_expr = (expr & ~a_bit & ~b_bit) | g_bit
                    new_exprs.append(new_expr)
                else:
                    new_exprs.append(expr)
            expressions = new_exprs
            
            # Check if all expressions are reduced to single wires or zero
            all_reduced = True
            for expr in expressions:
                if expr == 0:
                    continue
                if popcount(expr) > 1:
                    all_reduced = False
                    break
            if all_reduced:
                break
                
        circuit = _finalize_circuit(expressions, gates, next_wire, n, m)
        return circuit

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
            
    # 2. Greedy CSE with multiple random restarts
    try:
        num_restarts = 8
        time_per_restart = (deadline - 3.0) / num_restarts
        
        for seed in range(num_restarts):
            if time.time() > deadline - 2.0:
                break
            
            result = _greedy_cse_with_randomness(row_bits, n, m, deadline - 2.0, seed=seed)
            
            if result and _verify(result):
                d = _compute_depth(result['gates'], result['outputs'])
                if d <= depth_limit:
                    gc = len(result['gates'])
                    if gc < best_gate_count:
                        best_gate_count = gc
                        best_result = result
                        
    except Exception:
        pass

    if best_result is None:
        return baseline
        
    return best_result
