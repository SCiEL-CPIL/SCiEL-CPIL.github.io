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

    def _beam_search_cse(row_bits, n, m, time_budget, beam_width=5, max_iterations=100):
        """
        Beam search for Common Subexpression Elimination.
        State: (expressions, gates, next_wire, pool)
        Score: Estimated total gates = current_gates + sum(popcount(expr) - 1 for expr in expressions)
        We prioritize states with lower estimated total gates.
        """
        # Initial state
        initial_exprs = list(row_bits)
        initial_gates = []
        initial_next_wire = n + 1
        initial_pool = set(range(n))
        
        # State representation for beam
        # We store: [expressions, gates, next_wire, pool, score]
        # To make it hashable for deduplication if needed, we could use tuples, 
        # but expressions are lists of ints. Let's just keep a list of states.
        
        # Helper to estimate remaining gates
        def estimate_remaining_gates(exprs):
            total = 0
            for e in exprs:
                if e == 0:
                    continue
                # popcount(e) - 1
                total += bin(e).count('1') - 1
            return total

        current_beam = []
        # Start with one state
        current_beam.append({
            'exprs': list(initial_exprs),
            'gates': list(initial_gates),
            'next_wire': initial_next_wire,
            'pool': set(initial_pool),
            'score': estimate_remaining_gates(initial_exprs)
        })
        
        best_state = None
        best_final_score = float('inf')
        
        for _iter in range(max_iterations):
            if time.time() > time_budget:
                break
                
            next_beam = []
            
            for state in current_beam:
                exprs = state['exprs']
                gates = state['gates']
                next_wire = state['next_wire']
                pool = state['pool']
                
                # If all expressions are reduced to single wires or zero, we are done for this branch
                # But we still need to finalize. 
                # Actually, if no pair has count > 1, we can't improve further by CSE.
                
                # Find candidate pairs
                # We want to find pairs (a,b) in pool that appear together in >1 expression.
                # To speed up, we can limit the candidates we evaluate.
                current_pool = list(pool)
                pool_size = len(current_pool)
                
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
                        for expr in exprs:
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
                            
                # Sort candidates by count descending
                candidates.sort(key=lambda x: x[0], reverse=True)
                
                # Take top K candidates for this state
                K = 3
                for count, a, b in candidates[:K]:
                    # Create new state
                    new_gates = gates + [[a, b]]
                    g = next_wire
                    new_next_wire = next_wire + 1
                    new_pool = pool | {g}
                    
                    a_bit = 1 << a
                    b_bit = 1 << b
                    g_bit = 1 << g
                    
                    new_exprs = []
                    for expr in exprs:
                        if (expr & a_bit) and (expr & b_bit):
                            new_expr = (expr & ~a_bit & ~b_bit) | g_bit
                            new_exprs.append(new_expr)
                        else:
                            new_exprs.append(expr)
                            
                    new_score = estimate_remaining_gates(new_exprs)
                    
                    next_beam.append({
                        'exprs': new_exprs,
                        'gates': new_gates,
                        'next_wire': new_next_wire,
                        'pool': new_pool,
                        'score': new_score
                    })
                    
            if not next_beam:
                break
                
            # Keep only the top beam_width states from next_beam
            next_beam.sort(key=lambda x: x['score'])
            current_beam = next_beam[:beam_width]
            
            # Check if any state in the beam is "complete" (no more beneficial pairs)
            # If so, finalize it and update best
            # Actually, we should finalize all states in the beam at the end of the loop?
            # No, we should check if we can stop early.
            
            # Let's check if the best state in the beam has no more candidates
            # If the best state has no candidates, it's a local optimum for that branch.
            # We can finalize it.
            
            # To save time, we only finalize if we are running out of time or iterations
            # Or if the score is very good.
            
            # For now, we just let the loop run.
            
        # After the beam search, finalize the best state found
        # We need to pick the best state from the final beam or any intermediate one?
        # The beam search keeps the best states. The final beam contains the most promising states.
        # We should finalize all states in the final beam and pick the one with the fewest actual gates.
        
        best_circuit = None
        best_gate_count = float('inf')
        
        for state in current_beam:
            # Check time
            if time.time() > time_budget - 1.0:
                break
                
            circuit = _finalize_circuit(state['exprs'], list(state['gates']), state['next_wire'], n, m)
            
            # Verify
            if _verify(circuit):
                d = _compute_depth(circuit['gates'], circuit['outputs'])
                if d <= depth_limit:
                    gc = len(circuit['gates'])
                    if gc < best_gate_count:
                        best_gate_count = gc
                        best_circuit = circuit
                        
        if best_circuit:
            return best_circuit
        else:
            return None

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
            
    # 2. Beam Search CSE
    try:
        time_left = deadline - time.time()
        if time_left > 0.5:
            # Use a slightly smaller beam width for speed if n is large
            # n <= 16, so beam_width 5 is fine.
            result = _beam_search_cse(row_bits, n, m, deadline - 2.0, beam_width=5, max_iterations=150)
            if result:
                if len(result['gates']) < best_gate_count:
                    best_result = result
                    best_gate_count = len(result['gates'])
    except Exception:
        pass

    # 3. Fallback to simple greedy if beam search failed or took too long
    if time.time() < deadline - 5.0:
        try:
            # Simple greedy is fast, use as a fallback or to see if it finds something better
            # (Unlikely to beat beam search, but safe)
            # We can skip this if beam search worked well.
            pass
        except Exception:
            pass
            
    if best_result is None:
        return baseline
        
    return best_result
