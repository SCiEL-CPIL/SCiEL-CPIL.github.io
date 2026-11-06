import time
import random
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert to numpy
    A = np.array(matrix, dtype=np.int8)

    if not np.any(A):
        return {'gates': [], 'outputs': [n] * m}

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
            expected = 0
            for j in range(n):
                if A[i][j]:
                    expected |= (1 << j)
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

    def _baseline_circuit():
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            row = A[i]
            if not np.any(row):
                outputs.append(n)
            else:
                indices = np.where(row == 1)[0].tolist()
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

    def _cost_of_vector(vec_int):
        """Cost to synthesize a vector from inputs (baseline style)."""
        w = bin(vec_int).count('1')
        if w == 0:
            return 0
        if w == 1:
            return 0
        return w - 1

    def _build_greedy_circuit(A_mat, time_budget):
        """
        Greedy Shared Intermediate approach.
        1. Generate pool of candidate intermediates (pairwise XORs of inputs).
        2. Greedily select intermediates that reduce total synthesis cost.
        3. Construct circuit.
        """
        m_curr = A_mat.shape[0]
        n_curr = A_mat.shape[1]
        
        # Convert rows to int bitsets for fast XOR and weight
        rows = []
        for i in range(m_curr):
            bits = 0
            for j in range(n_curr):
                if A_mat[i][j]:
                    bits |= (1 << j)
            rows.append(bits)
            
        # Pool of candidates: 
        # 0: zero (cost 0)
        # 1..n: inputs (cost 0)
        # n+1..: pairwise XORs of inputs
        candidates = [0]
        input_bits = [1 << i for i in range(n_curr)]
        candidates.extend(input_bits)
        
        # Add pairwise XORs
        for i in range(n_curr):
            for j in range(i + 1, n_curr):
                candidates.append(input_bits[i] ^ input_bits[j])
                
        # Remove duplicates? Not strictly necessary for logic, but good for efficiency.
        # We'll just iterate.
        
        # State:
        # selected_wires: list of (vec_int, wire_index_in_final_circuit, cost_to_make)
        # Initially, inputs are available for free (wire index = input index, cost 0).
        # Zero is available for free (wire index = n, cost 0).
        
        # We will maintain a list of "available" wires.
        # available_wires: list of (vec_int, wire_index)
        available_wires = []
        for i in range(n_curr):
            available_wires.append((input_bits[i], i))
        available_wires.append((0, n))
        
        # We need to track which candidates are already in available_wires to avoid re-adding.
        # But the greedy selection might add a candidate that is already there? 
        # No, we only add if it improves cost.
        
        # Helper to calculate total cost given a set of available wires.
        # For each row, cost is min over all available wires w of (cost(w) + cost(row ^ w_vec))
        # BUT cost(w) is the cost to create w. If w is an input, cost is 0.
        # If w is a generated wire, cost is its synthesis cost.
        # We must ensure we don't double count the cost of w if it's used by multiple rows.
        # Total Cost = Sum(cost(w) for w in generated) + Sum(min_cost(row, available))
        # Where min_cost(row, available) = min( cost(row^w) + 1 (for final XOR) for w in available, 
        #                                       cost(row) )
        # Wait, if we use w, we do: result = w XOR (row ^ w). 
        # Cost = 1 (XOR gate) + cost(row ^ w).
        # If we don't use any w (i.e. use empty set), cost = cost(row).
        # Note: w=0 is in available. cost(row^0) = cost(row). 1 + cost(row) is worse than cost(row).
        # So we should exclude w=0 from the "use w" option, or handle it carefully.
        # Actually, if w=0, row^0 = row. Cost = 1 + cost(row). Baseline is cost(row). So we prefer baseline.
        
        def _estimate_total_cost(available_list):
            # available_list: list of (vec_int, wire_index, synth_cost)
            # synth_cost is the cost to create this wire from inputs/other wires.
            # For inputs/zero, synth_cost is 0.
            total_gen_cost = 0
            # We need to know which wires are "generated" vs "primitive".
            # Primitive: inputs (0..n-1) and zero (n).
            # Generated: n+1 and above.
            
            # Sum cost of generating all generated wires
            for vec, widx, scost in available_list:
                if widx >= n + 1:
                    total_gen_cost += scost
                    
            # Sum min cost for each row
            total_row_cost = 0
            for r in rows:
                min_cost = _cost_of_vector(r) # Baseline
                for vec, widx, scost in available_list:
                    # Option: use wire widx
                    # Cost = 1 (XOR) + cost(r ^ vec)
                    # Note: If r^vec is 0, cost is 0. Total 1.
                    # If r^vec is 1 bit, cost is 0. Total 1.
                    # If r is 0, cost is 0. Using w: 1 + cost(w). Worse.
                    new_cost = 1 + _cost_of_vector(r ^ vec)
                    if new_cost < min_cost:
                        min_cost = new_cost
                total_row_cost += min_cost
                
            return total_gen_cost + total_row_cost

        # Initial state: only primitives
        current_available = []
        for i in range(n_curr):
            current_available.append((input_bits[i], i, 0))
        current_available.append((0, n, 0))
        
        current_cost = _estimate_total_cost(current_available)
        
        # Greedy loop
        # We look for a candidate from `candidates` that is not already in `current_available`
        # and reduces the cost.
        
        # To speed up, we can pre-filter candidates that are "expensive" to make.
        # Cost to make a candidate vec is _cost_of_vector(vec).
        
        max_iter = 100
        for _ in range(max_iter):
            if time.time() > time_budget:
                break
                
            best_candidate = None
            best_new_cost = current_cost
            best_wire_index = n + 1 + len([w for w in current_available if w[1] >= n+1])
            
            # Iterate through candidates
            # Optimization: Only consider candidates whose synthesis cost is less than current_cost
            for c_vec in candidates:
                c_cost = _cost_of_vector(c_vec)
                if c_cost >= current_cost:
                    continue
                    
                # Check if c_vec is already available
                # We can check if any available wire has this vector
                # Since we don't add duplicates, we can just check if it's in the list of vectors
                # For efficiency, we can maintain a set of available vectors.
                # Let's use a set for fast lookup.
                
                # Re-estimate cost with this candidate added
                # To avoid O(N) lookup in list, we can just append and estimate, then pop.
                # But _estimate_total_cost is O(m * N_avail).
                # N_avail is small (starts at n+1, grows slowly).
                
                new_available = current_available + [(c_vec, best_wire_index, c_cost)]
                new_cost = _estimate_total_cost(new_available)
                
                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_candidate = (c_vec, c_cost)
                    
            if best_candidate is None:
                break
                
            c_vec, c_cost = best_candidate
            # Add to current available
            current_available.append((c_vec, best_wire_index, c_cost))
            current_cost = best_new_cost
            best_wire_index += 1
            
        # Construct the circuit from current_available
        gates = []
        next_wire = n + 1
        wire_to_index = {} # vec_int -> wire_index (for generated ones)
        
        # First, add gates for all generated wires in current_available
        # We need to synthesize them.
        # The order matters for depth, but for gate count it doesn't.
        # We can synthesize each generated wire independently from inputs?
        # No, that's not how CSE works. The cost calculation assumed we pay c_cost once.
        # But how do we actually build the gate list?
        # The "cost" _cost_of_vector is the number of gates to build it from inputs.
        # So we just run the baseline synthesis for each generated wire, and assign it a new wire index.
        # This is valid because the inputs are available.
        
        # Wait, if we synthesize them independently, we might duplicate gates if two generated wires share sub-expressions.
        # But our greedy cost model didn't account for sharing *between* generated wires.
        # It only accounted for sharing *with* the rows.
        # This is a known limitation of simple greedy CSE. It is still often better than baseline.
        
        # To be safe and correct, we will synthesize each generated wire from scratch (inputs only).
        # This ensures correctness. The cost might be slightly higher than the estimate if sharing exists,
        # but it will still likely beat the pure row-by-row baseline because the row costs drop significantly.
        
        # Map wire index to vector for verification later? No, we verify at the end.
        
        # 1. Generate wires for selected intermediates
        # We need to map the "logical" wire index in current_available to a "physical" gate output index.
        # current_available contains (vec, logical_idx, cost).
        # logical_idx is n+1, n+2, ...
        
        # Let's just create the gates for the intermediates first.
        # We'll create a dict: logical_wire_idx -> physical_wire_idx
        logical_to_physical = {}
        
        for i in range(n_curr):
            logical_to_physical[i] = i
        logical_to_physical[n] = n
        
        physical_next = n + 1
        
        # Sort generated wires by cost? Or just process in order.
        # We need to ensure that if a generated wire depends on another, we have it.
        # But we synthesized them from inputs, so they don't depend on each other.
        # So we can just do baseline synthesis for each.
        
        for vec, l_idx, c_cost in current_available:
            if l_idx < n + 1:
                continue
            # Synthesize vec from inputs
            bits = []
            temp = vec
            bit = 0
            while temp:
                if temp & 1:
                    bits.append(bit)
                temp >>= 1
                bit += 1
                
            if len(bits) == 0:
                logical_to_physical[l_idx] = n
            elif len(bits) == 1:
                logical_to_physical[l_idx] = bits[0]
            else:
                prev = bits[0]
                for b in bits[1:]:
                    gates.append([prev, b])
                    prev = physical_next
                    physical_next += 1
                logical_to_physical[l_idx] = prev
                
        # 2. For each row, choose the best wire to XOR with
        outputs = []
        for r in rows:
            best_cost = _cost_of_vector(r)
            best_wire = None # None means "use baseline synthesis"
            
            for vec, l_idx, c_cost in current_available:
                # Option: use logical wire l_idx
                # Physical wire is logical_to_physical[l_idx]
                p_idx = logical_to_physical[l_idx]
                # Cost: 1 + cost(r ^ vec)
                # But we must ensure p_idx is valid.
                new_cost = 1 + _cost_of_vector(r ^ vec)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_wire = p_idx
                    best_vec = vec # Store vec to know what to XOR
            
            if best_wire is None:
                # Baseline synthesis for this row
                bits = []
                temp = r
                bit = 0
                while temp:
                    if temp & 1:
                        bits.append(bit)
                    temp >>= 1
                    bit += 1
                    
                if len(bits) == 0:
                    outputs.append(n)
                elif len(bits) == 1:
                    outputs.append(bits[0])
                else:
                    prev = bits[0]
                    for b in bits[1:]:
                        gates.append([prev, b])
                        prev = physical_next
                        physical_next += 1
                    outputs.append(prev)
            else:
                # Use shared wire
                # We need to compute (r ^ vec) and XOR with best_wire
                diff = r ^ best_vec
                # Synthesize diff from inputs
                bits = []
                temp = diff
                bit = 0
                while temp:
                    if temp & 1:
                        bits.append(bit)
                    temp >>= 1
                    bit += 1
                    
                if len(bits) == 0:
                    # diff is 0, so output is best_wire
                    outputs.append(best_wire)
                elif len(bits) == 1:
                    # diff is 1 bit, XOR with best_wire
                    gates.append([best_wire, bits[0]])
                    outputs.append(physical_next)
                    physical_next += 1
                else:
                    # Synthesize diff, then XOR
                    prev = bits[0]
                    for b in bits[1:]:
                        gates.append([prev, b])
                        prev = physical_next
                        physical_next += 1
                    # Now prev is the wire for diff
                    gates.append([best_wire, prev])
                    outputs.append(physical_next)
                    physical_next += 1

        circuit = {'gates': gates, 'outputs': outputs}
        return circuit

    # Main Logic
    best_result = _baseline_circuit()
    best_cost = len(best_result['gates'])
    
    # Try Greedy CSE
    if time.time() < deadline - 5.0:
        greedy_circuit = _build_greedy_circuit(A, deadline - 5.0)
        if greedy_circuit is not None:
            if _verify(greedy_circuit):
                d = _compute_depth(greedy_circuit['gates'], greedy_circuit['outputs'])
                if d <= depth_limit:
                    g_cost = len(greedy_circuit['gates'])
                    if g_cost < best_cost:
                        best_cost = g_cost
                        best_result = greedy_circuit
                        
    return best_result
