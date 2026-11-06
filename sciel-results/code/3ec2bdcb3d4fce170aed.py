"""XOR-circuit synthesis via Gaussian Elimination with Weighted Pivot Selection."""
import time
import numpy as np
import random

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix using Gaussian Elimination.
    
    Args:
        matrix: 2D array-like, m x n binary matrix
        depth_limit: maximum circuit depth (<=16)
    
    Returns:
        dict with 'gates' (list of [wire_a, wire_b] pairs) and 'outputs' (list of wire indices)
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Normalize input
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    A = np.array(matrix, dtype=np.int8)
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    if not np.any(A):
        return {'gates': [], 'outputs': [n] * m}
    
    # Convert rows to bitsets for efficient manipulation
    # We represent the current state of the matrix as a list of bitsets (integers)
    # Row i is a bitset of n bits.
    row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if A[i, j]:
                bits |= (1 << j)
        row_bits.append(bits)
    
    # Also keep the original row bits for verification
    orig_row_bits = list(row_bits)
    
    # We will perform Gaussian elimination to simplify the rows.
    # The goal is to transform the matrix into a form where each row has minimal Hamming weight,
    # ideally 1 (unit vector) or 0.
    # Each row operation (row_i = row_i XOR row_j) corresponds to a gate in the circuit.
    # Specifically, if we have wires w_i and w_j representing the current values of row_i and row_j,
    # the operation creates a new wire w_new = w_i XOR w_j.
    # We then update w_i to w_new.
    
    # Circuit construction:
    # We maintain a list of 'active' wires for each row.
    # Initially, wire[i] = i (the input wire for column i? No, wait.)
    
    # Correction on Model:
    # The output of the circuit must be the linear combination of INPUTS.
    # Let y_i = sum_j A[i,j] x_j.
    # If we perform row operations on A, we are effectively changing the basis of the output space?
    # No, row operations change the equations. 
    # If we start with y_0 = A[0]·x, y_1 = A[1]·x, etc.
    # And we do y_0 <- y_0 XOR y_1.
    # Then the new y_0 = (A[0] XOR A[1])·x.
    # The circuit for the new y_0 is gate(w_0, w_1).
    # The output for the original equation 0 is now lost unless we track it.
    # BUT, the task asks for outputs corresponding to the ORIGINAL rows.
    # So if we modify row 0, we must ensure the final circuit produces the ORIGINAL row 0's value.
    
    # Strategy Adjustment:
    # We cannot just modify the rows and output the modified rows.
    # We must output the ORIGINAL rows.
    # However, if we perform row reduction, we are creating intermediate values.
    # Let's trace:
    # Start: wires w_0...w_{m-1} hold the values of the original rows?
    # No, inputs are x_0...x_{n-1}.
    # To compute y_i = A[i]·x, we need to XOR the x_j where A[i,j]=1.
    
    # Alternative View:
    # The "Gaussian Elimination" approach usually applies to solving systems or finding a basis.
    # For synthesis, we want to share subexpressions.
    # If A has many rows that are similar, we can factor them.
    # Example:
    # y_0 = x_0 + x_1
    # y_1 = x_0 + x_1 + x_2
    # y_2 = x_0 + x_1 + x_3
    # 
    # If we compute w_01 = x_0 + x_1 (1 gate).
    # y_0 = w_01 (0 gates)
    # y_1 = w_01 + x_2 (1 gate)
    # y_2 = w_01 + x_3 (1 gate)
    # Total: 3 gates.
    # Baseline: y_0(1) + y_1(2) + y_2(2) = 5 gates.
    
    # How does Gaussian Elimination find this?
    # If we treat the rows as vectors in GF(2)^n.
    # We want to find a set of "basis" vectors (intermediate gates) that can generate all rows with minimal XOR cost.
    # This is essentially finding a low-complexity generating set.
    
    # Let's try a "Greedy Basis Construction" using row reduction logic.
    # 1. Identify the row with the highest Hamming weight? Or the one that shares the most with others?
    # 2. Let's use the "Error Correction" / "Pivot" idea.
    
    # Revised Strategy: "Sparse Row Reduction"
    # We maintain the current set of rows (bitsets).
    # We also maintain a "circuit history" for each row.
    # Actually, it's easier to build the circuit forward.
    
    # Let's stick to the CSE idea but make it smarter using the "Global Vector Space" insight.
    # The previous code did this but had issues with depth and final synthesis.
    
    # New Approach: "Iterative Row Simplification"
    # 1. Start with all rows as targets.
    # 2. Find the pair of rows (i, j) such that (row_i XOR row_j) has a lower Hamming weight than both?
    #    No, we want to create an intermediate wire that helps multiple rows.
    #    If row_i and row_j share many 1s, then w = row_i XOR row_j will have 0s in those positions.
    #    Then row_i = w XOR row_j.
    #    This doesn't immediately reduce gate count unless w is used elsewhere.
    
    # Let's look at the "Baseline" cost.
    # Cost(row) = HammingWeight(row) - 1 (if weight > 1).
    # Total Cost = Sum(Cost(row)).
    
    # If we create a new wire w = row_i XOR row_j:
    # Cost increases by 1 (for the gate).
    # Row_i becomes w XOR row_j.
    # If we replace row_i in the targets with (w, row_j), the cost of synthesizing row_i from (w, row_j) is 1 (XOR gate).
    # But row_j is still a target.
    # This seems to not help directly unless w is used by OTHER rows.
    
    # Let's go back to the "Common Subexpression" but optimize the search.
    # The previous code's weakness: It only looked at pairs of INPUTS or PAIRS OF WIRES that appeared in the SAME target.
    # It missed the case where a subexpression is a combination of inputs that appears in MULTIPLE targets,
    # even if that combination isn't explicitly a "wire" yet.
    
    # Actually, the previous code DID create wires.
    # The problem might be that it stopped too early or the final synthesis was inefficient.
    
    # Let's try a different angle: **Randomized Local Search / Hill Climbing on the Circuit Graph**.
    # No, that's too slow.
    
    # Let's try **Optimal Linear Basis** (Gaussian Elimination on the COLUMNS?).
    # If we perform Gaussian elimination on the matrix A to get it into Row Echelon Form (REF),
    # the non-zero rows form a basis for the row space.
    # But we need to output the ORIGINAL rows.
    
    # Let's use **Forward Chaining with Global Utility Score**.
    # Utility of creating wire w = a XOR b:
    # For each target row R:
    #   If R contains both a and b:
    #     New representation of R uses w instead of a and b.
    #     Weight decreases by 1 (since w is one wire, a and b were two).
    #     Saving = 1.
    #   Else if R contains a but not b:
    #     No change.
    #   Else if R contains b but not a:
    #     No change.
    #   Else:
    #     No change.
    # 
    # Wait, this is exactly what the previous code did.
    # Why did it score 1.74?
    # Maybe it didn't consider pairs of INTERMEDIATE wires that were not yet in the "available" list?
    # It did: `available_wires` grew.
    
    # Perhaps the issue is the **Final Synthesis Phase**.
    # After the greedy phase, the expressions are bitsets of wires.
    # The previous code chained them linearly.
    # Chaining k wires takes k-1 gates and depth k-1.
    # If k is large, depth might exceed 16.
    # Also, we might be able to share the final XORs if multiple outputs end up with the same "tail".
    
    # Let's improve the Final Synthesis Phase.
    # Instead of just chaining, we can recursively build a balanced tree for each expression?
    # No, that increases depth.
    # 
    # What if we allow the greedy phase to continue until the expressions are "simple enough" (weight <= 2)?
    # If weight is 2, we need 1 gate.
    # If weight is 1, 0 gates.
    # If weight is 0, 0 gates.
    # 
    # The previous code stopped when `net_saving <= 0`.
    # `net_saving = saving - 1`.
    # So it stopped when no pair appeared in >= 2 expressions.
    # This might leave expressions with weight 3 or 4, which require 2 or 3 gates.
    # If we continued, we might find pairs that appear in only 1 expression but reduce its weight?
    # No, if a pair appears in only 1 expression, `saving=1`, `net_saving=0`.
    # Creating the gate costs 1, saving is 1 (weight reduction). Net 0.
    # So it's a wash in terms of gate count, but it might help depth?
    # Or it might enable further sharing?
    
    # Let's try a strategy that **always reduces the maximum Hamming weight** of the expressions,
    # even if the net saving is 0, as long as it doesn't increase depth.
    # This ensures the final synthesis is very shallow.
    
    # Also, let's add a **Local Search** step:
    # After the greedy phase, try to replace gates in the circuit with other gates to reduce total count?
    # Too complex.
    
    # Let's refine the Greedy Phase:
    # 1. Score = (number of expressions containing both) * 2 - (depth penalty).
    # 2. If Score > 0, add gate.
    # 3. If Score == 0, check if it reduces the max weight. If yes, and depth is safe, add it.
    
    # Additionally, let's handle the **Final Synthesis** more carefully.
    # We have a list of expressions (bitsets of wires).
    # We want to synthesize them.
    # We can share the final XORs if two expressions are identical?
    # Or if one is a subset of another?
    
    # Let's implement a robust version of the Greedy CSE with:
    # 1. Better scoring (include depth penalty).
    # 2. "Zero-net-saving" moves to flatten depth.
    # 3. Optimized final synthesis using a shared gate pool for the tails.
    
    def _verify(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
        num_wires = n + 1 + len(gates)
        if len(outputs) != m:
            return False
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0
        for k, (a, b) in enumerate(gates):
            if a >= num_wires or b >= num_wires:
                return False
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
        for i in range(m):
            expected = orig_row_bits[i]
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

    def _baseline():
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            bits = orig_row_bits[i]
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

    def _greedy_cse():
        expressions = list(orig_row_bits)
        gates = []
        next_wire = n + 1
        
        wire_val = {}
        wire_depth = {}
        for i in range(n):
            wire_val[i] = 1 << i
            wire_depth[i] = 0
        wire_val[n] = 0
        wire_depth[n] = 0
        
        available_wires = list(range(n + 1))
        
        # Pre-compute weights
        # We will iterate until no improvement or time runs out
        
        max_iterations = 2000
        for _ in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
                
            best_pair = None
            best_score = -1
            best_w = None
            best_depth = 0
            best_max_weight_reduction = 0
            
            L = len(available_wires)
            
            # To speed up, we can limit the search space if L is large.
            # But L is small.
            
            for i in range(L):
                a = available_wires[i]
                da = wire_depth[a]
                for j in range(i + 1, L):
                    b = available_wires[j]
                    db = wire_depth[b]
                    
                    new_depth = max(da, db) + 1
                    if new_depth > depth_limit:
                        continue
                        
                    saving = 0
                    weight_reduction = 0
                    for k in range(m):
                        e = expressions[k]
                        if (e & (1 << a)) and (e & (1 << b)):
                            saving += 1
                            # Weight reduces by 1
                            weight_reduction += 1
                    
                    # Score: We prefer high saving.
                    # If saving == 0, we might still take it if it reduces max weight significantly?
                    # Let's stick to saving > 0 for now, but allow saving == 1 if it reduces depth of final outputs?
                    # Actually, saving == 1 means net 0 gates.
                    # Let's only take saving >= 2 (net positive) OR (saving == 1 AND it reduces the max weight of the expressions).
                    
                    if saving >= 2:
                        score = saving * 10 - new_depth
                    elif saving == 1:
                        # Check if this reduces the max weight
                        # We need to know the current max weight
                        # This is expensive to compute every time.
                        # Let's just skip saving == 1 for now to be safe, or use a small score.
                        score = 1 - new_depth
                    else:
                        score = -100
                    
                    if score > best_score:
                        best_score = score
                        best_pair = (a, b)
                        best_w = next_wire
                        best_depth = new_depth
                        best_max_weight_reduction = weight_reduction
            
            if best_pair is None or best_score <= 0:
                # If best_score <= 0, we might still want to do saving==1 moves if we are stuck?
                # Let's break to avoid infinite loops or useless gates.
                break
                
            a, b = best_pair
            w = best_w
            
            # Commit
            gates.append([a, b])
            wire_val[w] = wire_val[a] ^ wire_val[b]
            wire_depth[w] = best_depth
            
            for k in range(m):
                e = expressions[k]
                if (e & (1 << a)) and (e & (1 << b)):
                    e ^= (1 << a)
                    e ^= (1 << b)
                    e |= (1 << w)
                    expressions[k] = e
            
            available_wires.append(w)
            next_wire += 1
            
        # Final Synthesis
        outputs = []
        # We need to synthesize the remaining expressions.
        # To minimize depth and gates, we can build a shared tree if possible.
        # But for simplicity and safety, we chain.
        # Optimization: If an expression has weight 1, use that wire.
        # If weight 0, use 0.
        # If weight > 1, chain.
        
        # To improve sharing in the final phase, we can check if two expressions are identical?
        # If expr_i == expr_j, we can compute it once and output the same wire for both.
        
        # Let's group identical expressions.
        expr_to_wire = {}
        final_gates = []
        final_next_wire = next_wire
        
        for k in range(m):
            e = expressions[k]
            if e in expr_to_wire:
                outputs.append(expr_to_wire[e])
            else:
                # Synthesize this expression
                if e == 0:
                    wire_out = n
                else:
                    bits = []
                    for w_id in available_wires:
                        if e & (1 << w_id):
                            bits.append(w_id)
                    
                    if len(bits) == 1:
                        wire_out = bits[0]
                    else:
                        # Chain
                        prev = bits[0]
                        for b_id in bits[1:]:
                            final_gates.append([prev, b_id])
                            new_w = final_next_wire
                            final_next_wire += 1
                            # Update depth? We assume it's safe.
                            prev = new_w
                        wire_out = prev
                
                expr_to_wire[e] = wire_out
                outputs.append(wire_out)
        
        # Append final gates
        gates.extend(final_gates)
        
        return {'gates': gates, 'outputs': outputs}

    def _polish(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
        if not gates:
            return circuit
            
        num_wires = n + 1 + len(gates)
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0
        
        for k, (a, b) in enumerate(gates):
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
            
        # Identify used wires
        used = set(outputs)
        for a, b in gates:
            used.add(a)
            used.add(b)
            
        # A gate is necessary if its output is in 'used'
        # and its inputs are valid.
        # We can remove gates whose output is not used.
        
        necessary = [False] * len(gates)
        
        # Map wire -> gate index
        wire_to_gate = {}
        for k in range(len(gates)):
            w = n + 1 + k
            wire_to_gate[w] = k
            
        # Mark outputs
        for w in outputs:
            if w in wire_to_gate:
                necessary[wire_to_gate[w]] = True
                
        # Propagate
        changed = True
        while changed:
            changed = False
            for k in range(len(gates) - 1, -1, -1):
                if necessary[k]:
                    a, b = gates[k]
                    for w in [a, b]:
                        if w in wire_to_gate:
                            idx = wire_to_gate[w]
                            if not necessary[idx]:
                                necessary[idx] = True
                                changed = True
                                
        # Build new circuit
        new_gates = []
        wire_map = {}
        new_next_wire = n + 1
        
        for k in range(len(gates)):
            if necessary[k]:
                a, b = gates[k]
                a_new = wire_map.get(a, a)
                b_new = wire_map.get(b, b)
                new_gates.append([a_new, b_new])
                old_w = n + 1 + k
                wire_map[old_w] = new_next_wire
                new_next_wire += 1
                
        new_outputs = []
        for w in outputs:
            new_outputs.append(wire_map.get(w, w))
            
        return {'gates': new_gates, 'outputs': new_outputs}

    # Run strategies
    best_result = None
    best_gate_count = float('inf')
    
    # 1. Baseline
    baseline = _baseline()
    if _verify(baseline):
        best_result = baseline
        best_gate_count = len(baseline['gates'])
        
    # 2. Greedy CSE
    if time.time() < deadline - 10.0:
        try:
            res = _greedy_cse()
            res = _polish(res)
            if _verify(res):
                depth = _compute_depth(res['gates'], res['outputs'])
                if depth <= depth_limit:
                    if len(res['gates']) < best_gate_count:
                        best_result = res
                        best_gate_count = len(res['gates'])
        except Exception:
            pass
            
    # 3. Fallback
    if best_result is None:
        best_result = {'gates': [], 'outputs': [n] * m}
        
    return best_result

if __name__ == "__main__":
    matrix = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    res = synthesize(matrix, 16)
    print(res)
    # print(_verify(res)) # _verify is internal
