"""XOR-circuit synthesis via Rank-Revealing Basis Reduction (Backward GA)."""
import time
import numpy as np
import random

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D array-like, m x n binary matrix
        depth_limit: maximum circuit depth (<=16)
    
    Returns:
        dict with 'gates' (list of [wire_a, wire_b] pairs) and 'outputs' (list of wire indices)
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Normalize input
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    A = np.array(matrix, dtype=np.uint8)
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    if not np.any(A):
        return {'gates': [], 'outputs': [n] * m}
    
    # Convert rows to bitsets for efficient manipulation
    orig_row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if A[i, j]:
                bits |= (1 << j)
        orig_row_bits.append(bits)
    
    # Helper: Verify circuit correctness
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

    # Helper: Compute circuit depth
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

    # Helper: Baseline circuit (no sharing)
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

    # Helper: Polish - remove unused gates
    def _polish(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
        if not gates:
            return circuit
            
        num_wires = n + 1 + len(gates)
        necessary = [False] * len(gates)
        wire_to_gate = {}
        for k in range(len(gates)):
            wire_to_gate[n + 1 + k] = k
            
        for w in outputs:
            if w in wire_to_gate:
                necessary[wire_to_gate[w]] = True
                
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
                
        new_outputs = [wire_map.get(w, w) for w in outputs]
        return {'gates': new_gates, 'outputs': new_outputs}

    # Core Strategy: Rank-Revealing Basis Reduction
    def _rank_revealing_basis():
        """
        1. Find a basis for the row space of A using Gaussian elimination.
        2. During elimination, track the linear combination (bitmask over original inputs) 
           that produces each basis vector.
        3. Synthesize each basis vector as a circuit.
        4. Express each original row as a combination of basis vectors and synthesize.
        """
        # Work with numpy arrays for vector ops, but track bitmasks for synthesis
        # Rows are bitmasks over n inputs
        
        # Step 1: Gaussian Elimination to find basis
        # basis[i] = bitmask of original inputs that XOR to form this basis vector
        # We want to minimize the total weight (number of 1s) of the basis vectors
        # Strategy: Process columns from left to right (or by some heuristic)
        
        # Let's try a simple column-order elimination first
        # We maintain a list of "active" rows that haven't been reduced yet
        # and a list of "basis" rows that form the current basis
        
        active = list(orig_row_bits)
        basis = []
        basis_mask = []  # For each basis vector, the bitmask of original inputs
        
        # To improve quality, we can try multiple pivot orders or use a heuristic
        # For now, let's do standard left-to-right, but with a twist:
        # When choosing a pivot for column j, pick the active row with the 
        # minimum weight among those having a 1 in column j.
        
        # Actually, a better heuristic for XOR circuits is to minimize the 
        # "cost" of the basis. Let's try a greedy approach where we pick 
        # the pivot that minimizes the resulting basis vector weight.
        
        # Alternative: Use the rows themselves as the initial basis, then 
        # reduce. But that's just the original matrix.
        
        # Let's stick to standard GE but with a twist: 
        # We'll perform GE on the matrix, but we'll choose pivots to minimize 
        # the weight of the resulting basis vectors.
        
        # Simpler approach for now: 
        # 1. Perform GE to find rank and basis.
        # 2. The basis vectors are the pivot rows after elimination.
        # 3. Synthesize each basis vector.
        # 4. Express original rows in terms of basis.
        
        # Let's implement a version that tries to minimize basis weight.
        # We'll use a priority queue or simple selection.
        
        # For efficiency, let's just do standard GE first and see how it scores.
        # We can optimize pivot selection later.
        
        # Convert to numpy for GE
        A_float = A.astype(np.float64)
        rank = 0
        pivot_cols = []
        basis_vectors = []  # Bitmasks
        
        # We'll do GE on the bitmasks directly for simplicity
        # Since n <= 16, bitmasks are fine
        
        # Process columns 0 to n-1
        for col in range(n):
            if time.time() > deadline - 5.0:
                break
                
            # Find a row with 1 in this column
            pivot_row = -1
            min_weight = float('inf')
            
            for i in range(len(active)):
                if (active[i] >> col) & 1:
                    w = bin(active[i]).count('1')
                    if w < min_weight:
                        min_weight = w
                        pivot_row = i
            
            if pivot_row == -1:
                continue
                
            # Swap pivot_row to top of active
            active[0], active[pivot_row] = active[pivot_row], active[0]
            
            # Eliminate this column from all other active rows
            for i in range(1, len(active)):
                if (active[i] >> col) & 1:
                    active[i] ^= active[0]
            
            # Add to basis
            basis_vectors.append(active[0])
            pivot_cols.append(col)
            rank += 1
            
            # Remove from active
            active = active[1:]
            
            if rank >= m:
                break
        
        if rank == 0:
            # All zeros
            return {'gates': [], 'outputs': [n] * m}
        
        # Now we have rank basis vectors
        # Synthesize each basis vector
        gates = []
        next_wire = n + 1
        basis_wires = []
        
        for bv in basis_vectors:
            indices = [j for j in range(n) if (bv >> j) & 1]
            if len(indices) == 0:
                basis_wires.append(n)
            elif len(indices) == 1:
                basis_wires.append(indices[0])
            else:
                # Synthesize with balanced tree for depth efficiency
                def build_tree(w_list):
                    nonlocal next_wire
                    nonlocal gates
                    if len(w_list) == 1:
                        return w_list[0]
                    mid = len(w_list) // 2
                    left = build_tree(w_list[:mid])
                    right = build_tree(w_list[mid:])
                    gates.append([left, right])
                    w_out = next_wire
                    next_wire += 1
                    return w_out
                
                w = build_tree(indices)
                basis_wires.append(w)
        
        # Now express each original row in terms of the basis
        # We need to find the coefficients c such that row = sum(c_i * basis_i)
        # This is solving a system of linear equations
        
        # Build the basis matrix (rank x n)
        B = np.zeros((rank, n), dtype=np.uint8)
        for i, bv in enumerate(basis_vectors):
            for j in range(n):
                if (bv >> j) & 1:
                    B[i, j] = 1
        
        # Solve B^T * c = row for each row
        # Actually, we need to express each row as a linear combination of basis vectors
        # row = sum_i c_i * basis_i
        # This means [row] = c^T * B
        # So c^T = row * B^{-1} (in the row space)
        
        # Easier: Do GE on the augmented system [B | I] to find the inverse
        # Or just do GE on each row against the basis
        
        # Let's do GE for each row
        outputs = []
        for i in range(m):
            if time.time() > deadline - 2.0:
                # Fallback: use baseline for remaining rows
                bits = orig_row_bits[i]
                indices = [j for j in range(n) if (bits >> j) & 1]
                if len(indices) == 0:
                    outputs.append(n)
                elif len(indices) == 1:
                    outputs.append(indices[0])
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
                continue
            
            # Express row i in terms of basis
            coeffs = [0] * rank
            temp = orig_row_bits[i]
            
            # Use the pivot columns to determine coefficients
            # For each pivot column, if temp has a 1 there, we need the basis vector
            for p, col in enumerate(pivot_cols):
                if (temp >> col) & 1:
                    coeffs[p] = 1
                    temp ^= basis_vectors[p]
            
            # temp should be 0 now
            # Synthesize the output
            active_basis = [p for p in range(rank) if coeffs[p] == 1]
            
            if len(active_basis) == 0:
                outputs.append(n)
            elif len(active_basis) == 1:
                outputs.append(basis_wires[active_basis[0]])
            else:
                # Synthesize combination of basis wires
                w_list = [basis_wires[p] for p in active_basis]
                
                def build_tree2(w_list):
                    nonlocal next_wire
                    nonlocal gates
                    if len(w_list) == 1:
                        return w_list[0]
                    mid = len(w_list) // 2
                    left = build_tree2(w_list[:mid])
                    right = build_tree2(w_list[mid:])
                    gates.append([left, right])
                    w_out = next_wire
                    next_wire += 1
                    return w_out
                
                w = build_tree2(w_list)
                outputs.append(w)
        
        return {'gates': gates, 'outputs': outputs}

    # Helper: Greedy CSE (simplified version of current approach)
    def _greedy_cse():
        expressions = list(orig_row_bits)
        gates = []
        next_wire = n + 1
        wire_depth = {i: 0 for i in range(n + 1)}
        available_wires = list(range(n + 1))
        
        max_iterations = 2000
        for _ in range(max_iterations):
            if time.time() > deadline - 10.0:
                break
                
            best_pair = None
            best_score = -1
            best_w = None
            best_depth = 0
            
            L = len(available_wires)
            if L > 40:  # Limit for efficiency
                break
                
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
                    for k in range(m):
                        e = expressions[k]
                        if (e & (1 << a)) and (e & (1 << b)):
                            saving += 1
                    
                    if saving >= 2:
                        score = saving * 100 - new_depth
                    elif saving == 1:
                        score = 10 - new_depth
                    else:
                        score = -100
                    
                    if score > best_score:
                        best_score = score
                        best_pair = (a, b)
                        best_w = next_wire
                        best_depth = new_depth
            
            if best_pair is None or best_score <= 0:
                break
                
            a, b = best_pair
            w = best_w
            
            gates.append([a, b])
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
        
        # Final synthesis
        final_gates = list(gates)
        outputs = []
        expr_to_wire = {}
        
        for k in range(m):
            e = expressions[k]
            if e in expr_to_wire:
                outputs.append(expr_to_wire[e])
            else:
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
                        def build_tree(w_list):
                            nonlocal next_wire
                            nonlocal final_gates
                            if len(w_list) == 1:
                                return w_list[0]
                            mid = len(w_list) // 2
                            left = build_tree(w_list[:mid])
                            right = build_tree(w_list[mid:])
                            final_gates.append([left, right])
                            w_out = next_wire
                            next_wire += 1
                            return w_out
                        
                        wire_out = build_tree(bits)
                
                expr_to_wire[e] = wire_out
                outputs.append(wire_out)
        
        return {'gates': final_gates, 'outputs': outputs}

    # Run strategies and pick the best
    best_result = None
    best_gate_count = float('inf')
    
    # 1. Baseline
    baseline = _baseline()
    if _verify(baseline):
        best_result = baseline
        best_gate_count = len(baseline['gates'])
    
    # 2. Rank-Revealing Basis (NEW STRATEGY)
    if time.time() < deadline - 20.0:
        try:
            res = _rank_revealing_basis()
            res = _polish(res)
            if _verify(res):
                depth = _compute_depth(res['gates'], res['outputs'])
                if depth <= depth_limit:
                    if len(res['gates']) < best_gate_count:
                        best_result = res
                        best_gate_count = len(res['gates'])
        except Exception:
            pass
    
    # 3. Greedy CSE (fallback/complement)
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
    
    # 4. Fallback
    if best_result is None:
        best_result = {'gates': [], 'outputs': [n] * m}
    
    return best_result

if __name__ == "__main__":
    matrix = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    res = synthesize(matrix, 16)
    print(res)
