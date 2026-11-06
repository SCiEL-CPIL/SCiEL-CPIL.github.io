import time
import random
import sys

def synthesize(matrix, depth_limit):
    start_time = time.time()
    # Safety margin: The task says 60s kill, but instructions say 90s wall-clock 
    # and to keep a guard. I will use 55s to be safe against overhead.
    DEADLINE = start_time + 55.0 

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Parse matrix to row masks (integers for fast bitwise ops)
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        row_masks.append(mask)

    def elapsed():
        return time.time() - start_time

    def build_circuit(basis_masks, row_reps, n):
        """
        Builds the circuit from a basis and row representations.
        Uses global memoization for XOR sub-expressions.
        """
        # next_wire starts at n+1 (0..n-1 are inputs, n is zero)
        next_wire = n + 1
        gates = []
        memo = {}
        
        # Initialize memo with inputs and zero
        for j in range(n):
            memo[1 << j] = j
        memo[0] = n
        
        def add_gate(a, b):
            nonlocal next_wire
            # Optimization: if a == b, result is 0
            if a == b:
                return n
            gates.append([a, b])
            w = next_wire
            next_wire += 1
            return w
        
        def compute_xor(mask):
            """Recursively compute wire for a mask using memoization."""
            if mask in memo:
                return memo[mask]
            if mask == 0:
                return n
            if mask & (mask - 1) == 0:
                # Single bit
                return mask.bit_length() - 1
            
            # Heuristic: Split mask into two parts to minimize depth or gate count.
            # A simple balanced split is usually good for gate count if we memoize.
            # However, for gate count, we want to reuse sub-expressions.
            # Let's try splitting by the lowest set bit vs the rest? 
            # Or just a balanced split by bit index.
            
            bits = []
            mm = mask
            while mm:
                b = mm & (-mm)
                bits.append(b.bit_length() - 1)
                mm ^= b
            
            # Balanced split
            mid = len(bits) // 2
            mask1 = 0
            for b in bits[:mid]:
                mask1 |= (1 << b)
            mask2 = mask ^ mask1
            
            w1 = compute_xor(mask1)
            w2 = compute_xor(mask2)
            
            if w1 == w2:
                result = n
            else:
                result = add_gate(w1, w2)
            
            memo[mask] = result
            return result

        # Build wires for basis vectors
        basis_wires = []
        for b_mask in basis_masks:
            w = compute_xor(b_mask)
            basis_wires.append(w)
        
        # Build outputs
        outputs = []
        for i in range(m):
            rep = row_reps[i]
            if not rep:
                outputs.append(n)
            elif len(rep) == 1:
                outputs.append(basis_wires[rep[0]])
            else:
                # Combine basis wires
                wires = [basis_wires[idx] for idx in rep]
                # XOR all wires together
                # We can just chain them or use a tree. Tree is better for depth, 
                # but gate count is the same (k-1 gates for k wires).
                # However, memoization might help if we've seen this combination before?
                # The mask for this output is the original row mask.
                target_mask = 0
                for idx in rep:
                    target_mask ^= basis_masks[idx]
                
                # If we already computed this mask as part of a basis or previous output,
                # we can reuse it.
                if target_mask in memo:
                    outputs.append(memo[target_mask])
                else:
                    # Compute it
                    # We can just call compute_xor(target_mask) which will use the 
                    # existing memo entries for the basis parts if they match, 
                    # but compute_xor splits by bits, not by basis vectors.
                    # So it's safer to just XOR the wires directly.
                    current_wire = wires[0]
                    for w in wires[1:]:
                        if current_wire != w:
                            current_wire = add_gate(current_wire, w)
                        else:
                            current_wire = n
                    outputs.append(current_wire)
                    memo[target_mask] = current_wire
        
        return {'gates': gates, 'outputs': outputs}

    def get_cost(basis_masks, row_reps):
        """
        Estimate the cost of a given basis/reps configuration.
        We return the actual gate count by building the circuit.
        """
        result = build_circuit(basis_masks, row_reps, n)
        return len(result['gates']), result

    def gaussian_elimination(col_order):
        """
        Performs Gaussian elimination with a specific column order for pivots.
        Returns (basis_masks, row_reps).
        """
        # Map column index to priority
        col_priority = {col: i for i, col in enumerate(col_order)}
        
        # basis_dict: pivot_bit -> mask
        basis_dict = {}
        basis_list = []
        basis_mask_to_idx = {}
        row_reps = []
        
        # We need to track which original row corresponds to which basis vector?
        # No, row_reps tracks which basis vectors compose each original row.
        
        # Standard RREF-like elimination
        # We iterate through rows. For each row, we reduce it using the current basis.
        
        for i in range(m):
            mask = row_masks[i]
            if mask == 0:
                row_reps.append([])
                continue
            
            cm = mask
            rep = []
            
            # Reduce using existing basis
            # We want to eliminate bits according to the priority order?
            # Actually, standard Gaussian elimination picks a pivot for a row.
            # Here, the "pivot" is determined by the first non-zero column in the 
            # preferred order.
            
            while cm:
                # Find the set bit with the lowest priority (most preferred)
                best_bit = -1
                best_prio = float('inf')
                mm = cm
                while mm:
                    b = mm & (-mm)
                    bit = b.bit_length() - 1
                    prio = col_priority.get(bit, float('inf'))
                    if prio < best_prio:
                        best_prio = prio
                        best_bit = bit
                    mm ^= b
                
                if best_bit in basis_dict:
                    b_mask = basis_dict[best_bit]
                    idx = basis_mask_to_idx[b_mask]
                    rep.append(idx)
                    cm ^= b_mask
                else:
                    break
            
            if cm:
                # This row is independent. Add it to basis.
                # Pivot is best_bit.
                b_idx = len(basis_list)
                basis_dict[best_bit] = cm
                basis_list.append(cm)
                basis_mask_to_idx[cm] = b_idx
                rep.append(b_idx)
            
            row_reps.append(rep)
        
        return basis_list, row_reps

    # Generate candidate column orderings
    
    # 1. Natural order
    candidates = [list(range(n))]
    
    # 2. Sorted by column weight (ascending)
    col_weights = [0] * n
    for mask in row_masks:
        mm = mask
        while mm:
            b = mm & (-mm)
            col_weights[b.bit_length() - 1] += 1
            mm ^= b
    
    candidates.append(sorted(range(n), key=lambda j: col_weights[j]))
    candidates.append(sorted(range(n), key=lambda j: -col_weights[j]))
    
    # 3. Greedy Heuristic:
    # At each step, pick the column that maximizes the reduction in total non-zeros
    # if we were to eliminate using that column? 
    # This is complex to simulate exactly. 
    # Instead, let's try a "randomized greedy" where we shuffle and pick the best 
    # few from a random set.
    
    # 4. Random Permutations
    rng = random.Random(42)
    # Number of random trials depends on time.
    # For n=16, 100 trials is very fast.
    num_random = 50
    for _ in range(num_random):
        perm = list(range(n))
        rng.shuffle(perm)
        candidates.append(perm)
        
    # Also try "Local Search" style: Start with best so far, swap adjacent, etc.
    # But first, let's evaluate the static ones.
    
    best_result = None
    best_gate_count = float('inf')
    
    for order in candidates:
        if elapsed() > DEADLINE - 2.0:
            break
            
        basis_list, row_reps = gaussian_elimination(order)
        
        # Quick check: if no basis, gate count is 0
        if not basis_list:
            best_gate_count = 0
            best_result = {'gates': [], 'outputs': [n] * m}
            break
            
        gate_count, result = get_cost(basis_list, row_reps, n)
        
        if gate_count < best_gate_count:
            best_gate_count = gate_count
            best_result = result
            
        if best_gate_count == 0:
            break

    # If we have time left, try a more intensive local search around the best permutation
    if best_result is not None and elapsed() < DEADLINE - 5.0:
        # We need to recover the best order to do local search.
        # Let's re-run the best candidate to get the order, or store it.
        # For simplicity, let's just add more randoms if we are fast.
        # Or, implement a simple local search on the best order found.
        # We didn't store the best order. Let's find it again.
        best_order = None
        for order in candidates:
            basis_list, row_reps = gaussian_elimination(order)
            if not basis_list:
                continue
            gate_count, _ = get_cost(basis_list, row_reps, n)
            if gate_count == best_gate_count:
                best_order = order
                break
                
        if best_order:
            # Local search: swap random pairs in best_order
            current_best_order = best_order
            current_best_count = best_gate_count
            no_improvement_count = 0
            max_iters = 100
            
            while no_improvement_count < 10 and max_iters > 0 and elapsed() < DEADLINE - 2.0:
                max_iters -= 1
                i = rng.randint(0, n - 2)
                j = rng.randint(0, n - 2)
                if i == j:
                    continue
                new_order = current_best_order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                
                basis_list, row_reps = gaussian_elimination(new_order)
                if not basis_list:
                    continue
                gate_count, result = get_cost(basis_list, row_reps, n)
                
                if gate_count < current_best_count:
                    current_best_count = gate_count
                    current_best_order = new_order
                    best_result = result
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    
    if best_result is None:
        # Fallback
        basis_list, row_reps = gaussian_elimination(list(range(n)))
        best_result = build_circuit(basis_list, row_reps, n)

    return best_result
