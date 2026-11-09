import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert matrix rows to integer bitmasks
    rows = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i][j]:
                val |= (1 << j)
        rows.append(val)
    
    # 1. Find a linear basis for the rows
    # We want to minimize the number of "independent" computations.
    # Gaussian Elimination to find basis and representation.
    
    # basis[i] holds the bitmask for the i-th pivot bit
    # row_repr[i] holds the bitmask of basis indices used to form row i
    # Actually, let's just find the basis vectors and which original rows map to which basis combos.
    
    # Standard Gaussian Elimination over GF(2)
    # We will maintain a list of basis vectors.
    basis = [0] * n  # basis[j] is a vector with leading bit at j
    row_to_basis = [0] * m # Bitmask of basis indices used for each original row
    
    # First pass: Build basis from rows
    # We need to track which combination of original rows forms each basis vector?
    # Or rather, express each original row in terms of the basis.
    
    # Let's use a different approach:
    # 1. Identify independent rows.
    # 2. Synthesize those.
    # 3. Express dependent rows as XORs of the independent ones.
    
    independent_indices = []
    # map from pivot bit to index in independent_indices
    pivot_to_idx = {}
    
    for i in range(m):
        r = rows[i]
        # Try to eliminate r using existing basis
        # We will store the basis vectors in a list `basis_vecs`
        # and `basis_repr` which tells us how to form each basis vector from original inputs?
        # No, we just need the values.
        
        # To express row i in terms of the basis, we need to know the "history" of eliminations.
        
        # Let's do standard GE keeping track of the linear combination.
        # Let `current_rows` be a list of tuples (value, combination_mask)
        # combination_mask is a bitmask over the original rows? No, that's m bits.
        # We just need to express the final basis vectors in terms of the *input wires*?
        # No, the basis vectors are just numbers.
        
        # Simpler:
        # 1. Find the set of linearly independent rows. Let's call them `basis_rows`.
        # 2. For each original row, determine which `basis_rows` XOR to it.
        
    # Step 1: Find basis rows
    basis_rows = []
    # pivot_to_basis_row_idx: dict from pivot bit position to index in basis_rows
    pivot_map = {}
    
    for i in range(m):
        r = rows[i]
        # Eliminate r using current basis_rows
        # Sort basis_rows by highest set bit?
        # Let's just iterate through basis_rows and XOR if it reduces the highest bit.
        
        # To make this efficient, we maintain basis in row-echelon form.
        # For each bit from high to low:
        for bit in range(n - 1, -1, -1):
            if not (r & (1 << bit)):
                continue
            if bit in pivot_map:
                # XOR with the basis row that has this pivot
                basis_idx = pivot_map[bit]
                r ^= basis_rows[basis_idx]
            else:
                # This is a new pivot
                basis_rows.append(r)
                pivot_map[bit] = len(basis_rows) - 1
                break
        # If r becomes 0, it's dependent.
        
    # Now we have `basis_rows`.
    # We need to synthesize wires for each vector in `basis_rows`.
    
    # Step 2: Synthesize basis rows using Global Greedy
    targets = set(basis_rows)
    # Remove 0 if present (it's always available)
    targets.discard(0)
    
    available = {}
    # Initialize available with inputs and zero
    available[0] = n # wire n is zero
    for j in range(n):
        available[1 << j] = j # wire j is input j
        
    gates = []
    next_wire = n + 1
    
    # Map from basis row value to its wire index (once synthesized)
    basis_val_to_wire = {}
    
    # Greedy synthesis loop
    # We need to be careful with time.
    # Max gates is roughly len(targets) * n.
    
    while targets:
        if time.time() > deadline:
            break
            
        # Check if any target is already available
        remaining_targets = set()
        for t in targets:
            if t in available:
                basis_val_to_wire[t] = available[t]
            else:
                remaining_targets.add(t)
        
        if not remaining_targets:
            break
            
        # Find the best next gate
        best_w = None
        best_score = -1
        best_pair = None
        
        # Iterate over all pairs in available
        # To speed up, we can limit the search space or use heuristics.
        # For n=16, available size grows.
        # Let's sample or iterate if small.
        avail_list = list(available.keys())
        
        # Optimization: Only consider pairs that produce a value not in available
        # and that has a "high score"
        
        for i in range(len(avail_list)):
            u = avail_list[i]
            for j in range(i + 1, len(avail_list)):
                v = avail_list[j]
                w = u ^ v
                if w in available:
                    continue
                
                # Score w
                score = 0
                if w in remaining_targets:
                    score += 10000 # Direct hit
                else:
                    # Check how many targets t can be formed by t ^ w if (t ^ w) in available
                    # i.e., t = available_val ^ w
                    # So for each t in remaining_targets, check if (t ^ w) in available
                    for t in remaining_targets:
                        if (t ^ w) in available:
                            score += 100
                            
                # Tie breaker: Prefer lower Hamming weight (simpler wire)
                if score > 0:
                    score -= bin(w).count('1')
                
                if score > best_score:
                    best_score = score
                    best_w = w
                    best_pair = (u, v)
                    
        if best_w is None or best_score <= 0:
            # No good pair found. Fallback: Build a target directly from inputs
            # Pick the target with lowest Hamming weight
            target_to_build = min(remaining_targets, key=lambda x: bin(x).count('1'))
            
            # Build target_to_build from inputs
            # Find a wire in available that is a subset of target_to_build?
            # Or just XOR inputs.
            
            # Let's try to find an available wire that shares bits.
            # If none, use pure inputs.
            
            # Construct chain
            current_val = 0
            current_wire = n # start with 0
            
            # We want to XOR inputs to get target_to_build
            # Iterate through bits of target_to_build
            for bit in range(n):
                if target_to_build & (1 << bit):
                    # XOR input bit
                    gates.append([current_wire, bit])
                    current_wire = next_wire
                    next_wire += 1
                    current_val ^= (1 << bit)
                    available[current_val] = current_wire
                    
            basis_val_to_wire[target_to_build] = current_wire
            remaining_targets.discard(target_to_build)
            continue
            
        # Execute best gate
        u, v = best_pair
        gates.append([available[u], available[v]])
        available[best_w] = next_wire
        next_wire += 1
        
    # Step 3: Express all original rows in terms of basis_rows
    # We need to know which basis_rows XOR to form each original row.
    # We can re-run Gaussian Elimination on the original rows using the basis_rows we found.
    
    row_wires = [None] * m
    
    # For each original row, express it as a combination of basis_rows
    # We do this by trying to eliminate the row using basis_rows.
    # We maintain a "combination" mask over the indices of basis_rows.
    
    for i in range(m):
        r = rows[i]
        if r == 0:
            row_wires[i] = n
            continue
            
        if r in basis_val_to_wire:
            row_wires[i] = basis_val_to_wire[r]
            continue
            
        # Express r in terms of basis_rows
        # We need to find indices k such that XOR(basis_rows[k] for k in indices) == r
        # Since basis_rows is in row-echelon form (pivots are unique), we can do this greedily.
        
        combo_mask = 0
        temp_r = r
        
        # Iterate from high bit to low bit
        for bit in range(n - 1, -1, -1):
            if not (temp_r & (1 << bit)):
                continue
            if bit in pivot_map:
                basis_idx = pivot_map[bit]
                # XOR with this basis row
                temp_r ^= basis_rows[basis_idx]
                combo_mask ^= (1 << basis_idx)
            else:
                # This shouldn't happen if basis_rows spans the row space
                # But if r was 0, we handled it.
                # If r is not in span, something is wrong.
                # But we built basis from rows, so it must be in span.
                pass
                
        # Now temp_r should be 0.
        # We have a combination of basis rows.
        # We need to create a wire for this combination.
        
        if combo_mask == 0:
            row_wires[i] = n
        else:
            # Find the indices of set bits in combo_mask
            wires_to_xor = []
            for k in range(len(basis_rows)):
                if combo_mask & (1 << k):
                    wires_to_xor.append(basis_val_to_wire[basis_rows[k]])
                    
            if len(wires_to_xor) == 1:
                row_wires[i] = wires_to_xor[0]
            else:
                # Chain XOR
                current_wire = wires_to_xor[0]
                for w in wires_to_xor[1:]:
                    gates.append([current_wire, w])
                    current_wire = next_wire
                    next_wire += 1
                row_wires[i] = current_wire
                
    # Check if any row_wire is None (should not happen)
    for i in range(m):
        if row_wires[i] is None:
            row_wires[i] = n

    return {'gates': gates, 'outputs': row_wires}
