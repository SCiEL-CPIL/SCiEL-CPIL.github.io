import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 55.0
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # 1. Parse matrix to integer bitmasks
    # Row i is an integer where bit j is matrix[i][j]
    rows = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j] % 2 == 1:
                mask |= (1 << j)
        rows.append(mask)
        
    # 2. Compute RREF and Transformation Matrix T
    # RREF = T * M
    # We maintain T as a list of bitmasks, initially Identity.
    # T has m rows.
    # We perform Gaussian elimination on 'rows' (M) and apply same ops to 'T'.
    
    # Copy rows to work on
    A = rows[:]
    T = [1 << i for i in range(m)] # Identity matrix rows as bitmasks
    
    pivot_row = 0
    pivot_cols = []
    
    for col in range(n):
        if pivot_row >= m:
            break
            
        # Find a row with a 1 in this column
        pivot = -1
        for r in range(pivot_row, m):
            if (A[r] >> col) & 1:
                pivot = r
                break
        
        if pivot == -1:
            continue
            
        # Swap pivot_row and pivot
        if pivot != pivot_row:
            A[pivot_row], A[pivot] = A[pivot], A[pivot]
            T[pivot_row], T[pivot] = T[pivot], T[pivot]
            
        # Eliminate this column from all other rows
        for r in range(m):
            if r != pivot_row and ((A[r] >> col) & 1):
                A[r] ^= A[pivot_row]
                T[r] ^= T[pivot_row]
                
        pivot_cols.append(col)
        pivot_row += 1
        
    # A is now RREF.
    # Non-zero rows of A are the basis.
    # T is the transformation matrix such that A = T * M.
    # We need M = T^{-1} * A.
    # So we need to compute T_inv.
    
    # 3. Compute T_inv
    # We can do this by performing Gaussian elimination on T to get Identity,
    # and applying the same operations to an Identity matrix.
    # Or, simpler: we know A = T M. We want coefficients c such that M_i = sum c_j A_j.
    # This is equivalent to finding the rows of T^{-1}.
    
    # Let's compute T_inv by inverting T.
    # T is m x m.
    # Augment T with Identity, reduce T to Identity, the right side becomes T_inv.
    
    # T rows are bitmasks of length m.
    # Identity rows are 1 << i.
    
    # If T is singular (should not happen if we only swapped and added, determinant is +/-1), 
    # but T is always invertible as it's a product of elementary row ops (swaps and adds).
    
    T_work = T[:]
    I_work = [1 << i for i in range(m)]
    
    t_pivot_row = 0
    for col in range(m):
        if t_pivot_row >= m:
            break
            
        pivot = -1
        for r in range(t_pivot_row, m):
            if (T_work[r] >> col) & 1:
                pivot = r
                break
                
        if pivot == -1:
            continue
            
        if pivot != t_pivot_row:
            T_work[t_pivot_row], T_work[pivot] = T_work[pivot], T_work[t_pivot_row]
            I_work[t_pivot_row], I_work[pivot] = I_work[pivot], I_work[t_pivot_row]
            
        for r in range(m):
            if r != t_pivot_row and ((T_work[r] >> col) & 1):
                T_work[r] ^= T_work[t_pivot_row]
                I_work[r] ^= I_work[t_pivot_row]
                
        t_pivot_row += 1
        
    # I_work is now T_inv.
    # Row i of I_work tells us which RREF basis rows to XOR to get Original Row i.
    
    # 4. Synthesize Basis (RREF rows)
    # Identify non-zero rows in A (RREF)
    basis_indices = []
    for i in range(m):
        if A[i] != 0:
            basis_indices.append(i)
            
    # Map RREF row index to its wire
    # We need to compute the wire for each A[i] where i in basis_indices.
    
    gates = []
    next_wire = n + 1
    memo = {}
    
    # Initialize memo with input wires and zero
    for j in range(n):
        memo[1 << j] = (j, 0)
    memo[0] = (n, 0)
    
    def add_gate(a, b, d_a, d_b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        depth = max(d_a, d_b) + 1
        return w, depth
        
    def compute_mask(mask):
        if mask in memo:
            return memo[mask]
            
        if time.time() > DEADLINE:
            # Fallback: simple tree
            # This is a safety net, should rarely be hit if RREF logic is fast
            elems = []
            tmp = mask
            while tmp:
                lsb = tmp & -tmp
                elems.append(lsb)
                tmp ^= lsb
            if not elems:
                return (n, 0)
            wires_depths = [(j, 0) for j in range(n) if (mask >> j) & 1]
            # Wait, j is index, not bit.
            wires_depths = []
            for j in range(n):
                if (mask >> j) & 1:
                    wires_depths.append((j, 0))
            while len(wires_depths) > 1:
                new_wires = []
                for i in range(0, len(wires_depths), 2):
                    if i + 1 < len(wires_depths):
                        w1, d1 = wires_depths[i]
                        w2, d2 = wires_depths[i+1]
                        w, d = add_gate(w1, w2, d1, d2)
                        new_wires.append((w, d))
                    else:
                        new_wires.append(wires_depths[i])
                wires_depths = new_wires
            result = wires_depths[0]
            memo[mask] = result
            return result

        # Greedy split: find a submask in memo
        # Since RREF rows are sparse, we can just XOR the inputs directly.
        # But maybe we can share subexpressions?
        # For now, direct XOR of inputs is O(k) gates.
        # Let's try to find a memoized subset to reduce cost.
        
        # Iterate through memo keys that are subsets of mask
        # Optimization: Only check keys with size <= popcount(mask)//2
        # Or just check all? For small n (<=16), it's fine.
        
        best_cost = float('inf')
        best_A = None
        best_B = None
        best_wire_A = None
        best_depth_A = None
        best_wire_B = None
        best_depth_B = None
        best_is_both_memo = False
        
        # Get list of memo keys
        # To avoid modifying dict while iterating, get keys first? 
        # Or just iterate.
        
        for A_mask, (wire_A, depth_A) in memo.items():
            if not (A_mask & mask) == A_mask: # A is not subset of mask
                continue
            if A_mask == mask:
                continue # Already handled by memo check at start
                
            B_mask = mask ^ A_mask
            
            if B_mask in memo:
                wire_B, depth_B = memo[B_mask]
                cost = 1
                if cost < best_cost:
                    best_cost = cost
                    best_A = A_mask
                    best_B = B_mask
                    best_wire_A = wire_A
                    best_depth_A = depth_A
                    best_wire_B = wire_B
                    best_depth_B = depth_B
                    best_is_both_memo = True
            else:
                # Estimate cost of B
                # Popcount - 1 is a rough estimate if no sharing
                cost = 1 + bin(B_mask).count('1') - 1
                if cost < best_cost:
                    best_cost = cost
                    best_A = A_mask
                    best_B = B_mask
                    best_wire_A = wire_A
                    best_depth_A = depth_A
                    best_wire_B = None
                    best_depth_B = None
                    best_is_both_memo = False
                    
        if best_A is not None:
            if best_is_both_memo:
                w, d = add_gate(best_wire_A, best_wire_B, best_depth_A, best_depth_B)
                memo[mask] = (w, d)
                return (w, d)
            else:
                w_B, d_B = compute_mask(best_B)
                w, d = add_gate(best_wire_A, w_B, best_depth_A, d_B)
                memo[mask] = (w, d)
                return (w, d)
                
        # Fallback: XOR all inputs
        wires_depths = []
        for j in range(n):
            if (mask >> j) & 1:
                wires_depths.append((j, 0))
                
        if not wires_depths:
            memo[mask] = (n, 0)
            return (n, 0)
            
        while len(wires_depths) > 1:
            new_wires = []
            for i in range(0, len(wires_depths), 2):
                if i + 1 < len(wires_depths):
                    w1, d1 = wires_depths[i]
                    w2, d2 = wires_depths[i+1]
                    w, d = add_gate(w1, w2, d1, d2)
                    new_wires.append((w, d))
                else:
                    new_wires.append(wires_depths[i])
            wires_depths = new_wires
            
        result = wires_depths[0]
        memo[mask] = result
        return result
        
    # Synthesize basis rows
    basis_wires = {}
    for i in basis_indices:
        w, d = compute_mask(A[i])
        basis_wires[i] = (w, d)
        
    # 5. Synthesize Original Rows
    outputs = [0] * m
    
    for i in range(m):
        # Get the combination of basis rows from T_inv (I_work)
        combo_mask = I_work[i]
        
        # combo_mask is a bitmask of length m.
        # Bit j is 1 if we need to XOR the j-th RREF row.
        # But we only have wires for non-zero RREF rows (basis_indices).
        # If combo_mask has a bit set for a zero RREF row, it's 0, so it doesn't matter.
        
        # We need to XOR the wires corresponding to the set bits in combo_mask.
        # Let's build the mask of basis indices we need to XOR.
        # We can just use the combo_mask itself, but we need to map RREF row index to wire.
        
        # If combo_mask is 0, output is 0.
        if combo_mask == 0:
            outputs[i] = n
            continue
            
        # Check if we have a memoized wire for this exact combination of basis wires?
        # The "wire" for a basis row is just a wire ID.
        # The combination is a set of wire IDs.
        # We can represent the combination by the set of wire IDs.
        # But wire IDs are unique.
        
        # Let's just compute it by XORing the basis wires.
        # To share subexpressions, we can memoize based on the frozenset of wire IDs?
        # Or just compute directly. Given m<=16, the number of basis rows is <=16.
        # The number of original rows is <=16.
        # So we have at most 16 combinations to compute.
        # Each combination involves XORing at most 16 wires.
        # This is very fast.
        
        wires_to_xor = []
        for j in range(m):
            if (combo_mask >> j) & 1:
                if j in basis_wires:
                    wires_to_xor.append(basis_wires[j])
        
        if not wires_to_xor:
            outputs[i] = n
            continue
            
        if len(wires_to_xor) == 1:
            outputs[i] = wires_to_xor[0][0]
            continue
            
        # XOR them
        w, d = wires_to_xor[0]
        for k in range(1, len(wires_to_xor)):
            w2, d2 = wires_to_xor[k]
            w, d = add_gate(w, w2, d, d2)
            
        outputs[i] = w
        
    return {'gates': gates, 'outputs': outputs}
