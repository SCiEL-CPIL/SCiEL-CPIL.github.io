"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 80.0  # safety margin before 90s kill
    
    mat = np.array(matrix, dtype=np.uint8)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    n_rows, n_cols = mat.shape
    n = n_cols
    zero_wire = n  # constant-zero wire
    
    # For each row, get the set of input indices where value is 1
    row_sets = []
    for i in range(n_rows):
        cols = np.nonzero(mat[i])[0]
        row_sets.append(frozenset(cols.tolist()))
    
    # Group rows by their input set
    set_to_rows = defaultdict(list)
    for i, s in enumerate(row_sets):
        set_to_rows[s].append(i)
    
    # Strategy: 
    # 1. Compute a basis for the row space using Gaussian elimination over GF(2).
    # 2. Express each unique row set in terms of basis vectors.
    # 3. Build balanced XOR trees for each basis vector, caching all intermediate subsets.
    # 4. For each output row, XOR the wires corresponding to its basis components.
    #    Use the cache to reuse intermediate results where possible.
    
    # Step 1: Gaussian elimination to find basis
    # We represent vectors as frozensets of column indices for consistency
    basis_info = []  # List of (pivot_index, frozenset_vector)
    
    for i in range(n_rows):
        if time.time() > DEADLINE:
            # Fallback: return simple circuit
            gates = []
            outputs = []
            next_wire = n + 1
            wire_cache = {}
            
            def get_wire_fallback(subset):
                nonlocal next_wire
                if len(subset) == 0:
                    return zero_wire
                if len(subset) == 1:
                    return next(iter(subset))
                if subset in wire_cache:
                    return wire_cache[subset]
                elems = sorted(subset)
                mid = len(elems) // 2
                s1 = frozenset(elems[:mid])
                s2 = frozenset(elems[mid:])
                w1 = get_wire_fallback(s1)
                w2 = get_wire_fallback(s2)
                if w1 == w2:
                    result = zero_wire
                else:
                    gates.append([w1, w2])
                    result = next_wire
                    next_wire += 1
                wire_cache[subset] = result
                return result
            
            for s in row_sets:
                outputs.append(get_wire_fallback(s))
            return {'gates': gates, 'outputs': outputs}
        
        # Convert current row to frozenset
        r_set = row_sets[i]
        
        # Reduce by existing basis vectors
        for pivot, bv in basis_info:
            if pivot in r_set:
                # XOR r_set with bv
                r_set = frozenset(r_set.symmetric_difference(bv))
        
        if len(r_set) > 0:
            # Find pivot (smallest index in the set)
            pivot = min(r_set)
            basis_info.append((pivot, r_set))
    
    # Step 2: Express each unique row set in terms of basis indices
    unique_sets = list(set_to_rows.keys())
    
    # Build a mapping from basis index to wire
    # First, we need to build the wires for the basis vectors themselves
    # We use a recursive get_wire function that caches all subsets
    
    gates = []
    next_wire = n + 1
    wire_cache = {}
    
    def get_wire(subset):
        nonlocal next_wire
        if len(subset) == 0:
            return zero_wire
        if len(subset) == 1:
            return next(iter(subset))
        if subset in wire_cache:
            return wire_cache[subset]
        
        # Split into two roughly equal parts
        elems = sorted(subset)
        mid = len(elems) // 2
        s1 = frozenset(elems[:mid])
        s2 = frozenset(elems[mid:])
        
        w1 = get_wire(s1)
        w2 = get_wire(s2)
        
        if w1 == w2:
            result = zero_wire
        else:
            gates.append([w1, w2])
            result = next_wire
            next_wire += 1
        
        wire_cache[subset] = result
        return result
    
    # Build wires for each basis vector
    basis_wires = []
    for pivot, bv in basis_info:
        w = get_wire(bv)
        basis_wires.append(w)
    
    # Now, for each unique set, express it in terms of basis and build its wire
    set_to_wire = {}
    
    for s in unique_sets:
        if s in wire_cache:
            set_to_wire[s] = wire_cache[s]
            continue
        
        # Express s in terms of basis
        indices = []
        temp = s
        for idx, (pivot, bv) in enumerate(basis_info):
            if pivot in temp:
                indices.append(idx)
                temp = frozenset(temp.symmetric_difference(bv))
        
        if not indices:
            set_to_wire[s] = zero_wire
        elif len(indices) == 1:
            set_to_wire[s] = basis_wires[indices[0]]
        else:
            # XOR the basis wires together
            # We can use get_wire on the union of the basis vectors? No, that's not correct.
            # We need to XOR the wires.
            # Let's build a small tree for these wires.
            # But we can also just add gates directly.
            # However, to maximize sharing, we should check if the resulting set is already in cache.
            # The resulting set IS 's'.
            # So we just need to create a wire for 's' by XORing the basis wires.
            
            # Let's just XOR them sequentially or in a tree.
            # Since the number of basis elements is likely small, a tree is fine.
            # But we must be careful: the wire for 's' is the XOR of the basis wires.
            # We can create a temporary list of wires and build a tree.
            
            wires_to_xor = [basis_wires[i] for i in indices]
            
            # Build a balanced tree for these wires
            # We can reuse the get_wire logic but on wires, not sets.
            # Actually, it's simpler to just add gates.
            # But we want to cache the result for 's'.
            
            # Let's implement a helper to XOR a list of wires
            def xor_wires(w_list):
                nonlocal next_wire
                if not w_list:
                    return zero_wire
                if len(w_list) == 1:
                    return w_list[0]
                
                # Check if this specific combination is already cached?
                # We don't have a cache for wire combinations, only set combinations.
                # But the result of XORing these wires corresponds to the set 's'.
                # If 's' is not in wire_cache, we need to create a new wire.
                
                mid = len(w_list) // 2
                left_w = xor_wires(w_list[:mid])
                right_w = xor_wires(w_list[mid:])
                
                if left_w == right_w:
                    return zero_wire
                
                gates.append([left_w, right_w])
                result = next_wire
                next_wire += 1
                return result
            
            set_to_wire[s] = xor_wires(wires_to_xor)
            # We should also cache this set in wire_cache for future reference
            wire_cache[s] = set_to_wire[s]
    
    # Build outputs
    outputs = []
    for i in range(n_rows):
        s = row_sets[i]
        outputs.append(set_to_wire[s])
    
    if time.time() > DEADLINE:
        pass
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
