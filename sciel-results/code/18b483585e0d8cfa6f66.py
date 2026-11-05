"""XOR-circuit synthesis with minimum-cost maximum-flow approach."""

def synthesize(matrix, depth_limit):
    """
    Synthesize an exact XOR circuit for the given binary matrix.
    Uses RREF basis synthesis to maximize sharing.
    """
    import numpy as np
    
    if matrix is None or len(matrix) == 0:
        return {'gates': [], 'outputs': []}
    
    matrix = np.array(matrix, dtype=np.int32)
    m, n = matrix.shape
    
    # Wire indices: 0..n-1 are inputs, n is constant zero
    wire_map = {}
    gates = []
    next_wire = n + 1
    
    for i in range(n):
        vec = np.zeros(n, dtype=np.int32)
        vec[i] = 1
        wire_map[tuple(vec.tolist())] = i
    
    zero_vec = tuple(np.zeros(n, dtype=np.int32).tolist())
    wire_map[zero_vec] = n
    
    def create_xor_gate(w1, w2):
        nonlocal next_wire
        gate = [w1, w2]
        gates.append(gate)
        wire_idx = next_wire
        next_wire += 1
        return wire_idx

    def synthesize_vector(vec_tuple):
        if vec_tuple in wire_map:
            return wire_map[vec_tuple]
        
        vec = np.array(vec_tuple, dtype=np.int32)
        set_bits = np.where(vec == 1)[0]
        if len(set_bits) == 0:
            return wire_map[zero_vec]
        
        best_bit = -1
        best_score = float('inf')
        best_remainder_weight = n + 1
        
        for bit in set_bits:
            remainder = vec.copy()
            remainder[bit] = 0
            rem_tuple = tuple(remainder.tolist())
            
            if rem_tuple in wire_map:
                score = 0
            else:
                score = np.sum(remainder)
                
            if score < best_score or (score == best_score and np.sum(remainder) < best_remainder_weight):
                best_score = score
                best_remainder_weight = np.sum(remainder)
                best_bit = bit
        
        remainder = vec.copy()
        remainder[best_bit] = 0
        remainder_tuple = tuple(remainder.tolist())
        
        remainder_wire = synthesize_vector(remainder_tuple)
        new_wire = create_xor_gate(best_bit, remainder_wire)
        wire_map[vec_tuple] = new_wire
        return new_wire

    # 1. Compute RREF and pivot columns
    rref = matrix.copy()
    pivot_cols = []
    rank = 0
    for c in range(n):
        pivot_row = -1
        for r in range(rank, m):
            if rref[r, c] == 1:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        rref[[rank, pivot_row]] = rref[[pivot_row, rank]]
        for r in range(m):
            if r != rank and rref[r, c] == 1:
                rref[r] ^= rref[rank]
        pivot_cols.append(c)
        rank += 1
        
    # 2. Extract basis vectors (non-zero rows of RREF)
    basis_vectors = [tuple(rref[i].tolist()) for i in range(rank)]
    
    # 3. Synthesize basis vectors (sorted by weight for better sharing)
    basis_vectors_sorted = sorted(basis_vectors, key=lambda v: sum(v))
    for bv in basis_vectors_sorted:
        synthesize_vector(bv)
        
    # 4. Synthesize outputs by combining basis wires
    outputs = []
    for i in range(m):
        row = matrix[i]
        coeffs = []
        for k in range(rank):
            p_col = pivot_cols[k]
            if row[p_col] == 1:
                coeffs.append(wire_map[basis_vectors[k]])
        
        if not coeffs:
            outputs.append(n)
        else:
            current = coeffs[0]
            for w in coeffs[1:]:
                current = create_xor_gate(current, w)
            outputs.append(current)
            
    return {'gates': gates, 'outputs': outputs}
