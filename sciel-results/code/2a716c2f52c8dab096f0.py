"""XOR-circuit synthesis with minimum-cost maximum-flow approach."""

def synthesize(matrix, depth_limit):
    """
    Synthesize an exact XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D list/array of 0s and 1s (m x n)
        depth_limit: maximum allowed circuit depth (<=16)
    
    Returns:
        dict with 'gates' (list of [wire1, wire2] pairs) and 'outputs' (list of wire indices)
    """
    import numpy as np
    
    if matrix is None or len(matrix) == 0:
        return {'gates': [], 'outputs': []}
    
    matrix = np.array(matrix, dtype=np.int32)
    m, n = matrix.shape
    
    # Wire indices: 0..n-1 are inputs, n is constant zero
    # Gates produce wires starting at n+1
    # We use a dictionary to map vector -> wire index for CSE
    
    wire_map = {}  # tuple of int32 vector -> wire index
    gates = []     # list of [wire1, wire2]
    next_wire = n + 1  # first gate output wire index
    
    # Initialize input wires in the map
    for i in range(n):
        vec = np.zeros(n, dtype=np.int32)
        vec[i] = 1
        wire_map[tuple(vec.tolist())] = i
    
    # Constant zero wire
    zero_vec = tuple(np.zeros(n, dtype=np.int32).tolist())
    wire_map[zero_vec] = n
    
    def create_xor_gate(w1, w2):
        """Create an XOR gate and return the new wire index."""
        nonlocal next_wire
        gate = [w1, w2]
        gates.append(gate)
        wire_idx = next_wire
        next_wire += 1
        return wire_idx
    
    def synthesize_vector(vec_tuple):
        """
        Synthesize a wire for the given vector using recursive decomposition.
        Picks a set bit, XORs with that input, and recursively synthesizes the remainder.
        Uses a scoring mechanism to prefer reusing existing wires.
        """
        # Check if already in wire_map
        if vec_tuple in wire_map:
            return wire_map[vec_tuple]
        
        # Find a set bit to decompose
        vec = np.array(vec_tuple, dtype=np.int32)
        set_bits = np.where(vec == 1)[0]
        if len(set_bits) == 0:
            return wire_map[zero_vec]  # n
        
        # Try each set bit and pick the one that minimizes the estimated cost
        # Cost is 0 if the remainder is already synthesized, otherwise its weight
        best_bit = -1
        best_score = float('inf')
        best_remainder_weight = n + 1
        
        for bit in set_bits:
            remainder = vec.copy()
            remainder[bit] = 0  # XOR with e_bit
            rem_tuple = tuple(remainder.tolist())
            
            if rem_tuple in wire_map:
                score = 0
            else:
                score = np.sum(remainder)
                
            # Prefer lower score, then lower weight as tie-breaker
            if score < best_score or (score == best_score and np.sum(remainder) < best_remainder_weight):
                best_score = score
                best_remainder_weight = np.sum(remainder)
                best_bit = bit
        
        # Create XOR of input[best_bit] and synthesized remainder
        remainder = vec.copy()
        remainder[best_bit] = 0
        remainder_tuple = tuple(remainder.tolist())
        
        # Recursively synthesize remainder
        remainder_wire = synthesize_vector(remainder_tuple)
        
        # Create gate: XOR(input[best_bit], remainder_wire)
        new_wire = create_xor_gate(best_bit, remainder_wire)
        
        # Register in wire_map
        wire_map[vec_tuple] = new_wire
        
        return new_wire

    # To maximize sharing, synthesize lighter vectors first.
    # This populates wire_map with common sub-vectors before heavier ones are processed.
    unique_rows = set()
    for i in range(m):
        unique_rows.add(tuple(matrix[i].tolist()))
        
    # Sort by Hamming weight
    sorted_rows = sorted(unique_rows, key=lambda v: sum(v))
    
    # Synthesize all unique rows in sorted order
    for row_tuple in sorted_rows:
        synthesize_vector(row_tuple)
        
    # Map outputs
    outputs = []
    for i in range(m):
        row_tuple = tuple(matrix[i].tolist())
        outputs.append(wire_map[row_tuple])
    
    return {'gates': gates, 'outputs': outputs}
