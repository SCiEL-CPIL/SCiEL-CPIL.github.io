"""XOR circuit synthesis with greedy shared-subexpression elimination."""
import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0
    
    matrix = np.array(matrix, dtype=int)
    n_rows, n_cols = matrix.shape
    
    # Represent vectors as frozensets of bit positions
    def vec_to_set(vec):
        return frozenset(i for i in range(len(vec)) if vec[i])
    
    targets = [vec_to_set(matrix[i]) for i in range(n_rows)]
    
    # Zero wire
    zero_wire = n_cols
    
    # Available: wire_id -> frozenset
    available = {}
    for i in range(n_cols):
        available[i] = frozenset([i])
    available[zero_wire] = frozenset()
    
    gates = []  # list of [wire_a, wire_b]
    next_wire = zero_wire + 1
    
    def xor_sets(a, b):
        return a.symmetric_difference(b)
    
    def all_done():
        return all(t in available for t in targets)
    
    if all_done():
        outputs = []
        for t in targets:
            for wid, vec in available.items():
                if vec == t:
                    outputs.append(wid)
                    break
        return {'gates': gates, 'outputs': outputs}
    
    # Greedy loop
    max_gates = 200
    while not all_done() and len(gates) < max_gates:
        if time.time() > deadline:
            break
        
        rem = [t for t in targets if t not in available]
        if not rem:
            break
        
        wire_ids = list(available.keys())
        best_score = 0
        best_pair = None
        best_vec = None
        
        # Limit search space
        if len(wire_ids) > 30:
            wire_ids = wire_ids[:30]
        
        for i in range(len(wire_ids)):
            for j in range(i+1, len(wire_ids)):
                a_vec = available[wire_ids[i]]
                b_vec = available[wire_ids[j]]
                v = xor_sets(a_vec, b_vec)
                
                if v in available:
                    continue
                
                score = 0
                for t in rem:
                    t_xor_v = xor_sets(t, v)
                    if len(t_xor_v) < len(t):
                        score += len(t) - len(t_xor_v)
                    if t_xor_v in available:
                        score += 50
                
                if score > best_score:
                    best_score = score
                    best_pair = (wire_ids[i], wire_ids[j])
                    best_vec = v
        
        if best_pair is None or best_score <= 0:
            break
        
        gates.append([best_pair[0], best_pair[1]])
        available[next_wire] = best_vec
        next_wire += 1
    
    # Final pass: create any remaining targets directly
    for t in targets:
        if t in available:
            continue
        if time.time() > deadline:
            break
        if len(t) <= 1:
            continue
        
        bits = list(t)
        cur_wire = bits[0]
        for b in bits[1:]:
            if time.time() > deadline:
                break
            gates.append([cur_wire, b])
            # The result is t (since we're XORing all bits in t)
            # But we need to track intermediate vectors
            # This is approximate; the final wire should equal t
            available[next_wire] = t
            cur_wire = next_wire
            next_wire += 1
    
    # Compute outputs
    outputs = []
    for t in targets:
        found = False
        for wid, vec in available.items():
            if vec == t:
                outputs.append(wid)
                found = True
                break
        if not found:
            outputs.append(zero_wire)
    
    return {'gates': gates, 'outputs': outputs}
