"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    t0 = time.time()
    DEADLINE = 85.0
    
    mat = np.array(matrix, dtype=np.uint8)
    if mat.size == 0:
        return {'gates': [], 'outputs': []}
    
    m, n = mat.shape
    zero_wire = n
    
    row_inputs = [np.where(mat[i] == 1)[0].tolist() for i in range(m)]
    row_sets = [frozenset(ri) for ri in row_inputs]
    
    # Find candidate shared subsets from row intersections
    candidates = defaultdict(set)
    for i in range(m):
        for j in range(i + 1, m):
            inter = row_sets[i] & row_sets[j]
            if len(inter) >= 2:
                candidates[inter].add(i)
                candidates[inter].add(j)
    
    sorted_cands = sorted(candidates.items(), key=lambda x: (len(x[0]), len(x[1])), reverse=True)
    
    remaining = [set(ri) for ri in row_inputs]
    extracted = [[] for _ in range(m)]
    used_subsets = set()
    
    gates = []
    wire_to_set = {}
    for i in range(n):
        wire_to_set[i] = frozenset([i])
    wire_to_set[zero_wire] = frozenset()
    next_wire = n + 1
    
    def build_xor_gate(inputs_list):
        nonlocal next_wire
        if len(inputs_list) == 1:
            return inputs_list[0]
        current = inputs_list[0]
        for inp in inputs_list[1:]:
            gates.append([current, inp])
            current = next_wire
            next_wire += 1
        return current
    
    for subset, rows in sorted_cands:
        if time.time() > DEADLINE - 5.0:
            break
        if len(subset) < 2:
            continue
        if subset in used_subsets:
            continue
        
        valid_rows = [r for r in rows if subset.issubset(remaining[r])]
        if len(valid_rows) < 2:
            continue
        
        savings = 0
        for r in valid_rows:
            rem_size = len(remaining[r])
            sub_size = len(subset)
            if rem_size == sub_size:
                savings += (sub_size - 1)
            else:
                savings += (sub_size - 2)
        
        if savings <= 0:
            continue
        
        used_subsets.add(subset)
        shared_wire = build_xor_gate(list(subset))
        wire_to_set[shared_wire] = frozenset(subset)
        
        for r in valid_rows:
            remaining[r] -= subset
            extracted[r].append(shared_wire)
    
    outputs = []
    for i in range(m):
        terms = extracted[i] + list(remaining[i])
        if len(terms) == 0:
            outputs.append(zero_wire)
        elif len(terms) == 1:
            outputs.append(terms[0])
        else:
            out_wire = build_xor_gate(terms)
            outputs.append(out_wire)
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
