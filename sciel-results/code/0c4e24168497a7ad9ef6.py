"""XOR-circuit synthesis with gate sharing optimization."""
import time
import numpy as np

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
    deadline = start_time + 75.0  # 75s safety margin
    
    # Normalize input
    m = len(matrix)
    if m == 0:
        return {'gates': [], 'outputs': []}
    n = len(matrix[0])
    A = np.array(matrix, dtype=np.int8)
    
    # Constant zero wire: n
    # Input wires: 0..n-1
    
    # If matrix is all zeros
    if not np.any(A):
        return {'gates': [], 'outputs': [n] * m}
    
    # Row sets: for each output row, the set of input indices that are 1
    row_sets = []
    for i in range(m):
        indices = np.where(A[i] != 0)[0].tolist()
        row_sets.append(set(indices))
    
    # Greedy sharing algorithm
    def greedy_synthesize(row_sets, n, time_budget):
        m = len(row_sets)
        expressions = [set(s) for s in row_sets]
        gates = []
        next_wire = n + 1
        
        # Helper: create gates to compute XOR of a set of wires, return final wire
        def create_xor_tree(wires):
            nonlocal next_wire
            wires = list(wires)
            if len(wires) == 0:
                return n  # zero
            if len(wires) == 1:
                return wires[0]
            prev = wires[0]
            for w in wires[1:]:
                gates.append([prev, w])
                prev = next_wire
                next_wire += 1
            return prev
        
        # Greedy: find best sharing opportunity
        max_iter = min(200, m * m * 2)
        for _ in range(max_iter):
            if time.time() > time_budget:
                break
            
            best_pair = None
            best_common = None
            best_savings = 0
            
            for i in range(m):
                for j in range(i + 1, m):
                    # Common input wires only
                    common = {w for w in (expressions[i] & expressions[j]) if w < n}
                    if len(common) >= 2:
                        savings = len(common) - 1
                        if savings > best_savings:
                            best_savings = savings
                            best_pair = (i, j)
                            best_common = common
            
            if best_pair is None:
                break
            
            i, j = best_pair
            S = best_common
            
            # Create XOR tree for S
            shared_wire = create_xor_tree(S)
            
            # Update expressions
            expressions[i] = (expressions[i] - S) | {shared_wire}
            expressions[j] = (expressions[j] - S) | {shared_wire}
        
        # Build final outputs
        outputs = []
        for i in range(m):
            if not expressions[i]:
                outputs.append(n)
            else:
                outputs.append(create_xor_tree(expressions[i]))
        
        return gates, outputs
    
    # Try greedy
    try:
        gates, outputs = greedy_synthesize(row_sets, n, deadline)
        
        # Verify
        def verify_circuit(A, gates, outputs, n, m):
            num_wires = n + 1 + len(gates)
            wire_val = [0] * num_wires
            for i in range(n):
                wire_val[i] = 1 << i
            wire_val[n] = 0
            for k, (a, b) in enumerate(gates):
                if a >= num_wires or b >= num_wires:
                    return False
                wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
            for i in range(m):
                expected = 0
                for j in range(n):
                    if A[i, j]:
                        expected |= (1 << j)
                if wire_val[outputs[i]] != expected:
                    return False
            return True
        
        if verify_circuit(A, gates, outputs, n, m):
            return {'gates': gates, 'outputs': outputs}
    except Exception:
        pass
    
    # Fallback: baseline
    gates = []
    outputs = []
    next_wire = n + 1
    
    for i in range(m):
        indices = list(row_sets[i])
        if not indices:
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
    
    return {'gates': gates, 'outputs': outputs}
