import time
import numpy as np

_DEADLINE_SECONDS = 85.0
_start_time = time.time()


def _time_remaining():
    return _DEADLINE_SECONDS - (time.time() - _start_time)


def _verify_circuit(matrix, circuit, depth_limit):
    m, n = matrix.shape
    wire_mask = [0] * (n + 1 + len(circuit['gates']))
    for i in range(n):
        wire_mask[i] = 1 << i
    wire_mask[n] = 0
    depth = [0] * (n + 1 + len(circuit['gates']))
    max_depth = 0
    for k, (a, b) in enumerate(circuit['gates']):
        wire = n + 1 + k
        wire_mask[wire] = wire_mask[a] ^ wire_mask[b]
        d = 1 + max(depth[a], depth[b])
        depth[wire] = d
        max_depth = max(max_depth, d)
    for i in range(m):
        expected = 0
        row = matrix[i]
        for j in range(n):
            if row[j]:
                expected |= (1 << j)
        out_wire = circuit['outputs'][i]
        if wire_mask[out_wire] != expected:
            return False
        max_depth = max(max_depth, depth[out_wire])
    if max_depth > depth_limit:
        return False
    return True


def _build_greedy_circuit(matrix, order=None):
    m, n = matrix.shape
    wire_mask = [0] * (n + 1 + m * 10)
    for i in range(n):
        wire_mask[i] = 1 << i
    wire_mask[n] = 0
    depth = [0] * (n + 1 + m * 10)
    gates = []
    wire_to_gate = {}
    next_gate = 0
    
    def get_xor(a, b):
        nonlocal next_gate
        if a == b:
            return n
        key = (min(a, b), max(a, b))
        if key in wire_to_gate:
            return wire_to_gate[key]
        if next_gate >= len(wire_mask) - n - 1:
            wire_mask.append(0)
            depth.append(0)
        wire = n + 1 + next_gate
        wire_mask[wire] = wire_mask[a] ^ wire_mask[b]
        depth[wire] = 1 + max(depth[a], depth[b])
        gates.append([a, b])
        wire_to_gate[key] = wire
        next_gate += 1
        return wire
    
    if order is None:
        order = list(range(m))
    
    # Build all wires needed
    output_wires = []
    for i in order:
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        # Find the wire with this mask or create it
        # We'll use a greedy approach: build the XOR from scratch but try to share
        # Actually, let me think differently. We want to minimize total gates.
        # For each output, we need a wire that computes that linear form.
        # We can build it as XOR of input wires, sharing sub-expressions.
        
        # Simple approach: build XOR tree for this row's ones
        ones = []
        for j in range(n):
            if matrix[i, j]:
                ones.append(j)
        
        if not ones:
            output_wires.append(n)
        elif len(ones) == 1:
            output_wires.append(ones[0])
        else:
            # Build balanced XOR tree
            def build_xor_tree(indices):
                if len(indices) == 1:
                    return indices[0]
                mid = len(indices) // 2
                left = build_xor_tree(indices[:mid])
                right = build_xor_tree(indices[mid:])
                return get_xor(left, right)
            output_wires.append(build_xor_tree(ones))
    
    # Reorder outputs to match original order
    output_order_map = [0] * m
    for new_idx, orig_idx in enumerate(order):
        output_order_map[orig_idx] = new_idx
    
    final_outputs = [output_wires[output_order_map[i]] for i in range(m)]
    
    return {'gates': gates, 'outputs': final_outputs}


def _try_improve_order(matrix, current_order, rng, iterations=20):
    m, n = matrix.shape
    best_order = current_order[:]
    best_circuit = _build_greedy_circuit(matrix, best_order)
    best_count = len(best_circuit['gates'])
    
    for _ in range(iterations):
        # Try swapping two elements
        i, j = rng.sample(range(m), 2)
        new_order = best_order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        circuit = _build_greedy_circuit(matrix, new_order)
        count = len(circuit['gates'])
        if count < best_count:
            best_count = count
            best_order = new_order
            best_circuit = circuit
    
    return best_circuit, best_order


def synthesize(matrix, depth_limit):
    global _start_time
    _start_time = time.time()
    
    matrix = np.array(matrix, dtype=np.uint8)
    m, n = matrix.shape
    
    # Deadline guard
    deadline = _start_time + _DEADLINE_SECONDS
    
    # Phase 1: Build baseline circuit
    baseline = _build_greedy_circuit(matrix)
    if not _verify_circuit(matrix, baseline, depth_limit):
        # Emergency fallback
        gates = []
        outputs = []
        for i in range(m):
            ones = np.nonzero(matrix[i])[0].tolist()
            if not ones:
                outputs.append(n)
            elif len(ones) == 1:
                outputs.append(ones[0])
            else:
                wire = ones[0]
                for j in range(1, len(ones)):
                    gates.append([wire, ones[j]])
                    wire = n + 1 + len(gates) - 1
                outputs.append(wire)
        return {'gates': gates, 'outputs': outputs}
    
    best_circuit = baseline
    best_count = len(baseline['gates'])
    best_order = list(range(m))
    
    # Phase 2: Try to improve with different orderings
    rng = np.random.RandomState(42)
    import random
    py_rng = random.Random(42)
    
    while time.time() < deadline - 10:
        improved, new_order = _try_improve_order(matrix, best_order, py_rng, iterations=30)
        if len(improved['gates']) < best_count:
            best_circuit = improved
            best_count = len(improved['gates'])
            best_order = new_order
        else:
            break
    
    return best_circuit
