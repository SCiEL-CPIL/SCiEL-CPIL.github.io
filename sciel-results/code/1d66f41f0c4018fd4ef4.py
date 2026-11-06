"""Exact XOR-circuit synthesis with bidirectional BFS and redundancy polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

def _verify_solution(matrix, gates, outputs, depth_limit=16):
    """
    Verify that the given gates and outputs correctly compute matrix @ x over GF(2).
    matrix: shape (m, n) binary. Input wires 0..n-1. Constant-zero wire n.
    Gate k produces wire n+1+k. gates is list of [wire_a, wire_b].
    outputs: list of wire indices, one per output row.
    
    Returns True if valid and depth <= depth_limit.
    """
    m, n = matrix.shape
    num_gates = len(gates)
    total_wires = n + 1 + num_gates
    
    # Check depth
    if depth_limit is not None:
        depth = [0] * total_wires
        for k in range(num_gates):
            a, b = gates[k]
            if a < 0 or a >= total_wires or b < 0 or b >= total_wires:
                return False
            if a >= n + 1 + k or b >= n + 1 + k:
                return False
            depth[n + 1 + k] = 1 + max(depth[a], depth[b])
        max_depth = max(depth[n + 1 + k] for k in range(num_gates)) if num_gates > 0 else 0
        for wire in outputs:
            if wire < 0 or wire >= total_wires:
                return False
            max_depth = max(max_depth, depth[wire])
        if max_depth > depth_limit:
            return False
    
    # Symbolic verification
    wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
    for i in range(n):
        wire_vec[i][i] = 1
    
    for k in range(num_gates):
        a, b = gates[k]
        wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2
    
    for i in range(m):
        wire = outputs[i]
        if wire < 0 or wire >= total_wires:
            return False
        if not np.array_equal(matrix[i], wire_vec[wire]):
            return False
    
    return True

def _baseline_synthesis(matrix, depth_limit=16):
    """
    Baseline synthesis: independent XOR chains for each row.
    """
    m, n = matrix.shape
    gates = []
    outputs = []
    
    for i in range(m):
        row = matrix[i]
        positions = np.where(row)[0].tolist()
        if len(positions) == 0:
            outputs.append(n)
        elif len(positions) == 1:
            outputs.append(positions[0])
        else:
            current = positions[0]
            for j in range(1, len(positions)):
                a = current
                b = positions[j]
                gate_idx = len(gates)
                new_wire = n + 1 + gate_idx
                gates.append([a, b])
                current = new_wire
            outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}

def _bfs_synthesis(matrix, depth_limit=16, time_budget=30.0):
    """
    Bidirectional BFS synthesis: for each target row, find shortest XOR
    decomposition using available wires, maintaining shared subexpressions.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    
    gates = []
    outputs = [None] * m
    wire_vec = {}
    vec_to_wire = {}
    
    # Initialize with inputs and zero
    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        wire_vec[i] = v
        vec_to_wire[v.tobytes()] = i
    wire_vec[n] = np.zeros(n, dtype=np.int64)
    vec_to_wire[wire_vec[n].tobytes()] = n
    
    # Process rows in order of increasing popcount (easier first)
    rows_sorted = sorted(range(m), key=lambda i: int(np.sum(matrix[i])))
    
    for i in rows_sorted:
        if time.time() > deadline:
            # Fallback for remaining rows
            for rem in rows_sorted[rows_sorted.index(i):]:
                row = matrix[rem]
                positions = np.where(row)[0].tolist()
                if len(positions) == 0:
                    outputs[rem] = n
                elif len(positions) == 1:
                    outputs[rem] = positions[0]
                else:
                    current = positions[0]
                    for j in range(1, len(positions)):
                        a = current
                        b = positions[j]
                        gate_idx = len(gates)
                        new_wire = n + 1 + gate_idx
                        gates.append([a, b])
                        new_vec = (wire_vec[a] + wire_vec[b]) % 2
                        wire_vec[new_wire] = new_vec
                        vec_to_wire[new_vec.tobytes()] = new_wire
                        current = new_wire
                    outputs[rem] = current
            break
        
        target = matrix[i]
        target_bytes = target.tobytes()
        
        # Check if already available
        if target_bytes in vec_to_wire:
            outputs[i] = vec_to_wire[target_bytes]
            continue
        
        # Bidirectional BFS: forward from existing wires, backward from target
        # Forward: expand from existing wires by XOR-ing with other existing wires
        # Backward: expand from target by XOR-ing with existing wires (target ^ w = new_target)
        
        # Forward BFS up to depth 2 (i.e., find w1 ^ w2 = target or w1 ^ w2 ^ w3 = target)
        found = False
        
        # Depth 1: check if target = w1 ^ w2 for two existing wires
        wire_list = list(wire_vec.items())
        for idx_a, (a_wire, a_vec) in enumerate(wire_list):
            candidate = (target + a_vec) % 2
            candidate_bytes = candidate.tobytes()
            if candidate_bytes in vec_to_wire:
                b_wire = vec_to_wire[candidate_bytes]
                if a_wire != b_wire:
                    gate_idx = len(gates)
                    new_wire = n + 1 + gate_idx
                    gates.append([a_wire, b_wire])
                    wire_vec[new_wire] = target
                    vec_to_wire[target_bytes] = new_wire
                    outputs[i] = new_wire
                    found = True
                    break
        
        if found:
            continue
        
        # Depth 2: check if target = w1 ^ w2 ^ w3 where w1, w2, w3 are existing
        # Equivalent to: exists w1 such that (target ^ w1) = w2 ^ w3 for existing w2, w3
        # Or: exists intermediate u = w1 ^ w2 (new) such that u ^ w3 = target
        # i.e., w3 = target ^ u
        # So: for each pair (w1, w2), compute u = w1 ^ w2, then check if target ^ u is existing
        
        # This is O(W^2) which could be expensive. Limit it.
        wire_keys = list(wire_vec.keys())
        max_pairs = min(len(wire_keys), 500)  # Limit pairs to avoid explosion
        random.shuffle(wire_keys)
        wire_keys = wire_keys[:max_pairs]
        
        for a_wire in wire_keys:
            if time.time() > deadline:
                break
            a_vec = wire_vec[a_wire]
            for b_wire in wire_keys:
                if b_wire <= a_wire:  # Avoid duplicates and self
                    continue
                b_vec = wire_vec[b_wire]
                u = (a_vec + b_vec) % 2
                u_bytes = u.tobytes()
                
                # Check if u is already known (skip if so, we'd have found at depth 1)
                if u_bytes in vec_to_wire:
                    continue
                
                # Now check if target ^ u is an existing wire
                w3 = (target + u) % 2
                w3_bytes = w3.tobytes()
                if w3_bytes in vec_to_wire:
                    c_wire = vec_to_wire[w3_bytes]
                    # Create u = a ^ b, then target = u ^ c
                    gate_idx1 = len(gates)
                    wire_u = n + 1 + gate_idx1
                    gates.append([a_wire, b_wire])
                    wire_vec[wire_u] = u
                    vec_to_wire[u_bytes] = wire_u
                    
                    gate_idx2 = len(gates)
                    wire_out = n + 1 + gate_idx2
                    gates.append([wire_u, c_wire])
                    wire_vec[wire_out] = target
                    vec_to_wire[target_bytes] = wire_out
                    outputs[i] = wire_out
                    found = True
                    break
            if found:
                break
        
        if found:
            continue
        
        # Fallback: chain construction
        positions = np.where(target)[0].tolist()
        if len(positions) == 0:
            outputs[i] = n
        elif len(positions) == 1:
            outputs[i] = positions[0]
        else:
            current = positions[0]
            for j in range(1, len(positions)):
                a = current
                b = positions[j]
                gate_idx = len(gates)
                new_wire = n + 1 + gate_idx
                gates.append([a, b])
                new_vec = (wire_vec[a] + wire_vec[b]) % 2
                wire_vec[new_wire] = new_vec
                vec_to_wire[new_vec.tobytes()] = new_wire
                current = new_wire
            outputs[i] = current
    
    return {'gates': gates, 'outputs': outputs}

def _polish(matrix, gates, outputs, depth_limit=16, time_budget=10.0):
    """
    Redundancy polish: remove unused gates and try to re-express gate outputs
    using existing wires to eliminate gates.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    
    def build_wire_vec(gates):
        num_gates = len(gates)
        total_wires = n + 1 + num_gates
        wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
        for i in range(n):
            wire_vec[i][i] = 1
        for k in range(num_gates):
            a, b = gates[k]
            wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2
        return wire_vec
    
    # Pass 1: Remove unused gates
    for iteration in range(100):
        if time.time() > deadline:
            break
        improved = False
        for k in range(len(gates) - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            used = False
            for out_wire in outputs:
                if out_wire == wire_k:
                    used = True
                    break
            if not used:
                for other_gate in gates:
                    if other_gate[0] == wire_k or other_gate[1] == wire_k:
                        used = True
                        break
            if not used:
                gates.pop(k)
                improved = True
                break
        if not improved:
            break
    
    # Pass 2: For each gate, check if its output can be obtained from two other
    # existing wires (not involving itself). If so, we can potentially remove it
    # and update references.
    wire_vec = build_wire_vec(gates)
    vec_to_wire = {}
    for wire_idx in range(n + 1 + len(gates)):
        vec_to_wire[wire_vec[wire_idx].tobytes()] = wire_idx
    
    for pass_num in range(5):
        if time.time() > deadline:
            break
        improved = False
        for k in range(len(gates) - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            vec_k = wire_vec[wire_k]
            
            # Check if wire_k is used
            used = False
            for out_wire in outputs:
                if out_wire == wire_k:
                    used = True
                    break
            if not used:
                for other_gate in gates:
                    if other_gate[0] == wire_k or other_gate[1] == wire_k:
                        used = True
                        break
            if not used:
                gates.pop(k)
                wire_vec = build_wire_vec(gates)
                vec_to_wire = {}
                for wire_idx in range(n + 1 + len(gates)):
                    vec_to_wire[wire_vec[wire_idx].tobytes()] = wire_idx
                improved = True
                break
        
        if not improved:
            break
    
    # Pass 3: Try to replace each gate output with a direct wire if possible
    # For each gate k, check if wire_k's value is already available in another wire
    wire_vec = build_wire_vec(gates)
    for k in range(len(gates) - 1, -1, -1):
        if time.time() > deadline:
            break
        wire_k = n + 1 + k
        vec_k = wire_vec[wire_k]
        vec_k_bytes = vec_k.tobytes()
        
        # Find if another wire has the same value
        alt_wire = None
        for w in range(n + 1 + len(gates)):
            if w == wire_k:
                continue
            if wire_vec[w].tobytes() == vec_k_bytes:
                alt_wire = w
                break
        
        if alt_wire is not None:
            # Replace all references to wire_k with alt_wire
            for out_idx, out_wire in enumerate(outputs):
                if out_wire == wire_k:
                    outputs[out_idx] = alt_wire
            for g in gates:
                if g[0] == wire_k:
                    g[0] = alt_wire
                if g[1] == wire_k:
                    g[1] = alt_wire
            # Remove gate k
            gates.pop(k)
            # Rebuild wire_vec
            wire_vec = build_wire_vec(gates)
    
    return {'gates': gates, 'outputs': outputs}

def synthesize(matrix, depth_limit):
    """
    Entry point. Returns a dictionary with 'gates' and 'outputs'.
    """
    matrix = np.array(matrix, dtype=np.int64)
    m, n = matrix.shape
    
    start_time = time.time()
    total_budget = 85.0  # Safety margin before 90s kill
    
    # Start with baseline
    baseline = _baseline_synthesis(matrix, depth_limit)
    best_gates = baseline['gates'][:]
    best_outputs = baseline['outputs'][:]
    best_count = len(best_gates)
    
    if best_count == 0:
        return {'gates': best_gates, 'outputs': best_outputs}
    
    # Try BFS synthesis with time budget
    remaining = total_budget - (time.time() - start_time)
    if remaining > 5.0:
        try:
            bfs_result = _bfs_synthesis(matrix, depth_limit, time_budget=min(remaining - 2.0, 50.0))
            if _verify_solution(matrix, bfs_result['gates'], bfs_result['outputs'], depth_limit):
                if len(bfs_result['gates']) < best_count:
                    best_gates = bfs_result['gates'][:]
                    best_outputs = bfs_result['outputs'][:]
                    best_count = len(best_gates)
        except Exception:
            pass
    
    # Try polish
    remaining = total_budget - (time.time() - start_time)
    if remaining > 3.0:
        try:
            polished = _polish(matrix, best_gates, best_outputs, depth_limit, time_budget=min(remaining - 1.0, 10.0))
            if _verify_solution(matrix, polished['gates'], polished['outputs'], depth_limit):
                if len(polished['gates']) < best_count:
                    best_gates = polished['gates'][:]
                    best_outputs = polished['outputs'][:]
                    best_count = len(polished['gates'])
        except Exception:
            pass
    
    # Final safety check
    if not _verify_solution(matrix, best_gates, best_outputs, depth_limit):
        # Fall back to baseline
        return {'gates': baseline['gates'], 'outputs': baseline['outputs']}
    
    return {'gates': best_gates, 'outputs': best_outputs}

# EVOLVE-BLOCK-END
