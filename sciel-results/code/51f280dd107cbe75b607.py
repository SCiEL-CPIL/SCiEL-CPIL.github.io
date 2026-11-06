"""Rewritten program for exact XOR-circuit synthesis with stronger algorithms."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

def _xor_count(vec):
    """Count number of 1s in a binary vector."""
    return int(np.sum(vec))

def _row_space_gf2(matrix):
    """Find a basis for the row space of a binary matrix over GF(2)."""
    A = matrix.copy().astype(np.int64)
    m, n = A.shape
    rank = 0
    for col in range(n):
        pivot = -1
        for row in range(rank, m):
            if A[row, col] % 2 == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        if pivot != rank:
            A[rank], A[pivot] = A[pivot].copy(), A[rank].copy()
        for row in range(m):
            if row != rank and A[row, col] % 2 == 1:
                A[row] = (A[row] + A[rank]) % 2
        rank += 1
    return A[0:rank], rank

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

def _greedy_shared_synthesis(matrix, depth_limit=16, time_budget=2.0):
    """
    Greedy synthesis with shared subexpressions.
    Process rows in an order that maximizes sharing.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    
    # Sort rows by popcount (ascending) to potentially share more
    rows_sorted = sorted(range(m), key=lambda i: _xor_count(matrix[i]))
    
    best_gates = None
    best_outputs = None
    best_count = float('inf')
    
    # Try multiple orderings
    for restart in range(30):
        if time.time() > deadline:
            break
        
        if restart == 0:
            order = rows_sorted
        else:
            order = list(range(m))
            random.shuffle(order)
        
        gates = []
        outputs = [None] * m
        wire_vec = {}
        vec_to_wire = {}
        
        for i in range(n):
            v = np.zeros(n, dtype=np.int64)
            v[i] = 1
            wire_vec[i] = v
            vec_to_wire[v.tobytes()] = i
        wire_vec[n] = np.zeros(n, dtype=np.int64)
        vec_to_wire[wire_vec[n].tobytes()] = n
        
        success = True
        for i in order:
            target = matrix[i]
            target_bytes = target.tobytes()
            
            if target_bytes in vec_to_wire:
                outputs[i] = vec_to_wire[target_bytes]
                continue
            
            # Try to find a, b in existing wires such that a XOR b = target
            found = False
            
            # Try each existing wire a, check if target XOR a is also existing
            for a_wire, a_vec in list(wire_vec.items()):
                candidate = (target + a_vec) % 2
                candidate_bytes = candidate.tobytes()
                if candidate_bytes in vec_to_wire:
                    b_wire = vec_to_wire[candidate_bytes]
                    if a_wire != b_wire:
                        gate_idx = len(gates)
                        new_wire = n + 1 + gate_idx
                        gates.append([a_wire, b_wire])
                        wire_vec[new_wire] = candidate
                        vec_to_wire[candidate_bytes] = new_wire
                        outputs[i] = new_wire
                        found = True
                        break
            
            if found:
                continue
            
            # Fallback: build chain using existing wires where possible
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
        
        # --- Second pass: try to eliminate gates by re-expressing outputs ---
        # For each gate, check if its output can be expressed as XOR of two other wires
        # This allows removing the gate and updating references
        if all(o is not None for o in outputs):
            # Build mapping from wire -> list of gates/outputs that use it
            # First, try to remove redundant gates from the end
            for elim_pass in range(3):
                if time.time() > deadline:
                    break
                improved = False
                for k in range(len(gates) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    wire_k = n + 1 + k
                    vec_k = wire_vec[wire_k]
                    vec_k_bytes = vec_k.tobytes()
                    
                    # Check if wire_k is used by any output or later gate
                    used = False
                    for out_wire in outputs:
                        if out_wire == wire_k:
                            used = True
                            break
                    if not used:
                        for other_k in range(k + 1, len(gates)):
                            a, b = gates[other_k]
                            if a == wire_k or b == wire_k:
                                used = True
                                break
                    if used:
                        continue
                    
                    # Not used, so we can remove this gate
                    gates.pop(k)
                    del wire_vec[wire_k]
                    del vec_to_wire[vec_k_bytes]
                    improved = True
                    break
                if not improved:
                    break
            
            # Now try to replace each output: if output is a gate wire,
            # check if that gate's output equals XOR of two other available wires
            # We can then replace the output with one of those wires if the other
            # is a constant or input, but generally we'd need to keep the gate.
            # Instead, try: for each gate, see if its result can be obtained
            # without it (i.e., from remaining wires). If so, remove it and
            # update all references.
            
            # More aggressive: rebuild wire_vec and check each gate
            num_gates = len(gates)
            total_wires = n + 1 + num_gates
            wire_vec2 = [None] * total_wires
            for i in range(n):
                wire_vec2[i] = wire_vec.get(i, np.zeros(n, dtype=np.int64))
                if i < n:
                    wire_vec2[i][i] = 1
            wire_vec2[n] = np.zeros(n, dtype=np.int64)
            for k in range(num_gates):
                a, b = gates[k]
                wire_vec2[n + 1 + k] = (wire_vec2[a] + wire_vec2[b]) % 2
            
            # Try to eliminate gates from the end
            for elim_pass in range(3):
                if time.time() > deadline:
                    break
                improved = False
                for k in range(len(gates) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    wire_k = n + 1 + k
                    vec_k = wire_vec2[wire_k]
                    
                    # Check if wire_k is used by any output or other gate
                    used = False
                    for out_wire in outputs:
                        if out_wire == wire_k:
                            used = True
                            break
                    if not used:
                        for other_k in range(len(gates)):
                            if other_k == k:
                                continue
                            a, b = gates[other_k]
                            if a == wire_k or b == wire_k:
                                used = True
                                break
                    if used:
                        continue
                    
                    # Gate k is unused, remove it
                    gates.pop(k)
                    # Rebuild wire_vec2
                    num_gates = len(gates)
                    total_wires = n + 1 + num_gates
                    wire_vec2 = [None] * total_wires
                    for i in range(n):
                        v = np.zeros(n, dtype=np.int64)
                        v[i] = 1
                        wire_vec2[i] = v
                    wire_vec2[n] = np.zeros(n, dtype=np.int64)
                    for kk in range(num_gates):
                        a, b = gates[kk]
                        wire_vec2[n + 1 + kk] = (wire_vec2[a] + wire_vec2[b]) % 2
                    improved = True
                    break
                if not improved:
                    break
        
        if all(o is not None for o in outputs):
            if _verify_solution(matrix, gates, outputs, depth_limit):
                if len(gates) < best_count:
                    best_count = len(gates)
                    best_gates = gates[:]
                    best_outputs = outputs[:]
    
    if best_gates is not None:
        return {'gates': best_gates, 'outputs': best_outputs}
    
    return _baseline_synthesis(matrix, depth_limit)

def _local_search_synthesis(matrix, depth_limit=16, time_budget=5.0):
    """
    Local search: start from a good solution and try to reduce gate count
    by removing redundant gates and re-expressing outputs.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    
    # Start from greedy shared
    result = _greedy_shared_synthesis(matrix, depth_limit, time_budget=1.0)
    gates = result['gates'][:]
    outputs = result['outputs'][:]
    best_count = len(gates)
    
    if best_count == 0:
        return result
    
    # Build wire_vec for current circuit
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
    
    # Try to remove redundant gates (unused gates)
    for iteration in range(200):
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
                best_count -= 1
                improved = True
                break
        
        if not improved:
            break
    
    # Now try to replace each gate output with an equivalent expression
    # For each gate k producing wire_k, check if wire_k's value can be
    # obtained from two other wires. If so, we can potentially remove gate k
    # and update all references to wire_k to use the new expression.
    # This is complex, so instead we do a simpler optimization:
    # For each output that is a gate wire, check if that gate's inputs
    # can be reordered or if the gate is redundant.
    
    # Try: for each gate, see if its two inputs produce the same result
    # as some other pair of existing wires
    wire_vec = build_wire_vec(gates)
    
    for pass_num in range(3):
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
                improved = True
                break
        
        if not improved:
            break
    
    result = {'gates': gates, 'outputs': outputs}
    
    if _verify_solution(matrix, gates, outputs, depth_limit):
        return result
    
    return _greedy_shared_synthesis(matrix, depth_limit, time_budget=1.0)

def synthesize(matrix, depth_limit):
    """
    Entry point. Returns a dictionary with 'gates' and 'outputs'.
    """
    matrix = np.array(matrix, dtype=np.int64)
    m, n = matrix.shape
    
    # Start with baseline
    baseline = _baseline_synthesis(matrix, depth_limit)
    baseline_gates = len(baseline['gates'])
    
    if baseline_gates == 0:
        return baseline
    
    # Time budget management
    start_time = time.time()
    total_budget = 60.0  # Must finish within 60s per evaluation
    
    # Try greedy shared synthesis with more time
    try:
        greedy = _greedy_shared_synthesis(matrix, depth_limit, time_budget=5.0)
        if len(greedy['gates']) < baseline_gates:
            baseline = greedy
            baseline_gates = len(greedy['gates'])
    except Exception:
        pass
    
    # Try local search
    try:
        local = _local_search_synthesis(matrix, depth_limit, time_budget=15.0)
        if len(local['gates']) < baseline_gates:
            baseline = local
            baseline_gates = len(local['gates'])
    except Exception:
        pass
    
    # Safety margin: return if we've used too much time
    elapsed = time.time() - start_time
    if elapsed > 45.0:
        return baseline
    
    return baseline

# EVOLVE-BLOCK-END
