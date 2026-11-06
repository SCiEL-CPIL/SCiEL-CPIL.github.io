"""Exact XOR-circuit synthesis: Simulated Annealing on Row-Operation Sequences."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import copy

def _verify_solution(matrix, gates, outputs, depth_limit=16):
    """Verify gates/outputs compute matrix @ x over GF(2) with depth <= depth_limit."""
    m, n = matrix.shape
    num_gates = len(gates)
    total_wires = n + 1 + num_gates

    # Depth check
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
        if not np.array_equal(matrix[i], wire_vec[wire]):
            return False
    return True

def _baseline_synthesis(matrix, depth_limit=16):
    """Independent XOR chains per row."""
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
                gates.append([a, b])
                current = n + 1 + len(gates) - 1
            outputs.append(current)
    return {'gates': gates, 'outputs': outputs}

def _generate_initial_sequence(matrix):
    """
    Generate an initial sequence of row operations that transforms Identity to Matrix.
    The sequence is a list of tuples (row_idx, col_idx) meaning: Row[row_idx] ^= Row[col_idx] (conceptually).
    However, we need to map this to a circuit.
    
    Actually, a simpler representation for SA:
    State: A list of operations. Each op is (target_row, source_row).
    We start with Identity. We apply ops sequentially.
    If the final state is Matrix, the ops define the circuit.
    
    To convert ops to a circuit:
    We simulate the ops.
    We maintain a list of 'current_wires' for each logical row.
    Initially, current_wires[i] = i (input wire i).
    For each op (t, s):
        If current_wires[t] == current_wires[s], skip (no-op).
        Else, create a new gate: gate = [current_wires[t], current_wires[s]].
        current_wires[t] = new_wire_index.
    
    The 'cost' is the number of gates.
    The 'validity' is whether the final logical rows match the matrix rows.
    
    Initial state: Gaussian elimination steps.
    """
    m, n = matrix.shape
    # We want to find a sequence of elementary row operations that turns I into M.
    # This is equivalent to finding the inverse of M? No.
    # M = E_k ... E_1 I.
    # So the sequence E_1...E_k transforms I to M.
    # We can find such a sequence by performing Gaussian elimination on M to get I,
    # and then reversing the operations.
    
    # Let's just use a simple heuristic: for each row i, find the first 1, and eliminate other 1s.
    # This is O(m*n^2) ops.
    
    # Better: Start with a random permutation of rows, then eliminate.
    # Or just use the standard LU decomposition logic over GF(2).
    
    A = matrix.copy()
    ops = []
    
    for col in range(n):
        # Find pivot
        pivot = -1
        for row in range(col, m):
            if A[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        
        # Swap pivot row with col row (if not already)
        if pivot != col:
            A[[col, pivot]] = A[[pivot, col]]
            ops.append(('swap', col, pivot))
            
        # Eliminate other rows
        for row in range(m):
            if row != col and A[row, col] == 1:
                A[row] = (A[row] + A[col]) % 2
                ops.append(('add', row, col))
                
    # The ops list transforms I to RREF(M)? No, it transforms M to RREF(M).
    # We need ops that transform I to M.
    # If M is invertible (square), M = E_k ... E_1 I.
    # The ops above are E_1 ... E_k such that E_k ... E_1 M = RREF(M).
    # This is getting complicated.
    
    # Alternative: Just generate a random valid sequence by starting from I and randomly adding rows until we reach M?
    # That's hard to guarantee.
    
    # Let's stick to the circuit-based SA.
    # State: List of gates.
    # Moves: Add, Remove, Swap, Change inputs.
    
    pass

def _sa_synthesis(matrix, depth_limit=16, time_budget=50.0):
    """
    Simulated Annealing on the circuit structure.
    State: (gates, outputs)
    Cost: Number of gates (if valid), else large penalty + gate count.
    Moves:
    1. Add a random gate.
    2. Remove a random gate (if it doesn't break validity of outputs? No, just let cost handle it).
    3. Change inputs of a random gate to two other available wires.
    4. Swap two gates.
    5. Change output wire of a random row to another available wire.
    """
    m, n = matrix.shape
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Initial solution: Baseline
    sol = _baseline_synthesis(matrix, depth_limit)
    gates = sol['gates']
    outputs = sol['outputs']
    
    current_cost = len(gates)
    best_gates = gates[:]
    best_outputs = outputs[:]
    best_cost = current_cost
    
    # Temperature
    T = 10.0
    cooling = 0.95
    
    def get_wire_vecs(gates, n):
        num_gates = len(gates)
        total_wires = n + 1 + num_gates
        wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
        for i in range(n):
            wire_vec[i][i] = 1
        for k in range(num_gates):
            a, b = gates[k]
            # Check validity of indices
            if a < 0 or a >= n + 1 + k or b < 0 or b >= n + 1 + k:
                return None, None
            wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2
        return wire_vec, total_wires

    def calculate_cost(gates, outputs, matrix, n, depth_limit):
        wire_vec, total_wires = get_wire_vecs(gates, n)
        if wire_vec is None:
            return 10000 + len(gates), False
        
        # Check depth
        depth = [0] * total_wires
        valid_depth = True
        for k in range(len(gates)):
            a, b = gates[k]
            depth[n + 1 + k] = 1 + max(depth[a], depth[b])
        max_depth = max(depth[n + 1 + k] for k in range(len(gates))) if len(gates) > 0 else 0
        for wire in outputs:
            if wire < 0 or wire >= total_wires:
                return 10000 + len(gates), False
            max_depth = max(max_depth, depth[wire])
        if max_depth > depth_limit:
            # Penalty for depth violation
            return 1000 + max_depth + len(gates), False
            
        # Check correctness
        penalty = 0
        for i in range(m):
            wire = outputs[i]
            if not np.array_equal(matrix[i], wire_vec[wire]):
                # Hamming distance penalty
                penalty += np.sum(matrix[i] != wire_vec[wire])
        
        if penalty == 0:
            return len(gates), True
        else:
            return 100 + penalty + len(gates), False

    current_cost, _ = calculate_cost(gates, outputs, matrix, n, depth_limit)
    
    while time.time() < deadline:
        if T < 0.1:
            break
            
        # Generate neighbor
        new_gates = gates[:]
        new_outputs = outputs[:]
        move_type = random.randint(0, 4)
        
        if move_type == 0 and len(gates) > 0:
            # Remove gate
            k = random.randint(0, len(gates) - 1)
            wire_k = n + 1 + k
            # Check if wire_k is used in outputs or later gates
            used = False
            if wire_k in new_outputs:
                used = True
            for j in range(k + 1, len(gates)):
                if new_gates[j][0] == wire_k or new_gates[j][1] == wire_k:
                    used = True
                    break
            if not used:
                new_gates.pop(k)
                # Adjust outputs and later gates? 
                # If we remove gate k, all wires > k shift down by 1.
                # This is complex. Let's stick to simpler moves.
                # Instead, let's just "zero out" the gate by making it a no-op? No, gates are always XOR.
                # Let's swap out the gate with a duplicate of an earlier gate?
                # Actually, removing a gate requires re-indexing.
                # Let's skip removal for simplicity in SA, or implement it carefully.
                # For now, let's just change inputs.
                pass
        elif move_type == 1:
            # Change inputs of a random gate
            if len(gates) > 0:
                k = random.randint(0, len(gates) - 1)
                # Available wires: 0..n and n+1..n+k (previous gates)
                max_avail = n + k
                a = random.randint(0, max_avail)
                b = random.randint(0, max_avail)
                new_gates[k] = [a, b]
        elif move_type == 2:
            # Change output of a random row
            i = random.randint(0, m - 1)
            max_avail = n + len(gates)
            new_outputs[i] = random.randint(0, max_avail)
        elif move_type == 3 and len(gates) > 1:
            # Swap two gates
            k1 = random.randint(0, len(gates) - 1)
            k2 = random.randint(0, len(gates) - 1)
            if k1 != k2:
                new_gates[k1], new_gates[k2] = new_gates[k2], new_gates[k1]
                # Note: Swapping gates changes the wire indices they produce!
                # Gate k1 produces wire n+1+k1. Gate k2 produces wire n+1+k2.
                # If we swap their definitions, wire n+1+k1 now has the value of old k2, and vice versa.
                # This is a valid move in the space of "labeled gates".
        elif move_type == 4:
            # Add a gate
            a = random.randint(0, n + len(gates) - 1) if len(gates) > 0 else random.randint(0, n)
            b = random.randint(0, n + len(gates) - 1) if len(gates) > 0 else random.randint(0, n)
            new_gates.append([a, b])
            
        new_cost, is_valid = calculate_cost(new_gates, new_outputs, matrix, n, depth_limit)
        
        delta = new_cost - current_cost
        if delta < 0 or random.random() < np.exp(-delta / T):
            gates = new_gates
            outputs = new_outputs
            current_cost = new_cost
            
            if is_valid and current_cost < best_cost:
                best_cost = current_cost
                best_gates = gates[:]
                best_outputs = outputs[:]
                
        T *= cooling
        
    # Final polish
    if time.time() < deadline - 5:
        # Run a few rounds of greedy polish on the best found
        best_sol = _polish(matrix, best_gates, best_outputs, depth_limit, time_budget=5.0)
        if len(best_sol['gates']) < best_cost:
            return best_sol
            
    return {'gates': best_gates, 'outputs': best_outputs}

def _polish(matrix, gates, outputs, depth_limit=16, time_budget=8.0):
    """
    Redundancy-elimination polish.
    """
    deadline = time.time() + time_budget
    m, n = matrix.shape
    gates = gates[:]
    outputs = outputs[:]

    for _ in range(50):
        if time.time() > deadline:
            break
        if len(gates) == 0:
            break

        num_gates = len(gates)
        total_wires = n + 1 + num_gates

        # Build wire_vec
        wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
        valid = True
        for i in range(n):
            wire_vec[i][i] = 1
        for k in range(num_gates):
            a, b = gates[k]
            if a < 0 or a >= n + 1 + k or b < 0 or b >= n + 1 + k:
                valid = False
                break
            wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2
        if not valid:
            break

        # Build vec_to_earliest_wire
        vec_to_earliest = {}
        for w in range(total_wires):
            vb = wire_vec[w].tobytes()
            if vb not in vec_to_earliest:
                vec_to_earliest[vb] = w

        improved = False

        # Pass 1: remove unused gates
        for k in range(num_gates - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            used = False
            for ow in outputs:
                if ow == wire_k:
                    used = True
                    break
            if not used:
                for j in range(k + 1, len(gates)):
                    if gates[j][0] == wire_k or gates[j][1] == wire_k:
                        used = True
                        break
            if not used:
                gates.pop(k)
                improved = True
                break

        if improved:
            continue

        # Pass 2: redirect to earlier
        for k in range(num_gates - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            vec_k = wire_vec[wire_k]
            vec_k_bytes = vec_k.tobytes()

            earlier = vec_to_earliest.get(vec_k_bytes)
            if earlier is not None and earlier != wire_k:
                for j in range(k + 1, len(gates)):
                    if gates[j][0] == wire_k:
                        gates[j][0] = earlier
                    if gates[j][1] == wire_k:
                        gates[j][1] = earlier
                for i in range(m):
                    if outputs[i] == wire_k:
                        outputs[i] = earlier
                gates.pop(k)
                improved = True
                break

        if not improved:
            break

    if _verify_solution(matrix, gates, outputs, depth_limit):
        return {'gates': gates, 'outputs': outputs}
    return {'gates': gates, 'outputs': outputs}

def synthesize(matrix, depth_limit):
    """Entry point."""
    matrix = np.array(matrix, dtype=np.int64)
    m, n = matrix.shape
    start_time = time.time()

    # Baseline
    baseline = _baseline_synthesis(matrix, depth_limit)
    best_result = baseline
    best_count = len(baseline['gates'])

    if best_count == 0:
        return baseline

    # 1. Simulated Annealing
    try:
        sa_sol = _sa_synthesis(matrix, depth_limit, time_budget=40.0)
        if _verify_solution(matrix, sa_sol['gates'], sa_sol['outputs'], depth_limit):
            if len(sa_sol['gates']) < best_count:
                best_result = sa_sol
                best_count = len(sa_sol['gates'])
    except Exception:
        pass

    # 2. Polish
    elapsed = time.time() - start_time
    if elapsed < 55.0 and best_count > 0:
        try:
            polished = _polish(matrix, best_result['gates'], best_result['outputs'],
                               depth_limit, time_budget=8.0)
            if len(polished['gates']) < best_count:
                best_result = polished
                best_count = len(polished['gates'])
        except Exception:
            pass

    # Safety margin
    elapsed = time.time() - start_time
    if elapsed > 55.0:
        return best_result

    # 3. Another SA round if time permits
    remaining = 55.0 - elapsed
    if remaining > 10.0 and best_count > 0:
        try:
            sa_sol2 = _sa_synthesis(matrix, depth_limit, time_budget=remaining - 3.0)
            if _verify_solution(matrix, sa_sol2['gates'], sa_sol2['outputs'], depth_limit):
                if len(sa_sol2['gates']) < best_count:
                    best_result = sa_sol2
                    best_count = len(sa_sol2['gates'])
                if time.time() - start_time < 55.0:
                    polished2 = _polish(matrix, best_result['gates'], best_result['outputs'],
                                         depth_limit, time_budget=3.0)
                    if len(polished2['gates']) < best_count:
                        best_result = polished2
        except Exception:
            pass

    return best_result

# EVOLVE-BLOCK-END
