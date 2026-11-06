import time
import random
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if matrix[i][j]:
                bits |= (1 << j)
        row_bits.append(bits)

    if not any(row_bits):
        return {'gates': [], 'outputs': [n] * m}

    def _verify(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
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
            expected = row_bits[i]
            if outputs[i] >= num_wires:
                return False
            if wire_val[outputs[i]] != expected:
                return False
        return True

    def _compute_depth(gates, outputs):
        if not gates:
            return 0
        num_wires = n + 1 + len(gates)
        wire_depth = [0] * num_wires
        for k, (a, b) in enumerate(gates):
            wire_depth[n + 1 + k] = max(wire_depth[a], wire_depth[b]) + 1
        max_depth = 0
        for w in outputs:
            if w < num_wires:
                max_depth = max(max_depth, wire_depth[w])
        return max_depth

    def _baseline_circuit():
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            bits = row_bits[i]
            if bits == 0:
                outputs.append(n)
            else:
                indices = [j for j in range(n) if (bits >> j) & 1]
                if len(indices) == 1:
                    outputs.append(indices[0])
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        return {'gates': gates, 'outputs': outputs}

    def _build_circuit_from_exprs(exprs, gates_so_far, next_wire):
        outputs = []
        for i in range(m):
            e = exprs[i]
            if e == 0:
                outputs.append(n)
            else:
                wires = []
                temp = e
                w = 0
                while temp:
                    if temp & 1:
                        wires.append(w)
                    temp >>= 1
                    w += 1

                if len(wires) == 1:
                    outputs.append(wires[0])
                elif len(wires) == 0:
                    outputs.append(n)
                else:
                    prev = wires[0]
                    for w in wires[1:]:
                        gates_so_far.append([prev, w])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        return {'gates': gates_so_far, 'outputs': outputs}

    def _local_search(circuit, time_budget, rng):
        """
        Try to improve the circuit by removing unused gates or merging gates.
        """
        gates = [list(g) for g in circuit['gates']]
        outputs = circuit['outputs'][:]
        
        # Map wire to gate index
        # wire n+1+k is produced by gates[k]
        
        def _is_valid():
            num_wires = n + 1 + len(gates)
            wire_val = [0] * num_wires
            for i in range(n):
                wire_val[i] = 1 << i
            wire_val[n] = 0
            for k, (a, b) in enumerate(gates):
                wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
            for i in range(m):
                if outputs[i] >= num_wires:
                    return False
                if wire_val[outputs[i]] != row_bits[i]:
                    return False
            return True

        if not _is_valid():
            return circuit

        improved = True
        while improved and time.time() < time_budget:
            improved = False
            # Try removing each gate
            for k in range(len(gates)):
                if time.time() > time_budget:
                    break
                # Check if gate k is used by any output or other gate
                used = False
                out_wire = n + 1 + k
                for o in outputs:
                    if o == out_wire:
                        used = True
                        break
                if not used:
                    for i in range(len(gates)):
                        if i == k: continue
                        a, b = gates[i]
                        if a == out_wire or b == out_wire:
                            used = True
                            break
                if not used:
                    # Remove gate k
                    gates.pop(k)
                    # Renumber outputs
                    # This is tricky because wire numbers change.
                    # Easier: just rebuild the circuit from scratch? No.
                    # Let's just mark it for removal and rebuild at the end.
                    # For now, this local search is complex.
                    # Let's skip complex local search and rely on better greedy.
                    pass

        return {'gates': gates, 'outputs': outputs}

    def _greedy_cse_v5(time_budget, seed):
        """
        Greedy CSE with randomization.
        """
        rng = random.Random(seed)
        exprs = list(row_bits)
        gates = []
        next_wire = n + 1

        max_iter = 500
        for _ in range(max_iter):
            if time.time() > time_budget:
                break

            co_occ = {}
            for e in exprs:
                if e == 0:
                    continue
                wires_in_e = []
                temp = e
                w = 0
                while temp:
                    if temp & 1:
                        wires_in_e.append(w)
                    temp >>= 1
                    w += 1

                for idx_a in range(len(wires_in_e)):
                    for idx_b in range(idx_a + 1, len(wires_in_e)):
                        a = wires_in_e[idx_a]
                        b = wires_in_e[idx_b]
                        key = (a, b)
                        co_occ[key] = co_occ.get(key, 0) + 1

            if not co_occ:
                break

            best_pair = None
            best_count = 1
            items = list(co_occ.items())
            rng.shuffle(items)
            for (a, b), cnt in items:
                if cnt > best_count:
                    best_count = cnt
                    best_pair = (a, b)

            if best_pair is None:
                break

            a, b = best_pair
            g = next_wire
            next_wire += 1
            gates.append([a, b])

            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g

            for i in range(m):
                e = exprs[i]
                if (e & a_bit) and (e & b_bit):
                    e = (e & ~a_bit & ~b_bit) | g_bit
                    exprs[i] = e

        return _build_circuit_from_exprs(exprs, gates, next_wire)

    baseline = _baseline_circuit()
    best_result = baseline
    best_gate_count = len(baseline['gates'])

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    for seed in seeds:
        if time.time() > deadline - 3.0:
            break

        try:
            result = _greedy_cse_v5(deadline - 3.0, seed)
            if _verify(result):
                d = _compute_depth(result['gates'], result['outputs'])
                if d <= depth_limit:
                    gc = len(result['gates'])
                    if gc < best_gate_count:
                        best_gate_count = gc
                        best_result = result
        except Exception:
            pass

    return best_result
