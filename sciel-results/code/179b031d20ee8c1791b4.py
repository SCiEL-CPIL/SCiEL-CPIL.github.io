import time
import random
import numpy as np
from collections import deque

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to bitsets for efficient GF(2) manipulation
    row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if matrix[i][j]:
                bits |= (1 << j)
        row_bits.append(bits)

    # If all zeros
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
        """Independent XOR of inputs for each output."""
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
        """Build the final circuit from expression bitsets and existing gates."""
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
                    # XOR all wires together
                    prev = wires[0]
                    for w in wires[1:]:
                        gates_so_far.append([prev, w])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        return {'gates': gates_so_far, 'outputs': outputs}

    def _greedy_cse(time_budget, seed):
        """
        Greedy Common Subexpression Elimination.
        Repeatedly find the pair of wires (a, b) that appear together in the most
        expressions, create a new wire g = a XOR b, and replace all occurrences
        of {a, b} with g in the expressions.
        """
        rng = random.Random(seed)
        exprs = list(row_bits)
        gates = []
        next_wire = n + 1
        pool = list(range(n + 1))  # inputs 0..n-1 and zero wire n

        max_iter = 300
        for _ in range(max_iter):
            if time.time() > time_budget:
                break

            # Count co-occurrence of each pair of wires
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

            # Find the pair with highest co-occurrence count
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
            pool.append(g)

            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g

            for i in range(m):
                e = exprs[i]
                if (e & a_bit) and (e & b_bit):
                    e = (e & ~a_bit & ~b_bit) | g_bit
                    exprs[i] = e

        return _build_circuit_from_exprs(exprs, gates, next_wire)

    def _global_vector_greedy(time_budget, seed):
        """
        Global vector-space greedy basis construction.
        At each step, select the XOR pair (a, b) from available wires that minimizes
        the total cost to produce all remaining targets, where cost is approximated
        by the minimum Hamming distance from each target to the set of available wires
        plus the new wire. Uses a soft-min surrogate for smoothness.
        """
        rng = random.Random(seed)

        # Initialize wires
        wire_vec = []
        for i in range(n):
            v = 1 << i
            wire_vec.append(v)
        wire_vec.append(0)  # wire n = zero

        gates = []
        outputs = [None] * m
        satisfied = [False] * m

        vec_to_wires = {}
        for w in range(n + 1):
            vec_to_wires[wire_vec[w]] = w

        # Check which targets are already satisfied
        for i in range(m):
            if row_bits[i] in vec_to_wires:
                outputs[i] = vec_to_wires[row_bits[i]]
                satisfied[i] = True

        unsatisfied = [i for i in range(m) if not satisfied[i]]
        if not unsatisfied:
            return {'gates': gates, 'outputs': outputs}

        # Precompute target popcounts
        target_popcounts = [bin(row_bits[i]).count('1') for i in range(m)]

        max_gates = m * n
        gate_count_limit = 200

        # Soft-min temperature
        beta = 1.5

        while unsatisfied and len(gates) < max_gates and len(gates) < gate_count_limit:
            if time.time() > time_budget:
                break

            W = n + 1 + len(gates)
            avail_vecs = wire_vec[:]

            # For each unsatisfied target, compute current min distance to available wires
            # and the score for each candidate new vector.
            U = len(unsatisfied)
            if U == 0:
                break

            # Compute distances from each unsatisfied target to all available wires
            # dist[i, a] = popcount(target_i XOR avail_vecs[a])
            min_dists = []
            for i in unsatisfied:
                t = row_bits[i]
                min_d = min(bin(t ^ av).count('1') for av in avail_vecs)
                min_dists.append(min_d)

            best_score = -1.0
            best_pair = None
            best_new_vec = None

            # Iterate over all pairs of available wires
            # For small W, this is O(W^2 * U * n)
            if W <= 60:
                for a in range(W):
                    for b in range(a + 1, W):
                        new_vec = avail_vecs[a] ^ avail_vecs[b]
                        if new_vec in vec_to_wires:
                            continue

                        # Compute score: sum over unsatisfied targets of
                        # softplus(beta * (min_dist[i] - hamming(t_i, new_vec)))
                        score = 0.0
                        for idx, i in enumerate(unsatisfied):
                            t = row_bits[i]
                            h = bin(t ^ new_vec).count('1')
                            diff = min_dists[idx] - h
                            if diff > 0:
                                score += np.log1p(np.exp(beta * diff))
                            else:
                                score += 0.0

                        if score > best_score:
                            best_score = score
                            best_pair = (a, b)
                            best_new_vec = new_vec
            else:
                # For large W, use a sampling-based approach
                sample_size = min(200, W * (W - 1) // 2)
                pairs = []
                for a in range(W):
                    for b in range(a + 1, W):
                        pairs.append((a, b))
                rng.shuffle(pairs)
                for a, b in pairs[:sample_size]:
                    new_vec = avail_vecs[a] ^ avail_vecs[b]
                    if new_vec in vec_to_wires:
                        continue

                    score = 0.0
                    for idx, i in enumerate(unsatisfied):
                        t = row_bits[i]
                        h = bin(t ^ new_vec).count('1')
                        diff = min_dists[idx] - h
                        if diff > 0:
                            score += np.log1p(np.exp(beta * diff))
                        else:
                            score += 0.0

                    if score > best_score:
                        best_score = score
                        best_pair = (a, b)
                        best_new_vec = new_vec

            if best_pair is None or best_new_vec is None:
                # Fallback: add a direct gate for the cheapest target
                break

            a, b = best_pair
            new_vec = best_new_vec
            new_wire = n + 1 + len(gates)
            gates.append([a, b])
            wire_vec.append(new_vec)
            vec_to_wires[new_vec] = new_wire

            # Check if any unsatisfied target is now satisfied
            new_unsatisfied = []
            for i in unsatisfied:
                if row_bits[i] == new_vec:
                    outputs[i] = new_wire
                    satisfied[i] = True
                else:
                    new_unsatisfied.append(i)
            unsatisfied = new_unsatisfied

        # Build outputs for any remaining unsatisfied targets
        for i in unsatisfied:
            e = row_bits[i]
            if e == 0:
                outputs[i] = n
            elif e in vec_to_wires:
                outputs[i] = vec_to_wires[e]
            else:
                # Direct computation
                indices = [j for j in range(n) if (e >> j) & 1]
                if len(indices) == 1:
                    outputs[i] = indices[0]
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = n + 1 + len(gates) - 1
                    outputs[i] = prev

        return {'gates': gates, 'outputs': outputs}

    def _steiner_mst(time_budget, seed):
        """
        Steiner tree / MST-inspired approach.
        Greedily compute target vectors in order of cheapest cost from
        already-computed vectors, using existing intermediates when possible.
        """
        rng = random.Random(seed)

        wire_to_vec = {}
        vec_to_wire = {}

        for i in range(n):
            vec = 1 << i
            wire_to_vec[i] = vec
            vec_to_wire[vec] = i
        wire_to_vec[n] = 0
        vec_to_wire[0] = n

        next_wire = n + 1
        gates = []

        targets = list(set(row_bits))
        targets = [t for t in targets if t != 0]

        if not targets:
            return {'gates': [], 'outputs': [n] * m}

        pending = list(targets)
        pending_set = set(pending)

        max_iter = 500
        for _ in range(max_iter):
            if time.time() > time_budget:
                break
            if not pending:
                break

            best_target = None
            best_cost = float('inf')
            best_indices = None

            # Shuffle pending for randomization
            rng.shuffle(pending)

            for t in pending:
                if t in vec_to_wire:
                    best_target = t
                    best_cost = 0
                    best_indices = None
                    break

                # Try to compute t from two existing vectors
                # t = u XOR w => u = t XOR w
                found = False
                for w_vec, w_wire in wire_to_vec.items():
                    u_vec = t ^ w_vec
                    if u_vec in vec_to_wire:
                        u_wire = vec_to_wire[u_vec]
                        # Cost: 1 gate
                        cost = 1
                        if cost < best_cost:
                            best_cost = cost
                            best_target = t
                            best_indices = (u_wire, w_wire)
                            found = True
                            break

                if found:
                    continue

                # Fallback: direct computation from inputs
                pc = bin(t).count('1')
                cost_direct = max(0, pc - 1)
                if cost_direct < best_cost:
                    best_cost = cost_direct
                    best_target = t
                    best_indices = 'direct'

            if best_target is None:
                break

            t = best_target

            if t in vec_to_wire:
                pending.remove(t)
                pending_set.discard(t)
                continue

            if best_indices == 'direct':
                indices = [j for j in range(n) if (t >> j) & 1]
                if len(indices) == 0:
                    wire_id = n
                elif len(indices) == 1:
                    wire_id = indices[0]
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    wire_id = prev
            else:
                u_wire, w_wire = best_indices
                gates.append([u_wire, w_wire])
                wire_id = next_wire
                next_wire += 1

            wire_to_vec[wire_id] = t
            vec_to_wire[t] = wire_id
            pending.remove(t)
            pending_set.discard(t)

        # Build outputs
        outputs = []
        for i in range(m):
            e = row_bits[i]
            if e == 0:
                outputs.append(n)
            elif e in vec_to_wire:
                outputs.append(vec_to_wire[e])
            else:
                # Fallback: direct
                indices = [j for j in range(n) if (e >> j) & 1]
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

    # First, get a baseline
    baseline = _baseline_circuit()
    best_result = baseline
    best_gate_count = len(baseline['gates'])

    # Try multiple strategies with different seeds
    strategies = [
        ('greedy_cse', _greedy_cse),
        ('global_vector_greedy', _global_vector_greedy),
        ('steiner_mst', _steiner_mst),
    ]

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for strategy_name, strategy_func in strategies:
        for seed in seeds:
            if time.time() > deadline - 3.0:
                break

            try:
                result = strategy_func(deadline - 3.0, seed)
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
