"""XOR-circuit synthesis with greedy shared-subexpression, BFS decomposition, and gate elimination."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    A = np.array(matrix, dtype=np.int64)
    m, n = A.shape

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to bitmasks
    targets = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if A[i, j] % 2 == 1:
                mask |= (1 << j)
        targets.append(mask)

    # Baseline: independent row synthesis
    baseline_gates = 0
    for t in targets:
        bits = bin(t).count('1')
        if bits > 1:
            baseline_gates += bits - 1
    baseline = max(baseline_gates, 1)

    def time_left():
        return deadline - time.time()

    def compute_wire_values(gates_list, n_inputs):
        """Compute the bitmask value of each wire from the gate list."""
        vals = [1 << i for i in range(n_inputs)]
        vals.append(0)
        for g in gates_list:
            a, b = g
            vals.append(vals[a] ^ vals[b])
        return vals

    def eliminate_gates(gates, outputs):
        """Eliminate redundant gates by checking if a gate's output is already available."""
        if not gates:
            return gates, outputs

        # First pass: identify which gates are actually needed
        # A gate is needed if its output wire is referenced by a later gate or an output
        # But also: if the same value is already available from an earlier wire, we can skip

        # Build wire values
        vals = compute_wire_values(gates, n)

        # For each gate, check if its output value is already in an earlier wire
        # If so, we can redirect all uses of that gate's output to the earlier wire
        # and remove the gate

        # This requires careful remapping, so let's do it iteratively
        # Start with all gates, try to remove each one

        # Build a map from value to first wire that has that value
        # But we need to process in order

        new_gates = []
        wire_remap = list(range(n + 1))  # wire_remap[i] = new wire number for old wire i
        # Old wires 0..n are inputs and zero, they stay the same
        # For each gate, we decide whether to keep it or not

        # Track which old wire numbers map to which new wire numbers
        # Initially: old wire i (0..n) -> new wire i
        # For each gate i (0-indexed in original gates):
        #   old wire number = n + 1 + i
        #   If we keep it: new wire number = n + 1 + len(new_gates)
        #   If we skip it: we need to redirect all future references

        # This is complex. Let's use a different approach:
        # 1. Compute all wire values
        # 2. For each gate, check if its output value appears in any earlier wire
        # 3. If yes, we can skip this gate and redirect all uses

        # Build a dict: value -> list of wire numbers that have that value (in order)
        val_to_wires = {}
        for i in range(n + 1):
            val_to_wires.setdefault(vals[i], []).append(i)

        # Process gates in order, maintaining a map from old wire to new wire
        old_to_new = list(range(n + 1))
        new_gates = []
        new_vals = vals[:n + 1]  # values for inputs and zero

        for gi in range(len(gates)):
            a, b = gates[gi]
            old_wire = n + 1 + gi
            val = vals[old_wire]

            # Check if this value is already available
            # Look in new_vals for the same value
            # But we need to be careful: we can only use wires that were created before this gate
            # In new_vals, all existing wires are available

            found_existing = -1
            for wi in range(len(new_vals)):
                if new_vals[wi] == val:
                    found_existing = wi
                    break

            if found_existing != -1:
                # This gate is redundant; redirect old_wire to found_existing
                old_to_new[old_wire] = found_existing
            else:
                # Keep this gate
                new_a = old_to_new[a]
                new_b = old_to_new[b]
                new_wire = n + 1 + len(new_gates)
                new_gates.append([new_a, new_b])
                new_vals.append(val)
                old_to_new[old_wire] = new_wire

        # Remap outputs
        new_outputs = []
        for o in outputs:
            if o < len(old_to_new):
                new_outputs.append(old_to_new[o])
            else:
                new_outputs.append(n)  # fallback to zero

        return new_gates, new_outputs

    def greedy_synthesize(target_order, seed=None, prefer_new=False, max_2step=100):
        """Greedy synthesis processing targets in given order with 1-step and 2-step search."""
        if seed is not None:
            random.seed(seed)

        wires = []
        for i in range(n):
            wires.append(1 << i)
        wires.append(0)

        wire_map = {w: i for i, w in enumerate(wires)}
        gates = []
        depths = [0] * (n + 1)

        for idx in target_order:
            target = targets[idx]

            if target in wire_map:
                continue

            # Try 1-step: find a, b in available such that a^b = target
            found = False
            avail_list = list(wire_map.keys())
            if seed is not None:
                random.shuffle(avail_list)
            elif prefer_new:
                avail_list.sort(key=lambda w: wire_map[w], reverse=True)

            for a in avail_list:
                b_needed = a ^ target
                if b_needed in wire_map:
                    wa = wire_map[a]
                    wb = wire_map[b_needed]
                    new_wire = n + 1 + len(gates)
                    gates.append([wa, wb])
                    wires.append(target)
                    wire_map[target] = new_wire
                    depths.append(max(depths[wa], depths[wb]) + 1)
                    found = True
                    break

            if found:
                continue

            # Try 2-step: find c in available such that target^c can be made from 2 available wires
            # target = a ^ b ^ c where a, b, c are all available
            # We need: a ^ b = target ^ c, and a, b in available
            # For each c in available, check if target^c can be made from 2 available wires

            best_cost = float('inf')
            best_decomp = None

            # Limit the number of c candidates for efficiency
            # Try zero and input bits first, then other available wires
            c_candidates = []
            if 0 in wire_map:
                c_candidates.append(0)
            for j in range(n):
                bit = 1 << j
                if bit in wire_map:
                    c_candidates.append(bit)
            # Add some other available wires (limited)
            other_wires = [w for w in wire_map.keys() if w not in c_candidates]
            if seed is not None:
                random.shuffle(other_wires)
            else:
                other_wires.sort(key=lambda w: wire_map[w], reverse=True)
            c_candidates.extend(other_wires[:max_2step])

            for c in c_candidates:
                if c not in wire_map:
                    continue
                remainder = target ^ c
                if remainder == 0:
                    continue
                if remainder in wire_map:
                    # target = c ^ remainder, 1 gate (should have been caught by 1-step)
                    wa = wire_map[c]
                    wb = wire_map[remainder]
                    if best_cost > 1:
                        best_cost = 1
                        best_decomp = (wa, wb, None)
                    continue
                # Check if remainder can be made from 2 available wires
                # For each a in a limited set, check if a^remainder is available
                # Use input bits and a few other wires
                a_candidates = [1 << j for j in range(n) if (1 << j) in wire_map]
                if seed is not None:
                    random.shuffle(a_candidates)
                for a in a_candidates:
                    b_needed = a ^ remainder
                    if b_needed in wire_map:
                        wa = wire_map[a]
                        wb = wire_map[b_needed]
                        wc = wire_map[c]
                        if best_cost > 2:
                            best_cost = 2
                            best_decomp = (wa, wb, wc)
                        break
                if best_cost <= 2:
                    break

            if best_decomp is not None:
                if best_cost == 1:
                    wa, wb, _ = best_decomp
                    new_wire = n + 1 + len(gates)
                    gates.append([wa, wb])
                    wires.append(target)
                    wire_map[target] = new_wire
                    depths.append(max(depths[wa], depths[wb]) + 1)
                elif best_cost == 2:
                    wa, wb, wc = best_decomp
                    inter_mask = wires[wa] ^ wires[wb]
                    new_wire1 = n + 1 + len(gates)
                    gates.append([wa, wb])
                    wires.append(inter_mask)
                    wire_map[inter_mask] = new_wire1
                    depths.append(max(depths[wa], depths[wb]) + 1)
                    new_wire2 = n + 1 + len(gates)
                    gates.append([new_wire1, wc])
                    wires.append(target)
                    wire_map[target] = new_wire2
                    depths.append(max(depths[new_wire1], depths[wc]) + 1)
                continue

            # Fallback: build from inputs one by one
            bits = []
            for j in range(n):
                if target & (1 << j):
                    bits.append(j)

            if not bits:
                continue
            if len(bits) == 1:
                continue

            current_wire = wire_map[1 << bits[0]]
            for j in bits[1:]:
                input_wire = wire_map[1 << j]
                new_wire = n + 1 + len(gates)
                gates.append([current_wire, input_wire])
                new_mask = wires[current_wire] ^ (1 << j)
                wires.append(new_mask)
                wire_map[new_mask] = new_wire
                depths.append(max(depths[current_wire], 0) + 1)
                current_wire = new_wire

        # Map outputs
        outputs = []
        for i in range(m):
            target = targets[i]
            if target in wire_map:
                outputs.append(wire_map[target])
            else:
                outputs.append(n)

        return gates, outputs

    def local_search_improve(gates, outputs):
        """Try to improve the circuit by finding better decompositions for individual targets."""
        if not gates:
            return gates, outputs

        # Compute wire values
        vals = compute_wire_values(gates, n)
        wire_map = {}
        for i, v in enumerate(vals):
            if v not in wire_map:
                wire_map[v] = i

        # For each output, try to find a cheaper decomposition
        # This is complex; for now, just return the same circuit
        # A proper implementation would try to re-synthesize individual targets
        # using the existing wires as a starting point

        return gates, outputs

    # Try multiple strategies
    best_gates = None
    best_outputs = None
    best_score = float('inf')

    def try_circuit(gates, outputs):
        nonlocal best_gates, best_outputs, best_score
        # Eliminate redundant gates
        gates, outputs = eliminate_gates(gates, outputs)
        score = len(gates) / baseline
        if score < best_score:
            best_score = score
            best_gates = gates
            best_outputs = outputs

    # Strategy 1: Sort by popcount ascending
    order1 = sorted(range(m), key=lambda i: bin(targets[i]).count('1'))
    g1, o1 = greedy_synthesize(order1)
    try_circuit(g1, o1)

    # Strategy 2: Sort by popcount descending
    order2 = sorted(range(m), key=lambda i: bin(targets[i]).count('1'), reverse=True)
    g2, o2 = greedy_synthesize(order2)
    try_circuit(g2, o2)

    # Strategy 3: Prefer newer wires
    order3 = sorted(range(m), key=lambda i: bin(targets[i]).count('1'))
    g3, o3 = greedy_synthesize(order3, prefer_new=True)
    try_circuit(g3, o3)

    # Strategy 4: Random restarts
    num_restarts = min(50, max(10, int(time_left() / 0.2)))
    for r in range(num_restarts):
        if time_left() < 5.0:
            break
        order = list(range(m))
        random.shuffle(order)
        g, o = greedy_synthesize(order, seed=r)
        try_circuit(g, o)

    # Strategy 5: Random with prefer_new
    num_restarts2 = min(30, max(5, int(time_left() / 0.2)))
    for r in range(num_restarts2):
        if time_left() < 3.0:
            break
        order = list(range(m))
        random.shuffle(order)
        g, o = greedy_synthesize(order, seed=r + 1000, prefer_new=True)
        try_circuit(g, o)

    if best_gates is None:
        best_gates = []
        best_outputs = [n] * m

    return {'gates': best_gates, 'outputs': best_outputs}
# EVOLVE-BLOCK-END
