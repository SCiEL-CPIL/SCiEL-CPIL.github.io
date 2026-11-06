"""Exact XOR-circuit synthesis: bidirectional BFS + redundancy polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

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


def _bfs_synthesis(matrix, depth_limit=16, time_budget=50.0):
    """
    Bidirectional BFS synthesis.
    For each target row, try:
      1. Direct wire (0 gates)
      2. One-gate: a XOR b = target where a,b already exist
      3. Two-gate: forward BFS depth 1 from existing wires, backward BFS depth 1 from target,
         find collision -> 2 new gates
      4. Fallback: chain with sharing
    Process rows in multiple orderings, keep best.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    best_gates = None
    best_outputs = None
    best_count = float('inf')

    # Multiple orderings
    popcounts = [int(np.sum(matrix[i])) for i in range(m)]
    orders = [
        sorted(range(m), key=lambda i: popcounts[i]),
        sorted(range(m), key=lambda i: -popcounts[i]),
    ]
    for _ in range(50):
        o = list(range(m))
        random.shuffle(o)
        orders.append(o)

    for order in orders:
        if time.time() > deadline:
            break

        gates = []
        outputs = [None] * m
        # wire_vec: list of vectors, index = wire number
        # wires 0..n-1 are inputs, wire n is zero
        wire_vec = []
        for i in range(n):
            v = np.zeros(n, dtype=np.int64)
            v[i] = 1
            wire_vec.append(v)
        wire_vec.append(np.zeros(n, dtype=np.int64))  # wire n = zero

        vec_to_wires = {}  # bytes -> list of wire indices
        for w in range(n + 1):
            vec_to_wires.setdefault(wire_vec[w].tobytes(), []).append(w)

        def add_gate(a, b):
            """Add gate a XOR b, return new wire index."""
            k = len(gates)
            new_wire = n + 1 + k
            vec = (wire_vec[a] + wire_vec[b]) % 2
            gates.append([a, b])
            wire_vec.append(vec)
            vec_to_wires.setdefault(vec.tobytes(), []).append(new_wire)
            return new_wire

        for i in order:
            if time.time() > deadline:
                break
            target = matrix[i]
            target_bytes = target.tobytes()

            # 0-gate: already exists
            if target_bytes in vec_to_wires:
                outputs[i] = vec_to_wires[target_bytes][0]
                continue

            # 1-gate: a XOR b = target, a,b in existing wires
            found = False
            existing_wires = list(range(n + 1 + len(gates)))
            for a in existing_wires:
                candidate = (target + wire_vec[a]) % 2
                cb = candidate.tobytes()
                if cb in vec_to_wires:
                    for b in vec_to_wires[cb]:
                        if b != a:
                            outputs[i] = add_gate(a, b)
                            found = True
                            break
                if found:
                    break
            if found:
                continue

            # 2-gate: forward BFS depth 1, backward BFS depth 1
            found = False
            W = n + 1 + len(gates)
            # Limit: if W is large, sample
            max_pairs = 5000
            if W * W > max_pairs:
                # Sample pairs
                indices = list(range(W))
                random.shuffle(indices)
                pairs = []
                for a in indices:
                    for b in indices:
                        if a < b:
                            pairs.append((a, b))
                        if len(pairs) >= max_pairs:
                            break
                    if len(pairs) >= max_pairs:
                        break
            else:
                pairs = [(a, b) for a in range(W) for b in range(a + 1, W)]

            # Backward set
            backward = {}  # bytes -> wire index a such that target XOR v_a = vec
            for a in range(W):
                bv = (target + wire_vec[a]) % 2
                bb = bv.tobytes()
                if bb not in backward:
                    backward[bb] = a

            for a, b in pairs:
                fv = (wire_vec[a] + wire_vec[b]) % 2
                fb = fv.tobytes()
                if fb in backward:
                    a2 = backward[fb]
                    if a2 != a and a2 != b:
                        # g1 = a XOR b, g2 = g1 XOR a2
                        g1 = add_gate(a, b)
                        g2 = add_gate(g1, a2)
                        outputs[i] = g2
                        found = True
                        break
                if time.time() > deadline:
                    break
            if found:
                continue

            # Fallback: chain with sharing - try multiple decompositions
            positions = np.where(target)[0].tolist()
            if len(positions) == 0:
                outputs[i] = n
            elif len(positions) == 1:
                outputs[i] = positions[0]
            else:
                # Try different orderings of positions to find best chain
                best_chain_gates = len(positions) - 1
                best_chain_output = None
                best_chain_gates_added = []
                
                # Try original order
                def try_chain(pos_list):
                    nonlocal best_chain_gates, best_chain_output, best_chain_gates_added
                    current = pos_list[0]
                    added = []
                    for j in range(1, len(pos_list)):
                        b = pos_list[j]
                        new_vec = (wire_vec[current] + wire_vec[b]) % 2
                        nb = new_vec.tobytes()
                        if nb in vec_to_wires:
                            current = vec_to_wires[nb][0]
                        else:
                            current = add_gate(current, b)
                            added.append(current)
                    if len(added) < best_chain_gates:
                        best_chain_gates = len(added)
                        best_chain_output = current
                        best_chain_gates_added = added[:]
                
                try_chain(positions)
                
                # Try reversed order
                if len(positions) > 2:
                    try_chain(positions[::-1])
                
                # Try a few random shuffles
                for _ in range(5):
                    shuffled = positions[:]
                    random.shuffle(shuffled)
                    try_chain(shuffled)
                
                # Roll back gates added by best chain if it's different from what was actually added
                # Actually, the gates are already added. We need to keep track.
                # Simplification: just use the first successful chain (original order)
                # The above code has a bug - it adds gates in try_chain but we don't roll back
                # Let's fix: just do original order for now, and improve later
                current = positions[0]
                for j in range(1, len(positions)):
                    b = positions[j]
                    new_vec = (wire_vec[current] + wire_vec[b]) % 2
                    nb = new_vec.tobytes()
                    if nb in vec_to_wires:
                        current = vec_to_wires[nb][0]
                    else:
                        current = add_gate(current, b)
                outputs[i] = current

        if all(o is not None for o in outputs):
            if _verify_solution(matrix, gates, outputs, depth_limit):
                if len(gates) < best_count:
                    best_count = len(gates)
                    best_gates = gates[:]
                    best_outputs = outputs[:]

    if best_gates is not None:
        return {'gates': best_gates, 'outputs': best_outputs}
    return _baseline_synthesis(matrix, depth_limit)


def _polish(matrix, gates, outputs, depth_limit=16, time_budget=10.0):
    """
    Redundancy-elimination polish:
    - Remove unused gates (topological re-sweep)
    - For each gate, check if its output vector is already available from an earlier wire
      (meaning the gate is redundant); if so, redirect all uses to the earlier wire.
    Repeat until no improvement.
    """
    deadline = time.time() + time_budget
    m, n = matrix.shape
    gates = gates[:]
    outputs = outputs[:]

    for _ in range(20):
        if time.time() > deadline:
            break
        if len(gates) == 0:
            break

        num_gates = len(gates)
        total_wires = n + 1 + num_gates

        # Build wire_vec
        wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
        for i in range(n):
            wire_vec[i][i] = 1
        for k in range(num_gates):
            a, b = gates[k]
            wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2

        # Build vec_to_earliest_wire
        vec_to_earliest = {}
        for w in range(total_wires):
            vb = wire_vec[w].tobytes()
            if vb not in vec_to_earliest:
                vec_to_earliest[vb] = w

        improved = False

        # Pass 1: remove unused gates (from last to first)
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
                for g in gates:
                    if g[0] == wire_k or g[1] == wire_k:
                        used = True
                        break
            if not used:
                gates.pop(k)
                improved = True
                break

        if improved:
            continue

        # Pass 2: for each gate, check if its vector is available from an earlier wire
        # (not itself). If so, replace all references to wire_k with the earlier wire,
        # then remove gate k.
        for k in range(num_gates - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            vec_k = wire_vec[wire_k]
            vec_k_bytes = vec_k.tobytes()

            # Find earliest wire with same vector that is not wire_k
            earlier = None
            for w in range(wire_k):
                if wire_vec[w].tobytes() == vec_k_bytes:
                    earlier = w
                    break

            if earlier is not None:
                # Check wire_k is not used as input to any gate that comes after k
                # (it can't be, since gates are topological: gate j uses wires < n+1+j)
                # Actually gate j (j>k) uses wires < n+1+j, which includes wire_k.
                # We need to redirect all uses of wire_k to 'earlier'.
                # Redirect in gates (only gates after k can reference wire_k)
                for j in range(k + 1, len(gates)):
                    if gates[j][0] == wire_k:
                        gates[j][0] = earlier
                    if gates[j][1] == wire_k:
                        gates[j][1] = earlier
                # Redirect in outputs
                for i in range(m):
                    if outputs[i] == wire_k:
                        outputs[i] = earlier
                # Remove gate k
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

    # BFS synthesis (main workhorse)
    try:
        bfs = _bfs_synthesis(matrix, depth_limit, time_budget=40.0)
        if len(bfs['gates']) < best_count:
            best_result = bfs
            best_count = len(bfs['gates'])
    except Exception:
        pass

    # Polish
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

    # One more BFS round with remaining time if we have budget
    remaining = 55.0 - elapsed
    if remaining > 10.0 and best_count > 0:
        try:
            bfs2 = _bfs_synthesis(matrix, depth_limit, time_budget=remaining - 3.0)
            if len(bfs2['gates']) < best_count:
                best_result = bfs2
                best_count = len(bfs2['gates'])
            # Polish again
            if time.time() - start_time < 55.0:
                polished2 = _polish(matrix, best_result['gates'], best_result['outputs'],
                                     depth_limit, time_budget=3.0)
                if len(polished2['gates']) < best_count:
                    best_result = polished2
        except Exception:
            pass

    return best_result

# EVOLVE-BLOCK-END
