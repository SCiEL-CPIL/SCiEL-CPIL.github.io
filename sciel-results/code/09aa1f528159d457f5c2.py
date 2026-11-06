"""Exact XOR-circuit synthesis: global vector-space greedy basis construction."""
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


def _global_greedy_synthesis(matrix, depth_limit=16, time_budget=40.0):
    """
    Global vector-space greedy basis construction.
    
    At each step, we have a set of available wires (inputs, zero, and gates created so far).
    We want to satisfy all target rows (rows of matrix) using as few gates as possible.
    
    Key insight: instead of processing rows one by one, we iteratively add the single gate
    whose new vector provides the greatest *global* benefit — i.e., it reduces the
    minimum-gate-distance to the most unsatisfied targets collectively.
    
    For each target row not yet satisfied, we compute its "distance" = minimum number of
    additional XOR gates needed to produce it from available wires. A new wire w reduces
    the distance of target t by 1 if dist(t, with w) < dist(t, without w).
    
    We approximate distance via BFS in the vector space: from available wires, what's the
    shortest XOR chain to reach target t?
    
    Optimization: we maintain for each target a BFS frontier. When a new wire is added,
    we update frontiers. The gate with the largest total reduction across all targets wins.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    
    # Targets: rows of matrix that are not already available as input wires or zero
    # Available wires start as 0..n (inputs + zero)
    wire_vec = []
    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        wire_vec.append(v)
    wire_vec.append(np.zeros(n, dtype=np.int64))  # wire n = zero
    
    vec_to_wire = {}
    for w in range(n + 1):
        vec_to_wire[wire_vec[w].tobytes()] = w
    
    # Determine which targets are already satisfied
    target_vecs = []
    target_satisfied = []
    for i in range(m):
        tv = matrix[i]
        tb = tv.tobytes()
        target_vecs.append(tv)
        if tb in vec_to_wire:
            target_satisfied.append(vec_to_wire[tb])
        else:
            target_satisfied.append(-1)
    
    # If all satisfied, done
    if all(s >= 0 for s in target_satisfied):
        gates = []
        outputs = target_satisfied[:]
        return {'gates': gates, 'outputs': outputs}
    
    # For each unsatisfied target, compute initial min-distance to available wires
    # dist[i] = minimum number of XOR gates to produce target i from available wires
    # We'll track this via a simple BFS per target (limited depth)
    
    def compute_min_dist(target, available_vecs, max_depth=4):
        """Min number of XORs from available_vecs to reach target. -1 if not found within max_depth."""
        target_b = target.tobytes()
        # Depth 0: is target already available?
        if target_b in vec_to_wire:
            return 0
        # Depth 1: target = a XOR b for a,b in available
        # This is expensive; use a different approach
        # Actually, let's use a BFS from target backwards
        # Or: for small available sets, enumerate pairs
        
        # For efficiency, we'll compute this lazily and cache
        return -1  # placeholder
    
    # Better approach: maintain a "frontier" for each target
    # frontier[t] = set of vectors at distance d from target t (reachable in d more gates)
    # Initially frontier[t] = {target XOR w : w in available} at distance 1
    # If any frontier element is in available, target is at distance 1
    
    # Actually, let's use a simpler but effective heuristic:
    # For each candidate new wire (a XOR b for a,b in available), compute how many
    # unsatisfied targets it would help (i.e., targets where (target XOR new_wire) is
    # closer to available than target was).
    
    # Simplified: a target t is "helped" by new wire w if (t XOR w) is available
    # (i.e., t can be produced in 1 more gate after w is created).
    # More generally, t is helped if dist(t, available ∪ {w}) < dist(t, available).
    
    # Let's track for each target the set of "one-step-away" vectors:
    # one_step[t] = {target[t] XOR w : w in available}
    # If v is in one_step[t] and v is available, then t is at distance 1.
    # When we add a new wire w, t becomes at distance 1 if target[t] XOR w is in available
    # OR if target[t] XOR w == w' for some available w' (trivially true).
    
    # Even simpler effective heuristic:
    # For candidate new wire v = wire_a XOR wire_b:
    #   benefit = number of unsatisfied targets t where (target[t] XOR v) is in available_vecs
    #   (meaning t would be producible in 1 gate after v exists)
    #   PLUS: number of unsatisfied targets t where target[t] == v (directly satisfied)
    
    gates = []
    W = n + 1  # current number of available wires
    
    while any(s < 0 for s in target_satisfied):
        if time.time() > deadline:
            break
        
        # Find the best new wire to add
        best_benefit = 0
        best_a = -1
        best_b = -1
        best_vec = None
        
        # Enumerate all pairs (a, b) with a < b from available wires
        # For efficiency, if W is large, sample
        if W <= 40:
            pairs = [(a, b) for a in range(W) for b in range(a + 1, W)]
        else:
            # Sample pairs, prioritizing pairs that produce useful vectors
            pairs = []
            indices = list(range(W))
            random.shuffle(indices)
            for a in indices:
                for b in indices:
                    if a < b:
                        pairs.append((a, b))
                    if len(pairs) >= 3000:
                        break
                if len(pairs) >= 3000:
                    break
        
        for a, b in pairs:
            if time.time() > deadline:
                break
            v = (wire_vec[a] + wire_vec[b]) % 2
            vb = v.tobytes()
            
            # Skip if already available
            if vb in vec_to_wire:
                continue
            
            # Compute benefit: how many unsatisfied targets would be directly satisfied
            # or become 1-step-away
            benefit = 0
            for i in range(m):
                if target_satisfied[i] >= 0:
                    continue
                # Direct satisfaction
                if target_vecs[i].tobytes() == vb:
                    benefit += 10  # high weight for direct satisfaction
                    continue
                # 1-step: target[i] XOR v is available
                cand = (target_vecs[i] + v) % 2
                if cand.tobytes() in vec_to_wire:
                    benefit += 1
            
            if benefit > best_benefit:
                best_benefit = benefit
                best_a = a
                best_b = b
                best_vec = v
        
        if best_benefit <= 0 or best_a < 0:
            # No beneficial gate found; fall back to chain for remaining targets
            break
        
        # Add the best gate
        k = len(gates)
        new_wire = n + 1 + k
        gates.append([best_a, best_b])
        wire_vec.append(best_vec)
        vec_to_wire[best_vec.tobytes()] = new_wire
        W += 1
        
        # Update satisfied targets
        for i in range(m):
            if target_satisfied[i] < 0 and target_vecs[i].tobytes() == best_vec.tobytes():
                target_satisfied[i] = new_wire
    
    # For remaining unsatisfied targets, use chain with sharing
    for i in range(m):
        if target_satisfied[i] >= 0:
            continue
        if time.time() > deadline:
            break
        target = target_vecs[i]
        tb = target.tobytes()
        
        # Try 1-gate: a XOR b = target
        found = False
        for a in range(W):
            cand = (target + wire_vec[a]) % 2
            cb = cand.tobytes()
            if cb in vec_to_wire:
                b = vec_to_wire[cb]
                if b != a:
                    k = len(gates)
                    new_wire = n + 1 + k
                    gates.append([a, b])
                    wire_vec.append(cand)
                    vec_to_wire[cb] = new_wire
                    W += 1
                    target_satisfied[i] = new_wire
                    found = True
                    break
        if found:
            continue
        
        # Try 2-gate
        found = False
        # Forward: v_a XOR v_b for a,b in available
        # Backward: target XOR v_a for a in available
        # Collision means 2 gates suffice
        backward = {}
        for a in range(W):
            bv = (target + wire_vec[a]) % 2
            bb = bv.tobytes()
            if bb not in backward:
                backward[bb] = a
        
        max_pairs_check = 5000
        if W * W <= max_pairs_check:
            pair_list = [(a, b) for a in range(W) for b in range(a + 1, W)]
        else:
            pair_list = []
            indices = list(range(W))
            random.shuffle(indices)
            for a in indices:
                for b in indices:
                    if a < b:
                        pair_list.append((a, b))
                    if len(pair_list) >= max_pairs_check:
                        break
                if len(pair_list) >= max_pairs_check:
                    break
        
        for a, b in pair_list:
            fv = (wire_vec[a] + wire_vec[b]) % 2
            fb = fv.tobytes()
            if fb in backward:
                a2 = backward[fb]
                if a2 != a and a2 != b:
                    # g1 = a XOR b, g2 = g1 XOR a2
                    k1 = len(gates)
                    g1_wire = n + 1 + k1
                    gates.append([a, b])
                    wire_vec.append(fv)
                    vec_to_wire[fb] = g1_wire
                    W += 1
                    
                    k2 = len(gates)
                    g2_wire = n + 1 + k2
                    gates.append([g1_wire, a2])
                    wire_vec.append((fv + wire_vec[a2]) % 2)
                    vec_to_wire[wire_vec[-1].tobytes()] = g2_wire
                    W += 1
                    target_satisfied[i] = g2_wire
                    found = True
                    break
            if time.time() > deadline:
                break
        if found:
            continue
        
        # Fallback: chain
        positions = np.where(target)[0].tolist()
        if len(positions) == 0:
            target_satisfied[i] = n
        elif len(positions) == 1:
            target_satisfied[i] = positions[0]
        else:
            current = positions[0]
            for j in range(1, len(positions)):
                b = positions[j]
                new_vec = (wire_vec[current] + wire_vec[b]) % 2
                nb = new_vec.tobytes()
                if nb in vec_to_wire:
                    current = vec_to_wire[nb]
                else:
                    k = len(gates)
                    new_wire = n + 1 + k
                    gates.append([current, b])
                    wire_vec.append(new_vec)
                    vec_to_wire[nb] = new_wire
                    W += 1
                    current = new_wire
            target_satisfied[i] = current
    
    outputs = target_satisfied[:]
    if any(o < 0 for o in outputs):
        # Shouldn't happen, but fallback
        return _baseline_synthesis(matrix, depth_limit)
    
    return {'gates': gates, 'outputs': outputs}


def _polish(matrix, gates, outputs, depth_limit=16, time_budget=8.0):
    """
    Redundancy-elimination polish:
    - Remove unused gates
    - For each gate, check if its output vector is already available from an earlier wire
    """
    deadline = time.time() + time_budget
    m, n = matrix.shape
    gates = gates[:]
    outputs = outputs[:]

    for _ in range(30):
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
        for k in range(num_gates - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            vec_k = wire_vec[wire_k]
            vec_k_bytes = vec_k.tobytes()

            earlier = vec_to_earliest.get(vec_k_bytes)
            if earlier is not None and earlier != wire_k:
                # Redirect all uses of wire_k to 'earlier'
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

    # Global greedy synthesis (main workhorse)
    try:
        greedy = _global_greedy_synthesis(matrix, depth_limit, time_budget=35.0)
        if len(greedy['gates']) < best_count:
            best_result = greedy
            best_count = len(greedy['gates'])
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

    # Additional greedy round with remaining time
    remaining = 55.0 - elapsed
    if remaining > 10.0 and best_count > 0:
        try:
            greedy2 = _global_greedy_synthesis(matrix, depth_limit, time_budget=remaining - 3.0)
            if len(greedy2['gates']) < best_count:
                best_result = greedy2
                best_count = len(greedy2['gates'])
            if time.time() - start_time < 55.0:
                polished2 = _polish(matrix, best_result['gates'], best_result['outputs'],
                                     depth_limit, time_budget=3.0)
                if len(polished2['gates']) < best_count:
                    best_result = polished2
        except Exception:
            pass

    return best_result

# EVOLVE-BLOCK-END
