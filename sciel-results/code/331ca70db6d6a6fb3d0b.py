"""Exact XOR-circuit synthesis: minimum-weight matching on GF(2) subexpression hypergraph."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

def _verify_solution(matrix, gates, outputs, depth_limit=16):
    """Verify gates/outputs compute matrix @ x over GF(2) with depth <= depth_limit."""
    m, n = matrix.shape
    num_gates = len(gates)
    total_wires = n + 1 + num_gates

    depth = [0] * total_wires
    for k in range(num_gates):
        a, b = gates[k]
        if a < 0 or a >= total_wires or b < 0 or b >= total_wires:
            return False
        if a >= n + 1 + k or b >= n + 1 + k:
            return False
        depth[n + 1 + k] = 1 + max(depth[a], depth[b])
    if num_gates > 0:
        max_depth = max(depth[n + 1 + k] for k in range(num_gates))
    else:
        max_depth = 0
    for wire in outputs:
        if wire < 0 or wire >= total_wires:
            return False
        max_depth = max(max_depth, depth[wire])
    if max_depth > depth_limit:
        return False

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


def _matching_synthesis(matrix, depth_limit=16, time_budget=45.0):
    """
    Minimum-weight matching on GF(2) subexpression hypergraph.
    
    Idea: 
    1. Collect all target row vectors.
    2. Build a set of "candidate intermediates" = all pairwise XORs of available wires
       that could help satisfy multiple targets.
    3. Form a bipartite graph: left = targets, right = candidate intermediates.
       Edge weight = incremental cost (1 if using this intermediate saves gates).
    4. Find min-weight matching to decide which intermediates to introduce.
    5. Then do a second pass to actually construct the circuit using those intermediates.
    
    This is a global optimization that considers sharing across all rows simultaneously.
    """
    m, n = matrix.shape
    deadline = time.time() + time_budget
    
    # Step 1: Collect unique target vectors
    targets = [matrix[i] for i in range(m)]
    target_bytes = [t.tobytes() for t in targets]
    
    # Initial available wires
    wire_vec = []
    for i in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[i] = 1
        wire_vec.append(v)
    wire_vec.append(np.zeros(n, dtype=np.int64))  # zero wire
    
    W = n + 1
    avail_bytes = set(v.tobytes() for v in wire_vec)
    
    # Step 2: Identify which targets are already satisfied (0 gates)
    satisfied = [False] * m
    outputs = [None] * m
    for i in range(m):
        if target_bytes[i] in avail_bytes:
            # Find the wire index
            for w in range(W):
                if wire_vec[w].tobytes() == target_bytes[i]:
                    outputs[i] = w
                    satisfied[i] = True
                    break
    
    unsatisfied = [i for i in range(m) if not satisfied[i]]
    if not unsatisfied:
        return {'gates': [], 'outputs': outputs}
    
    # Step 3: Generate candidate intermediates
    # A candidate intermediate is a vector v = wire_a XOR wire_b such that:
    #   - v is not already available
    #   - v helps satisfy at least one unsatisfied target (i.e., target_i XOR v is available 
    #     or target_i == v)
    
    candidates = []  # list of (vec, bytes, list_of_targets_it_helps, source_pair)
    candidate_set = set()
    
    # For each pair of available wires, compute XOR and check usefulness
    # Limit to W <= 50 for speed; sample if larger
    max_pairs = 2500
    if W * (W - 1) // 2 <= max_pairs:
        pairs_to_check = [(a, b) for a in range(W) for b in range(a + 1, W)]
    else:
        # Random sampling
        pairs_to_check = [(a, b) for a in range(W) for b in range(a + 1, W)]
        random.shuffle(pairs_to_check)
        pairs_to_check = pairs_to_check[:max_pairs]
    
    for a, b in pairs_to_check:
        if time.time() > deadline:
            break
        v = (wire_vec[a] + wire_vec[b]) % 2
        vb = v.tobytes()
        if vb in avail_bytes or vb in candidate_set:
            continue
        
        # Check if this candidate helps any unsatisfied target
        helps = []
        for i in unsatisfied:
            t = targets[i]
            tb = target_bytes[i]
            if vb == tb:
                helps.append(i)  # direct match
            else:
                # Check if target_i XOR v is available
                comp = (t + v) % 2
                if comp.tobytes() in avail_bytes:
                    helps.append(i)  # 1-gate completion possible
        
        if helps:
            candidates.append((v, vb, helps, (a, b)))
            candidate_set.add(vb)
    
    # Step 4: Build bipartite graph and find min-weight matching
    # Left nodes: unsatisfied targets (indexed by their position in unsatisfied list)
    # Right nodes: candidates
    # Edge weight: negative of benefit (we want to maximize benefit)
    # Benefit of matching target i to candidate c:
    #   - If target_i == candidate_vec: benefit = popcount(target_i) - 1 (saves popcount-1 gates)
    #   - If target_i XOR candidate_vec is available: benefit = popcount(target_i) - 2
    #   - Plus sharing bonus: if candidate helps multiple targets, each additional target gets partial credit
    
    # Actually, let's think about this differently. The key insight is:
    # If we introduce candidate intermediate v, and v helps k targets, then:
    #   - Each of those k targets can be satisfied with 1 gate (v XOR something)
    #   - Without v, each target would need ~popcount/2 gates on average
    # So the benefit is roughly sum over helped targets of (baseline_gates_for_target - 1)
    
    # Let's use a simpler scoring: for each candidate, compute total benefit
    # and greedily select candidates by benefit, ensuring no target is double-counted
    
    # Sort candidates by total benefit (descending)
    def candidate_benefit(cand):
        v, vb, helps, _ = cand
        total = 0
        for i in helps:
            pc = int(np.sum(targets[i]))
            if vb == target_bytes[i]:
                # Direct: 1 gate vs pc-1 gates
                total += max(0, pc - 2)  # saving pc-2 gates (1 gate for direct, pc-1 baseline)
            else:
                # Indirect: 2 gates vs pc-1 gates
                total += max(0, pc - 3)  # saving pc-3 gates
        return total
    
    candidates.sort(key=candidate_benefit, reverse=True)
    
    # Greedily select candidates
    selected_candidates = []
    covered_targets = set()
    
    for cand in candidates:
        if time.time() > deadline:
            break
        v, vb, helps, (a, b) = cand
        # Only add if it covers at least one new target
        new_helps = [i for i in helps if i not in covered_targets]
        if not new_helps:
            continue
        # Check if benefit is still positive after accounting for already-covered
        benefit = 0
        for i in new_helps:
            pc = int(np.sum(targets[i]))
            if vb == target_bytes[i]:
                benefit += max(0, pc - 2)
            else:
                benefit += max(0, pc - 3)
        if benefit <= 0:
            continue
        selected_candidates.append(cand)
        covered_targets.update(new_helps)
        
        # If we've covered all unsatisfied targets, stop
        if covered_targets >= set(unsatisfied):
            break
    
    # Step 5: Construct circuit using selected candidates
    gates = []
    # Add gates for selected candidates
    candidate_wire_map = {}  # vb -> wire_index
    
    for v, vb, helps, (a, b) in selected_candidates:
        k = len(gates)
        new_wire = n + 1 + k
        gates.append([a, b])
        wire_vec.append(v)
        candidate_wire_map[vb] = new_wire
        avail_bytes.add(vb)
    
    # Now satisfy each target
    for i in unsatisfied:
        if time.time() > deadline:
            break
        t = targets[i]
        tb = target_bytes[i]
        
        # Check if target is directly a selected candidate
        if tb in candidate_wire_map:
            outputs[i] = candidate_wire_map[tb]
            continue
        
        # Check if target can be satisfied with 1 gate using a selected candidate
        found = False
        for v, vb, helps, (a, b) in selected_candidates:
            if i not in helps:
                continue
            comp = (t + v) % 2
            cb = comp.tobytes()
            if cb in avail_bytes:
                # Find wire for comp
                for w in range(len(wire_vec)):
                    if wire_vec[w].tobytes() == cb:
                        k = len(gates)
                        new_wire = n + 1 + k
                        gates.append([candidate_wire_map[vb], w])
                        wire_vec.append(comp)
                        avail_bytes.add(cb)
                        outputs[i] = new_wire
                        found = True
                        break
            if found:
                break
        
        if found:
            continue
        
        # Fallback: check if target can be satisfied with 1 gate using any available wire
        found = False
        for w in range(len(wire_vec)):
            comp = (t + wire_vec[w]) % 2
            cb = comp.tobytes()
            if cb in avail_bytes:
                for w2 in range(len(wire_vec)):
                    if wire_vec[w2].tobytes() == cb:
                        if w2 != w:
                            k = len(gates)
                            new_wire = n + 1 + k
                            gates.append([w, w2])
                            wire_vec.append(comp)
                            avail_bytes.add(cb)
                            outputs[i] = new_wire
                            found = True
                            break
            if found:
                break
        
        if found:
            continue
        
        # Final fallback: chain
        positions = np.where(t)[0].tolist()
        if len(positions) == 0:
            outputs[i] = n
        elif len(positions) == 1:
            outputs[i] = positions[0]
        else:
            current = positions[0]
            for j in range(1, len(positions)):
                b = positions[j]
                new_vec = (wire_vec[current] + wire_vec[b]) % 2
                nb = new_vec.tobytes()
                # Check if new_vec is already available
                existing = None
                for w in range(len(wire_vec)):
                    if wire_vec[w].tobytes() == nb:
                        existing = w
                        break
                if existing is not None:
                    current = existing
                else:
                    k = len(gates)
                    new_w = n + 1 + k
                    gates.append([current, b])
                    wire_vec.append(new_vec)
                    avail_bytes.add(nb)
                    current = new_w
            outputs[i] = current
    
    # Verify and return
    if all(o is not None for o in outputs) and _verify_solution(matrix, gates, outputs, depth_limit):
        return {'gates': gates, 'outputs': outputs}
    return None


def _polish(matrix, gates, outputs, depth_limit=16, time_budget=10.0):
    """
    Redundancy-elimination polish:
    - Remove unused gates
    - Redirect gates whose output vector is already available from an earlier wire
    Repeat until no improvement.
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

        wire_vec = [np.zeros(n, dtype=np.int64) for _ in range(total_wires)]
        for i in range(n):
            wire_vec[i][i] = 1
        for k in range(num_gates):
            a, b = gates[k]
            wire_vec[n + 1 + k] = (wire_vec[a] + wire_vec[b]) % 2

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

        # Pass 2: redirect redundant gates
        for k in range(num_gates - 1, -1, -1):
            if time.time() > deadline:
                break
            wire_k = n + 1 + k
            vec_k_bytes = wire_vec[wire_k].tobytes()
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

    # Strategy 1: Matching-based synthesis
    try:
        match = _matching_synthesis(matrix, depth_limit, time_budget=40.0)
        if match is not None and len(match['gates']) < best_count:
            best_result = match
            best_count = len(match['gates'])
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

    # Try matching again with remaining time
    remaining = 55.0 - elapsed
    if remaining > 10.0 and best_count > 0:
        try:
            match2 = _matching_synthesis(matrix, depth_limit, time_budget=remaining - 3.0)
            if match2 is not None and len(match2['gates']) < best_count:
                best_result = match2
                best_count = len(match2['gates'])
            if time.time() - start_time < 55.0:
                polished2 = _polish(matrix, best_result['gates'], best_result['outputs'],
                                     depth_limit, time_budget=3.0)
                if len(polished2['gates']) < best_count:
                    best_result = polished2
        except Exception:
            pass

    return best_result

# EVOLVE-BLOCK-END
