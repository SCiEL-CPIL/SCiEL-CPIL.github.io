"""
Exact XOR-circuit synthesis via minimum-weight basis completion + local search.

Structural approach:
1. Build an initial circuit using shared XOR trees with multiple heuristics.
2. Apply iterative gate-elimination local search: for each gate, try to
   re-route all consumers to avoid it, keeping the circuit valid.
3. Use a minimum-weight basis completion strategy to select which
   intermediate vectors to create, minimizing average residual weight.

The local search is the key optimization: it eliminates redundant gates
by finding alternative paths for each gate's consumers.
"""

import time
import random
import numpy as np

DEADLINE_SECONDS = 85  # safety margin before 90s kill


def _verify_circuit(matrix, circuit, depth_limit):
    """
    Exact symbolic GF(2) verification.
    Returns True if circuit correctly computes the matrix.
    """
    m, n = matrix.shape
    wire_mask = {}
    for i in range(n):
        wire_mask[i] = (1 << i)
    wire_mask[n] = 0

    for k, (a, b) in enumerate(circuit['gates']):
        wire = n + 1 + k
        wire_mask[wire] = wire_mask.get(a, 0) ^ wire_mask.get(b, 0)

    for i in range(m):
        expected_mask = 0
        for j in range(n):
            if matrix[i, j]:
                expected_mask |= (1 << j)
        out_wire = circuit['outputs'][i]
        if wire_mask.get(out_wire, 0) != expected_mask:
            return False

    if depth_limit < 0:
        return True

    depth = {}
    for i in range(n + 1):
        depth[i] = 0
    max_depth = 0
    for k, (a, b) in enumerate(circuit['gates']):
        wire = n + 1 + k
        d = 1 + max(depth.get(a, 0), depth.get(b, 0))
        depth[wire] = d
        if d > max_depth:
            max_depth = d
    for out in circuit['outputs']:
        if depth.get(out, 0) > max_depth:
            max_depth = depth.get(out, 0)

    return max_depth <= depth_limit


def _compute_wire_masks(circuit, n):
    """Compute the GF(2) mask for every wire in the circuit."""
    m_gates = len(circuit['gates'])
    wire_mask = {}
    for i in range(n):
        wire_mask[i] = (1 << i)
    wire_mask[n] = 0
    for k, (a, b) in enumerate(circuit['gates']):
        wire = n + 1 + k
        wire_mask[wire] = wire_mask.get(a, 0) ^ wire_mask.get(b, 0)
    return wire_mask


def _local_search(matrix, circuit, depth_limit, time_budget=20.0):
    """
    Iterative gate-elimination local search.
    
    For each gate k, check if all consumers of wire (n+1+k) can be
    re-routed to avoid using it. If so, remove the gate and update
    consumer references. Repeat until no more gates can be eliminated.
    
    Also tries swapping equivalent sub-expressions and reordering.
    """
    start_time = time.time()
    m, n = matrix.shape
    
    best_gates = [g[:] for g in circuit['gates']]
    best_outputs = circuit['outputs'][:]
    best_count = len(best_gates)
    
    if best_count == 0:
        return {'gates': best_gates, 'outputs': best_outputs}
    
    # Build consumer map: wire -> list of (gate_idx, position) or ('output', row_idx)
    def build_consumers(gates, outputs):
        consumers = {}
        for k, (a, b) in enumerate(gates):
            for w in (a, b):
                consumers.setdefault(w, []).append(('gate', k, 0 if w == a else 1))
        for i, w in enumerate(outputs):
            consumers.setdefault(w, []).append(('output', i, 0))
        return consumers
    
    def try_eliminate_gate(gates, outputs, gate_idx, wire_masks, n, depth_limit):
        """
        Try to eliminate gate at gate_idx.
        The gate produces wire w = n + 1 + gate_idx with value v = wire_masks[w].
        All consumers of w need to be re-routed.
        
        Strategy: For each consumer, find an alternative wire with the same mask
        that doesn't depend on w, or restructure the consumer gate.
        
        Simplified approach: If w is only consumed by one gate or output,
        we can inline w's value (which is gates[gate_idx][0] ^ gates[gate_idx][1])
        by replacing w with one of its operands in the consumer, but this changes
        the semantics. Instead, we check if w's mask equals some other existing wire.
        """
        w = n + 1 + gate_idx
        v = wire_masks[w]
        
        # Find all consumers of w
        consumers = []
        for k, (a, b) in enumerate(gates):
            if k == gate_idx:
                continue
            if a == w:
                consumers.append(('gate', k, 0))
            elif b == w:
                consumers.append(('gate', k, 1))
        for i, ow in enumerate(outputs):
            if ow == w:
                consumers.append(('output', i, 0))
        
        if not consumers:
            # No consumers, just remove the gate
            new_gates = [g for k, g in enumerate(gates) if k != gate_idx]
            # Remap wire indices: wires after gate_idx shift down by 1
            # This is complex; instead we rebuild
            return new_gates, outputs, True
        
        # Check if v equals mask of some other wire that doesn't depend on w
        # (i.e., a wire with index < n+1+gate_idx, since those don't use w)
        alt_wire = None
        for wid in range(n + 1 + gate_idx):
            if wire_masks.get(wid, -1) == v and wid != n:  # skip zero wire unless v==0
                alt_wire = wid
                break
        
        if alt_wire is None and v == 0:
            alt_wire = n
        
        if alt_wire is None:
            return None, None, False
        
        # Replace all consumers of w with alt_wire
        new_gates = []
        for k, (a, b) in enumerate(gates):
            if k == gate_idx:
                continue
            a_new = alt_wire if a == w else a
            b_new = alt_wire if b == w else b
            if a_new == b_new:
                # This gate would produce zero; need to handle
                # Find another wire with mask 0
                # Actually if a_new == b_new, the gate is redundant
                # We can't just skip it because other gates might reference it
                # For now, mark as failure
                return None, None, False
            new_gates.append([a_new, b_new])
        
        new_outputs = []
        for i, ow in enumerate(outputs):
            new_outputs.append(alt_wire if ow == w else ow)
        
        return new_gates, new_outputs, True
    
    # Iterative elimination pass
    max_rounds = 50
    for round_num in range(max_rounds):
        if time.time() - start_time > time_budget:
            break
        
        gates = best_gates
        outputs = best_outputs
        count = len(gates)
        
        if count == 0:
            break
        
        wire_masks = _compute_wire_masks({'gates': gates, 'outputs': outputs}, n)
        
        # Try eliminating gates in reverse order (later gates first, 
        # since they have fewer potential consumers)
        eliminated_any = False
        for gate_idx in range(count - 1, -1, -1):
            if time.time() - start_time > time_budget:
                break
            
            new_gates, new_outputs, success = try_eliminate_gate(
                gates, outputs, gate_idx, wire_masks, n, depth_limit
            )
            
            if success:
                # Verify the new circuit
                new_circuit = {'gates': new_gates, 'outputs': new_outputs}
                if _verify_circuit(matrix, new_circuit, depth_limit):
                    gates = new_gates
                    outputs = new_outputs
                    wire_masks = _compute_wire_masks(new_circuit, n)
                    eliminated_any = True
        
        if eliminated_any:
            best_gates = gates
            best_outputs = outputs
            best_count = len(gates)
        else:
            break
    
    return {'gates': best_gates, 'outputs': best_outputs}


def _build_circuit_from_rows(rows, n):
    """
    Build a circuit from a list of rows using a global gate registry.
    Uses balanced XOR trees with sharing.
    """
    gate_registry = {}
    gates = []
    next_gate = 0

    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate += 1
        return wire

    outputs = []
    for row in rows:
        if len(row) == 0:
            outputs.append(n)
        elif len(row) == 1:
            outputs.append(row[0])
        else:
            def xor_tree(indices):
                if len(indices) == 1:
                    return indices[0]
                mid = len(indices) // 2
                left = xor_tree(indices[:mid])
                right = xor_tree(indices[mid:])
                return get_xor_wire(left, right)
            outputs.append(xor_tree(row))

    return {'gates': gates, 'outputs': outputs}


def _build_circuit_basis_completion(matrix, n, depth_limit=16, time_budget=10.0):
    """
    Minimum-weight basis completion in GF(2)^n.
    
    Greedily select intermediate vectors (XOR of existing wires) that
    minimize the average Hamming weight of remaining target residuals.
    This restricts the search to linearly independent sub-expressions
    that maximize future sharing.
    """
    start_time = time.time()
    m = matrix.shape[0]
    
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        target_masks.append(mask)
    
    # Available wires
    wire_masks = {}
    for i in range(n):
        wire_masks[i] = (1 << i)
    wire_masks[n] = 0
    
    residuals = target_masks[:]
    satisfied = set()
    for i in range(m):
        if residuals[i] == 0:
            satisfied.add(i)
        else:
            for w, mask in wire_masks.items():
                if mask == residuals[i]:
                    satisfied.add(i)
                    break
    
    gates = []
    gate_registry = {}
    next_gate = 0
    
    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        wire_masks[wire] = wire_masks[a] ^ wire_masks[b]
        next_gate += 1
        return wire
    
    # Greedy: at each step, pick the pair (a,b) whose XOR mask
    # minimizes the sum of popcount(residual[i] ^ new_mask) over
    # all unsatisfied targets, among pairs where new_mask is
    # linearly independent of existing wire masks.
    
    max_iters = 100
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        unsatisfied = [i for i in range(m) if i not in satisfied]
        if not unsatisfied:
            break
        
        wire_ids = list(wire_masks.keys())
        num_w = len(wire_ids)
        
        if num_w < 2:
            break
        
        best_score = float('inf')
        best_pair = None
        
        # Sample a subset of pairs for speed
        if num_w > 40:
            import random as _rnd
            _rnd.seed(42 + it)
            pairs = set()
            for _ in range(500):
                ai = _rnd.randrange(num_w)
                bi = _rnd.randrange(num_w)
                if ai != bi:
                    a = wire_ids[min(ai, bi)]
                    b = wire_ids[max(ai, bi)]
                    pairs.add((a, b))
            pair_list = list(pairs)
        else:
            pair_list = [(wire_ids[i], wire_ids[j]) 
                        for i in range(num_w) for j in range(i+1, num_w)]
        
        for a, b in pair_list:
            new_mask = wire_masks[a] ^ wire_masks[b]
            if new_mask == 0:
                continue
            
            # Check linear independence: new_mask should not be expressible
            # as XOR of existing wires (simplified: just check it's not already there)
            if new_mask in wire_masks.values():
                continue
            
            # Score: average popcount of (residual ^ new_mask) for unsatisfied targets
            # Only count targets where XORing with new_mask reduces popcount
            score = 0
            for i in unsatisfied:
                r = residuals[i]
                if bin(r ^ new_mask).count('1') < bin(r).count('1'):
                    score -= 1  # negative = good
            if score >= 0:
                continue
            
            if score < best_score:
                best_score = score
                best_pair = (a, b)
        
        if best_pair is None:
            break
        
        a, b = best_pair
        new_wire = get_xor_wire(a, b)
        new_mask = wire_masks[new_wire]
        
        # Update residuals
        newly_satisfied = []
        for i in unsatisfied:
            r = residuals[i]
            nr = r ^ new_mask
            if bin(nr).count('1') < bin(r).count('1'):
                residuals[i] = nr
                if nr == 0:
                    newly_satisfied.append(i)
        
        for i in newly_satisfied:
            satisfied.add(i)
    
    # Resolve remaining targets
    outputs = []
    for i in range(m):
        r = residuals[i]
        if r == 0:
            # Find a wire with mask 0
            outputs.append(n)
            continue
        # Find a wire with this mask
        found = False
        for w, mask in wire_masks.items():
            if mask == r:
                outputs.append(w)
                found = True
                break
        if not found:
            ones = [j for j in range(n) if (r >> j) & 1]
            if not ones:
                outputs.append(n)
            elif len(ones) == 1:
                outputs.append(ones[0])
            else:
                def _xt(indices):
                    if len(indices) == 1:
                        return indices[0]
                    mid = len(indices) // 2
                    return get_xor_wire(_xt(indices[:mid]), _xt(indices[mid:]))
                outputs.append(_xt(ones))
    
    return {'gates': gates, 'outputs': outputs}


def _build_circuit_incremental(matrix, n, depth_limit=16):
    """
    Build a circuit using MST-based row ordering and incremental updates.
    Order rows via minimum spanning tree on Hamming distances,
    then build each row sharing sub-expressions with neighbors.
    """
    m = matrix.shape[0]
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)

    # MST ordering for minimum total transition cost
    if m <= 1:
        ordered = list(range(m))
    else:
        # Prim's MST
        in_mst = [False] * m
        in_mst[0] = True
        min_dist = [bin(row_masks[0] ^ row_masks[i]).count('1') for i in range(m)]
        min_dist[0] = 0
        parent = [0] * m
        ordered = [0]

        for _ in range(m - 1):
            best = -1
            best_d = float('inf')
            for i in range(m):
                if not in_mst[i] and min_dist[i] < best_d:
                    best_d = min_dist[i]
                    best = i
            if best == -1:
                break
            in_mst[best] = True
            ordered.append(best)
            for j in range(m):
                if not in_mst[j]:
                    d = bin(row_masks[best] ^ row_masks[j]).count('1')
                    if d < min_dist[j]:
                        min_dist[j] = d
                        parent[j] = best

    gate_registry = {}
    gates = []
    next_gate = 0

    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate += 1
        return wire

    def xor_tree(indices):
        if len(indices) == 1:
            return indices[0]
        mid = len(indices) // 2
        return get_xor_wire(xor_tree(indices[:mid]), xor_tree(indices[mid:]))

    outputs = [None] * m
    for idx in ordered:
        row_mask = row_masks[idx]
        ones = [j for j in range(n) if (row_mask >> j) & 1]
        if len(ones) == 0:
            outputs[idx] = n
        elif len(ones) == 1:
            outputs[idx] = ones[0]
        else:
            outputs[idx] = xor_tree(ones)

    return {'gates': gates, 'outputs': outputs}


def _build_circuit_sequential(matrix, n, depth_limit=16):
    """
    Sequential construction: build rows one by one, reusing existing
    intermediate wires where possible. For each row, greedily pick
    the available wire that reduces the residual the most.
    """
    m = matrix.shape[0]
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        target_masks.append(mask)
    
    gate_registry = {}
    gates = []
    next_gate = 0
    
    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate += 1
        return wire
    
    # Track all wire masks
    wire_masks = {}
    for i in range(n):
        wire_masks[i] = (1 << i)
    wire_masks[n] = 0
    
    def update_wire_masks():
        for k, (a, b) in enumerate(gates):
            wire = n + 1 + k
            wire_masks[wire] = wire_masks[a] ^ wire_masks[b]
    
    outputs = []
    for i in range(m):
        residual = target_masks[i]
        if residual == 0:
            outputs.append(n)
            continue
        
        # Try to build this row by greedily XORing with available wires
        wire = None
        for _ in range(n + 1):  # at most n+1 XORs needed
            # Find the available wire that most reduces residual
            best_wire = None
            best_reduction = 0
            for w, mask in wire_masks.items():
                if mask == 0:
                    continue
                new_res = residual ^ mask
                reduction = bin(residual).count('1') - bin(new_res).count('1')
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_wire = w
            
            if best_wire is None or best_reduction <= 0:
                break
            
            if wire is None:
                wire = best_wire
            else:
                wire = get_xor_wire(wire, best_wire)
            
            residual ^= wire_masks[best_wire]
        
        if residual == 0:
            outputs.append(wire)
        else:
            # Fallback: build from scratch
            ones = [j for j in range(n) if (target_masks[i] >> j) & 1]
            if not ones:
                outputs.append(n)
            elif len(ones) == 1:
                outputs.append(ones[0])
            else:
                def _xt(indices):
                    if len(indices) == 1:
                        return indices[0]
                    mid = len(indices) // 2
                    return get_xor_wire(_xt(indices[:mid]), _xt(indices[mid:]))
                outputs.append(_xt(ones))
        
        update_wire_masks()
    
    return {'gates': gates, 'outputs': outputs}


def _build_circuit_with_common_factors(matrix, n, depth_limit=16):
    """
    Build a circuit by identifying common sub-expressions across rows.
    Pre-compute frequent XOR pairs, then use them during row construction.
    """
    m = matrix.shape[0]

    pair_counts = {}
    for i in range(m):
        ones = [j for j in range(n) if matrix[i, j]]
        for a in range(len(ones)):
            for b in range(a + 1, len(ones)):
                key = (min(ones[a], ones[b]), max(ones[a], ones[b]))
                pair_counts[key] = pair_counts.get(key, 0) + 1

    gate_registry = {}
    gates = []
    next_gate = 0

    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate += 1
        return wire

    # Pre-compute frequent pairs
    frequent_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])
    for pair, count in frequent_pairs:
        if count >= 3:
            get_xor_wire(pair[0], pair[1])

    outputs = []
    for i in range(m):
        ones = [j for j in range(n) if matrix[i, j]]
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            remaining = set(ones)
            wire = None
            while remaining:
                found = False
                for a in list(remaining):
                    for b in remaining:
                        if a == b:
                            continue
                        key = (min(a, b), max(a, b))
                        if key in gate_registry:
                            if wire is None:
                                wire = gate_registry[key]
                            else:
                                wire = get_xor_wire(wire, gate_registry[key])
                            remaining.discard(a)
                            remaining.discard(b)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    for elem in remaining:
                        if wire is None:
                            wire = elem
                        else:
                            wire = get_xor_wire(wire, elem)
                    break
            outputs.append(wire)

    return {'gates': gates, 'outputs': outputs}


def _gaussian_elimination(matrix):
    """
    Perform Gaussian elimination over GF(2).
    Returns rref matrix and pivot column indices.
    """
    m, n = matrix.shape
    mat = matrix.copy().astype(np.uint8)
    pivot_cols = []
    row = 0

    for col in range(n):
        if row >= m:
            break
        pivot_row = -1
        for i in range(row, m):
            if mat[i, col]:
                pivot_row = i
                break
        if pivot_row == -1:
            continue
        mat[[row, pivot_row]] = mat[[pivot_row, row]]
        pivot_cols.append(col)
        for i in range(m):
            if i != row and mat[i, col]:
                mat[i] = (mat[i] + mat[row]) % 2
        row += 1

    return mat, pivot_cols


def _synthesize_direct(matrix, depth_limit):
    """
    Direct synthesis with shared XOR trees.
    """
    m, n = matrix.shape
    rows = [np.nonzero(matrix[i])[0].tolist() for i in range(m)]
    return _build_circuit_from_rows(rows, n)


def _build_circuit_global_greedy(matrix, n, depth_limit=16, time_budget=30.0):
    """
    Global greedy wire-building: iteratively select the XOR pair that
    maximizes the number of remaining targets it helps decompose.
    Uses sampling for speed when wire count is large.
    """
    start_time = time.time()
    m = matrix.shape[0]

    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        target_masks.append(mask)

    wire_masks = {}
    for i in range(n):
        wire_masks[i] = (1 << i)
    wire_masks[n] = 0

    residuals = target_masks[:]
    satisfied = set()
    for i in range(m):
        if residuals[i] == 0:
            satisfied.add(i)
        else:
            for w, mask in wire_masks.items():
                if mask == residuals[i]:
                    satisfied.add(i)
                    break

    gates = []
    gate_registry = {}
    next_gate = 0

    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        wire_masks[wire] = wire_masks[a] ^ wire_masks[b]
        next_gate += 1
        return wire

    max_iterations = 150
    for iteration in range(max_iterations):
        if time.time() - start_time > time_budget:
            break

        unsatisfied = [i for i in range(m) if i not in satisfied]
        if not unsatisfied:
            break

        wire_ids = list(wire_masks.keys())
        num_w = len(wire_ids)
        if num_w < 2:
            break

        best_score = 0
        best_pair = None

        # Use sampling for large wire sets
        if num_w > 30:
            import random as _rnd
            _rnd.seed(123 + iteration)
            candidates = set()
            for _ in range(min(400, num_w * (num_w - 1) // 2)):
                ai = _rnd.randrange(num_w)
                bi = _rnd.randrange(num_w)
                if ai != bi:
                    a = wire_ids[min(ai, bi)]
                    b = wire_ids[max(ai, bi)]
                    candidates.add((a, b))
            pair_list = list(candidates)
        else:
            pair_list = [(wire_ids[i], wire_ids[j]) 
                        for i in range(num_w) for j in range(i + 1, num_w)]

        for a, b in pair_list:
            pair_mask = wire_masks[a] ^ wire_masks[b]
            if pair_mask == 0:
                continue

            score = 0
            for t in unsatisfied:
                if bin(residuals[t] ^ pair_mask).count('1') < bin(residuals[t]).count('1'):
                    score += 1

            if score > best_score:
                best_score = score
                best_pair = (a, b)

        if best_pair is None or best_score == 0:
            break

        a, b = best_pair
        new_wire = get_xor_wire(a, b)
        new_mask = wire_masks[new_wire]

        newly_satisfied = []
        for t in unsatisfied:
            new_res = residuals[t] ^ new_mask
            if bin(new_res).count('1') < bin(residuals[t]).count('1'):
                residuals[t] = new_res
                if new_res == 0:
                    newly_satisfied.append(t)

        for t in newly_satisfied:
            satisfied.add(t)

    # Resolve remaining targets
    outputs = []
    for i in range(m):
        target = target_masks[i]
        found = False
        for w, mask in wire_masks.items():
            if mask == target:
                outputs.append(w)
                found = True
                break
        if not found:
            ones = [j for j in range(n) if (target >> j) & 1]
            if not ones:
                outputs.append(n)
            elif len(ones) == 1:
                outputs.append(ones[0])
            else:
                def _xt(indices):
                    if len(indices) == 1:
                        return indices[0]
                    mid = len(indices) // 2
                    return get_xor_wire(_xt(indices[:mid]), _xt(indices[mid:]))
                outputs.append(_xt(ones))

    return {'gates': gates, 'outputs': outputs}


def _try_alternative_construction(matrix, depth_limit):
    """
    Alternative construction: group rows by pattern, compute each unique
    pattern once with shared XOR tree.
    """
    m, n = matrix.shape

    row_patterns = {}
    for i in range(m):
        ones = tuple(np.nonzero(matrix[i])[0].tolist())
        if ones not in row_patterns:
            row_patterns[ones] = []
        row_patterns[ones].append(i)

    gate_registry = {}
    gates = []
    next_gate = 0

    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate += 1
        return wire

    pattern_to_wire = {}
    for pattern in row_patterns:
        if len(pattern) == 0:
            pattern_to_wire[pattern] = n
        elif len(pattern) == 1:
            pattern_to_wire[pattern] = pattern[0]
        else:
            def xor_tree(indices):
                if len(indices) == 1:
                    return indices[0]
                mid = len(indices) // 2
                return get_xor_wire(xor_tree(indices[:mid]), xor_tree(indices[mid:]))
            pattern_to_wire[pattern] = xor_tree(list(pattern))

    outputs = []
    for i in range(m):
        ones = tuple(np.nonzero(matrix[i])[0].tolist())
        outputs.append(pattern_to_wire[ones])

    return {'gates': gates, 'outputs': outputs}


def _build_circuit_weighted_order(matrix, n, depth_limit=16):
    """
    Build circuit with rows ordered by increasing Hamming weight,
    so simpler rows are built first and their sub-expressions
    are available for reuse by more complex rows.
    """
    m = matrix.shape[0]
    row_info = []
    for i in range(m):
        ones = [j for j in range(n) if matrix[i, j]]
        row_info.append((len(ones), i, ones))
    
    # Sort by weight (ascending)
    row_info.sort(key=lambda x: x[0])
    
    gate_registry = {}
    gates = []
    next_gate = 0

    def get_xor_wire(a, b):
        nonlocal next_gate
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate += 1
        return wire

    def xor_tree(indices):
        if len(indices) == 1:
            return indices[0]
        mid = len(indices) // 2
        return get_xor_wire(xor_tree(indices[:mid]), xor_tree(indices[mid:]))

    outputs = [None] * m
    for _, idx, ones in row_info:
        if len(ones) == 0:
            outputs[idx] = n
        elif len(ones) == 1:
            outputs[idx] = ones[0]
        else:
            outputs[idx] = xor_tree(ones)

    return {'gates': gates, 'outputs': outputs}


def _build_circuit_optimized(matrix, depth_limit):
    """
    Optimized construction: try multiple strategies, pick the best.
    """
    m, n = matrix.shape

    c1 = _build_circuit_from_rows(
        [np.nonzero(matrix[i])[0].tolist() for i in range(m)], n
    )
    c2 = _try_alternative_construction(matrix, depth_limit)
    c3 = _build_circuit_with_common_factors(matrix, n, depth_limit)
    c4 = _build_circuit_incremental(matrix, n, depth_limit)
    c5 = _build_circuit_weighted_order(matrix, n, depth_limit)

    best = min([c1, c2, c3, c4, c5], key=lambda c: len(c['gates']))
    return best


def _local_improvement(matrix, circuit, depth_limit, time_budget=5.0):
    """
    Local improvement: gate-elimination local search.
    Tries to eliminate each gate by finding an equivalent wire,
    and also tries randomized row-order reconstructions.
    """
    start = time.time()
    best = {'gates': [g[:] for g in circuit['gates']], 'outputs': circuit['outputs'][:]}
    best_count = len(best['gates'])
    m, n = matrix.shape

    # Phase 1: Gate elimination
    if best_count > 0:
        eliminated = _local_search(matrix, best, depth_limit, time_budget=time_budget * 0.6)
        if _verify_circuit(matrix, eliminated, depth_limit):
            best = eliminated
            best_count = len(best['gates'])

    # Phase 2: Randomized row-order reconstruction
    remaining_time = time_budget - (time.time() - start)
    if remaining_time > 1.0:
        rng = random.Random(42)
        attempts = 0
        while time.time() - start < time_budget and attempts < 15:
            attempts += 1
            row_order = list(range(m))
            rng.shuffle(row_order)
            rows = [np.nonzero(matrix[i])[0].tolist() for i in row_order]
            c = _build_circuit_from_rows(rows, n)
            if len(c['gates']) < best_count:
                new_outputs = [None] * m
                for new_idx, orig_idx in enumerate(row_order):
                    new_outputs[orig_idx] = c['outputs'][new_idx]
                best = {'gates': c['gates'], 'outputs': new_outputs}
                best_count = len(c['gates'])
                # Try gate elimination on the new best
                if time.time() - start < time_budget - 1.0:
                    elim = _local_search(matrix, best, depth_limit, time_budget=1.0)
                    if _verify_circuit(matrix, elim, depth_limit):
                        best = elim
                        best_count = len(best['gates'])

    return best


def _fallback_chain(matrix, n):
    """
    Guaranteed-valid fallback: sequential XOR chain per row, no sharing.
    """
    m = matrix.shape[0]
    gates = []
    outputs = []
    for i in range(m):
        ones = [j for j in range(n) if matrix[i, j]]
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            wire = ones[0]
            for j in range(1, len(ones)):
                wire = n + 1 + len(gates)
                gates.append([wire - (n + 1) + n + 1, ones[j]])  # previous wire, ones[j]
                # Actually let's track properly:
                # The previous wire is the one we just computed
            # Let me redo this properly
            pass

    # Redo properly
    gates = []
    outputs = []
    for i in range(m):
        ones = [j for j in range(n) if matrix[i, j]]
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            current = ones[0]
            for j in range(1, len(ones)):
                new_wire = n + 1 + len(gates)
                gates.append([current, ones[j]])
                current = new_wire
            outputs.append(current)

    return {'gates': gates, 'outputs': outputs}


def synthesize(matrix, depth_limit):
    """
    Main entry point for XOR-circuit synthesis.

    Args:
        matrix: 2D array (m x n) of 0s and 1s
        depth_limit: maximum allowed depth

    Returns:
        {'gates': [[wire_a, wire_b], ...], 'outputs': [wire, ...]}
    """
    matrix = np.array(matrix, dtype=np.uint8)
    m, n = matrix.shape

    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    candidates = []

    # Strategy 1: Direct with sharing
    if time.time() < deadline - 20:
        c1 = _synthesize_direct(matrix, depth_limit)
        candidates.append(c1)

    # Strategy 2: Pattern grouping
    if time.time() < deadline - 20:
        c2 = _try_alternative_construction(matrix, depth_limit)
        candidates.append(c2)

    # Strategy 3: Common factors
    if time.time() < deadline - 20:
        c3 = _build_circuit_with_common_factors(matrix, n, depth_limit)
        candidates.append(c3)

    # Strategy 4: MST incremental
    if time.time() < deadline - 20:
        c4 = _build_circuit_incremental(matrix, n, depth_limit)
        candidates.append(c4)

    # Strategy 5: Weighted order
    if time.time() < deadline - 20:
        c5 = _build_circuit_weighted_order(matrix, n, depth_limit)
        candidates.append(c5)

    # Strategy 6: Basis completion (new structural approach)
    if time.time() < deadline - 25:
        c6 = _build_circuit_basis_completion(matrix, n, depth_limit,
                                             time_budget=min(15.0, deadline - time.time() - 25))
        candidates.append(c6)

    # Strategy 7: Global greedy with sampling
    if time.time() < deadline - 20:
        c7 = _build_circuit_global_greedy(matrix, n, depth_limit,
                                          time_budget=min(15.0, deadline - time.time() - 20))
        candidates.append(c7)

    # Pick the best verified candidate
    best_circuit = None
    best_count = float('inf')
    for c in candidates:
        if _verify_circuit(matrix, c, depth_limit):
            count = len(c['gates'])
            if count < best_count:
                best_count = count
                best_circuit = c

    # If no valid candidate, use fallback
    if best_circuit is None:
        best_circuit = _fallback_chain(matrix, n)
        if not _verify_circuit(matrix, best_circuit, depth_limit):
            best_circuit = _synthesize_direct(matrix, depth_limit)

    # Local improvement (gate elimination + randomized reconstruction)
    if time.time() < deadline - 3:
        remaining_time = deadline - time.time() - 3
        if remaining_time > 2.0:
            improved = _local_improvement(matrix, best_circuit, depth_limit,
                                          time_budget=min(remaining_time, 30.0))
            if _verify_circuit(matrix, improved, depth_limit):
                best_circuit = improved

    return best_circuit
