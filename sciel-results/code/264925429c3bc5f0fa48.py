import numpy as np
import time
import heapq
from collections import defaultdict

def _popcount(x):
    return bin(x).count('1')

def _verify(gates, outputs, matrix, n):
    """Verify circuit correctness symbolically over GF(2)."""
    m = len(matrix)
    # wires: 0..n-1 = inputs, n = 0, n+1.. = gates
    wire_val = [0] * (n + 1 + len(gates))
    for i in range(n):
        wire_val[i] = 1 << i
    wire_val[n] = 0
    for k, (a, b) in enumerate(gates):
        wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]
    for i in range(m):
        expected = 0
        for j in range(n):
            if matrix[i][j]:
                expected |= (1 << j)
        if i >= len(outputs):
            return False
        if wire_val[outputs[i]] != expected:
            return False
    return True

def _circuit_depth(gates, n):
    """Compute max depth of the circuit."""
    if not gates:
        return 0
    depth = [0] * (n + 1 + len(gates))
    for k, (a, b) in enumerate(gates):
        depth[n + 1 + k] = max(depth[a], depth[b]) + 1
    return max(depth) if depth else 0

def _baseline_gate_count(matrix):
    """Baseline: each output built independently from inputs (chain XOR)."""
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    total = 0
    for i in range(m):
        bits = sum(1 for j in range(n) if matrix[i][j])
        total += max(0, bits - 1)  # chain of (bits-1) gates
    return total

def _build_direct(matrix, n, m):
    """Build circuit with no sharing: each output from scratch."""
    gates = []
    outputs = []
    next_wire = n + 1
    for i in range(m):
        bits = [j for j in range(n) if matrix[i][j]]
        if not bits:
            outputs.append(n)
        elif len(bits) == 1:
            outputs.append(bits[0])
        else:
            curr = bits[0]
            for b in bits[1:]:
                gates.append([curr, b])
                curr = next_wire
                next_wire += 1
            outputs.append(curr)
    return gates, outputs

def _build_greedy_cse(matrix, n, m, deadline):
    """
    Greedy CSE: maintain a set of available wire masks.
    At each step, find the pair (a,b) of available wires whose XOR w
    maximizes utility: number of remaining targets T where (T ^ w) has
    low popcount (easy to complete) OR w is itself a target.
    """
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j]:
                mask |= (1 << j)
        target_masks.append(mask)
    
    unique_targets = set(target_masks)
    
    available = set()
    for i in range(n):
        available.add(1 << i)
    available.add(0)
    
    mask_to_wire = {}
    for i in range(n):
        mask_to_wire[1 << i] = i
    mask_to_wire[0] = n
    
    gates = []
    next_wire = n + 1
    remaining = set(unique_targets)
    
    # Remove already-available targets
    for t in list(remaining):
        if t in available:
            remaining.discard(t)
    
    while remaining:
        if time.time() > deadline:
            break
        
        avail_list = list(available)
        L = len(avail_list)
        
        best_w = None
        best_score = -1
        best_a = None
        best_b = None
        
        # Score: for candidate w = a^b (new wire):
        # +1000 if w in remaining (direct solve)
        # +10 * count of T in remaining where (T^w) in available (one-step complete)
        # +5 * count of T in remaining where popcount(T^w) <= 2 (cheap complete)
        # -1 * popcount(w) (penalize complex wires slightly)
        
        for i in range(L):
            a = avail_list[i]
            for j in range(i + 1, L):
                b = avail_list[j]
                w = a ^ b
                if w in available:
                    continue
                
                score = 0
                if w in remaining:
                    score += 10000
                
                for t in remaining:
                    diff = t ^ w
                    if diff in available:
                        score += 100
                    elif _popcount(diff) <= 2:
                        score += 10
                
                # Small penalty for high popcount to prefer simpler intermediates
                score -= _popcount(w)
                
                if score > best_score:
                    best_score = score
                    best_w = w
                    best_a = a
                    best_b = b
                    if w in remaining and score > 10000:
                        break
            if best_w in remaining and best_score > 10000:
                break
        
        if best_w is None:
            # No useful pair found; fallback to direct for remaining
            break
        
        wa = mask_to_wire[best_a]
        wb = mask_to_wire[best_b]
        gates.append([wa, wb])
        new_idx = next_wire
        next_wire += 1
        mask_to_wire[best_w] = new_idx
        available.add(best_w)
        
        if best_w in remaining:
            remaining.discard(best_w)
    
    # Fallback for unsolved targets
    unsolved = set(target_masks) - available
    for t in unsolved:
        if time.time() > deadline:
            # Must still produce valid output; use direct chain
            bits = [j for j in range(n) if (t >> j) & 1]
            if not bits:
                continue
            elif len(bits) == 1:
                mask_to_wire[t] = bits[0]
                available.add(t)
            else:
                curr = bits[0]
                for b in bits[1:]:
                    gates.append([curr, b])
                    curr = next_wire
                    next_wire += 1
                mask_to_wire[t] = curr
                available.add(t)
            continue
        
        # Try to use existing wires to reduce cost
        best_pair = None
        best_cost = _popcount(t)
        for w in available:
            diff = t ^ w
            if diff in available:
                # Can complete in 1 gate
                if _popcount(w) + _popcount(diff) < best_cost:
                    best_cost = _popcount(w) + _popcount(diff)
                    best_pair = (w, diff)
                break  # one gate is optimal if possible
        
        if best_pair is None:
            # Find w that minimizes popcount(t^w)
            best_w = 0
            min_diff_pop = best_cost
            for w in available:
                diff = t ^ w
                p = _popcount(diff)
                if p < min_diff_pop:
                    min_diff_pop = p
                    best_w = w
            best_pair = (best_w, t ^ best_w)
        
        w1, w2 = best_pair
        if w2 in available:
            gates.append([mask_to_wire[w1], mask_to_wire[w2]])
            new_idx = next_wire
            next_wire += 1
            mask_to_wire[t] = new_idx
            available.add(t)
        else:
            # Build w2 from inputs
            bits = [j for j in range(n) if (w2 >> j) & 1]
            if not bits:
                wire_w2 = n
            elif len(bits) == 1:
                wire_w2 = bits[0]
            else:
                curr = bits[0]
                for b in bits[1:]:
                    gates.append([curr, b])
                    curr = next_wire
                    next_wire += 1
                wire_w2 = curr
            gates.append([mask_to_wire[w1], wire_w2])
            new_idx = next_wire
            next_wire += 1
            mask_to_wire[t] = new_idx
            available.add(t)
    
    outputs = []
    for t in target_masks:
        if t in mask_to_wire:
            outputs.append(mask_to_wire[t])
        else:
            outputs.append(n)
    
    return gates, outputs

def _local_search(gates, outputs, matrix, n, m, deadline):
    """
    Attempt to reduce gate count by:
    1. Identifying redundant gates (gate output never used).
    2. Replacing 2-gate chains with direct wires where possible.
    """
    if not gates:
        return gates, outputs
    
    # Track which wires are used
    used = set()
    for o in outputs:
        used.add(o)
    for a, b in gates:
        used.add(a)
        used.add(b)
    
    # Remove unused gates (iterate to fixpoint)
    changed = True
    while changed:
        changed = False
        new_gates = []
        # Rebuild mask_to_wire for current gates
        mask_to_wire = {}
        for i in range(n):
            mask_to_wire[1 << i] = i
        mask_to_wire[0] = n
        wire_mask = {}
        for i in range(n):
            wire_mask[i] = 1 << i
        wire_mask[n] = 0
        for k, (a, b) in enumerate(gates):
            idx = n + 1 + k
            wire_mask[idx] = wire_mask[a] ^ wire_mask[b]
            mask_to_wire[wire_mask[idx]] = idx
        
        # Determine which gate outputs are actually needed
        needed_outputs = set(outputs)
        # BFS from outputs to find all needed wires
        needed = set(needed_outputs)
        frontier = list(needed_outputs)
        while frontier:
            w = frontier.pop()
            if w >= n + 1:
                k = w - n - 1
                if 0 <= k < len(gates):
                    a, b = gates[k]
                    if a not in needed:
                        needed.add(a)
                        frontier.append(a)
                    if b not in needed:
                        needed.add(b)
                        frontier.append(b)
        
        # Keep only gates whose output is in needed
        new_gates = []
        new_mask_to_wire = {}
        for i in range(n):
            new_mask_to_wire[1 << i] = i
        new_mask_to_wire[0] = n
        new_wire_mask = {}
        for i in range(n):
            new_wire_mask[i] = 1 << i
        new_wire_mask[n] = 0
        remap = {}
        for k, (a, b) in enumerate(gates):
            old_idx = n + 1 + k
            if old_idx in needed:
                new_idx = n + 1 + len(new_gates)
                remap[old_idx] = new_idx
                new_gates.append([a, b])
                new_wire_mask[new_idx] = new_wire_mask[a] ^ new_wire_mask[b]
                new_mask_to_wire[new_wire_mask[new_idx]] = new_idx
        
        if len(new_gates) < len(gates):
            gates = new_gates
            changed = True
        
        # Remap outputs
        if remap:
            new_outputs = []
            for o in outputs:
                if o in remap:
                    new_outputs.append(remap[o])
                else:
                    new_outputs.append(o)
            outputs = new_outputs
            # Also remap gate inputs
            new_gates2 = []
            for a, b in gates:
                na = remap.get(a, a)
                nb = remap.get(b, b)
                new_gates2.append([na, nb])
            gates = new_gates2
    
    return gates, outputs

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 85.0  # Safety margin before 90s kill
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}
    
    # Try greedy CSE
    gates, outputs = _build_greedy_cse(matrix, n, m, DEADLINE - 10.0)
    
    # Verify
    if not _verify(gates, outputs, matrix, n):
        # Fallback to direct
        gates, outputs = _build_direct(matrix, n, m)
    
    # Local search to remove redundant gates
    if time.time() < DEADLINE - 5.0:
        gates, outputs = _local_search(gates, outputs, matrix, n, m, DEADLINE - 5.0)
    
    # Final verification
    if not _verify(gates, outputs, matrix, n):
        gates, outputs = _build_direct(matrix, n, m)
    
    return {'gates': gates, 'outputs': outputs}
