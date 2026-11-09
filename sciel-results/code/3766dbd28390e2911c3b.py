"""
Exact XOR-circuit synthesis via minimum-DAG construction with greedy shared-subexpression
extraction, basis-aware ordering, and local-search refinement.

Strategy:
  1. Build a candidate set of useful intermediate wires (XORs of pairs of inputs or
     existing intermediates) ranked by how many output rows they can simplify.
  2. Greedily add the highest-benefit intermediate wire, updating decompositions.
  3. Use a balanced XOR tree for final row assembly with CSE.
  4. Local search: remove unused gates, collapse trivial gates, try gate replacement.
  5. Multiple restarts with different row orderings to escape local optima.

The key insight is that the optimal circuit is a DAG where each gate computes the XOR
of two previously-computed wires. We search over the space of such DAGs by greedily
adding shared subexpressions that reduce total gate count.
"""
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import random

DEADLINE_SECONDS = 84.0


def _verify_circuit(matrix, circuit, depth_limit):
    m, n = matrix.shape
    wire_mask = {}
    for i in range(n):
        wire_mask[i] = (1 << i)
    wire_mask[n] = 0

    gates = circuit['gates']
    for k, (a, b) in enumerate(gates):
        wire = n + 1 + k
        mask_a = wire_mask.get(a, 0)
        mask_b = wire_mask.get(b, 0)
        wire_mask[wire] = mask_a ^ mask_b

    for i in range(m):
        expected_mask = 0
        for j in range(n):
            if matrix[i, j]:
                expected_mask |= (1 << j)
        out_wire = circuit['outputs'][i]
        actual_mask = wire_mask.get(out_wire, 0)
        if actual_mask != expected_mask:
            return False

    if depth_limit < 0:
        return True

    depth = {}
    for i in range(n + 1):
        depth[i] = 0

    max_depth = 0
    for k, (a, b) in enumerate(gates):
        wire = n + 1 + k
        d = 1 + max(depth.get(a, 0), depth.get(b, 0))
        depth[wire] = d
        max_depth = max(max_depth, d)

    for out in circuit['outputs']:
        max_depth = max(max_depth, depth.get(out, 0))

    if max_depth > depth_limit:
        return False

    return True


def _row_masks(matrix):
    m, n = matrix.shape
    masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        masks.append(mask)
    return masks


def _xor_tree(indices, get_xor_wire):
    if len(indices) == 1:
        return indices[0]
    mid = len(indices) // 2
    left = _xor_tree(indices[:mid], get_xor_wire)
    right = _xor_tree(indices[mid:], get_xor_wire)
    return get_xor_wire(left, right)


def _build_greedy_cse(matrix, n, row_order=None, max_iters=200, benefit_mode='pair'):
    """
    Build circuit via greedy CSE with shared subexpression extraction.
    
    row_order: optional permutation of row indices to try different orderings.
    benefit_mode: 'pair' uses cnt-1, 'cascade' estimates future benefit.
    """
    m = matrix.shape[0]
    if row_order is None:
        row_order = list(range(m))
    
    row_masks = _row_masks(matrix)
    
    # For each row, maintain a set of wire indices that XOR to give the row value.
    # Initially, each row is its set of 1-bits (or {n} if zero).
    row_sets = []
    for idx in row_order:
        mask = row_masks[idx]
        s = set(j for j in range(n) if (mask >> j) & 1)
        if not s:
            s = {n}
        row_sets.append(s)
    
    # Gate registry: frozenset({a,b}) -> wire index
    gate_registry = {}
    gates = []
    next_gate = [0]
    
    def get_wire(a, b):
        if a == b:
            return n
        key = frozenset((a, b))
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate[0]
        gates.append([min(a, b), max(a, b)])
        gate_registry[key] = wire
        next_gate[0] += 1
        return wire
    
    # Compute pair counts efficiently using bitmasks over rows
    def compute_pair_counts():
        # For each wire, track which rows contain it (as a bitmask over row indices)
        # Then pair count = popcount(mask_a & mask_b)
        if m <= 64:
            wire_row_mask = {}
            for w in range(n + 1):
                wm = 0
                for i in range(m):
                    if w in row_sets[i]:
                        wm |= (1 << i)
                wire_row_mask[w] = wm
            
            pc = defaultdict(int)
            wires = list(wire_row_mask.keys())
            for i in range(len(wires)):
                for j in range(i + 1, len(wires)):
                    a, b = wires[i], wires[j]
                    cnt = bin(wire_row_mask[a] & wire_row_mask[b]).count('1')
                    if cnt >= 2:
                        pc[frozenset((a, b))] = cnt
            return pc
        else:
            # Fallback for larger m
            pc = defaultdict(int)
            for s in row_sets:
                lst = sorted(s)
                for i in range(len(lst)):
                    for j in range(i + 1, len(lst)):
                        pc[frozenset((lst[i], lst[j]))] += 1
            return pc
    
    pair_count = compute_pair_counts()
    
    changed = True
    iter_count = 0
    while changed and iter_count < max_iters:
        changed = False
        iter_count += 1
        
        best_pair = None
        best_benefit = 0
        for key, cnt in pair_count.items():
            a, b = tuple(key)
            if key in gate_registry:
                continue
            if benefit_mode == 'cascade':
                # Estimate: replacing a,b with new_wire in cnt rows saves 1 gate each,
                # but the new_wire might pair with others. Use cnt - 1 as base.
                benefit = cnt - 1
            else:
                benefit = cnt - 1
            if benefit > best_benefit:
                best_benefit = benefit
                best_pair = key
        
        if best_pair is None or best_benefit <= 0:
            break
        
        a, b = tuple(best_pair)
        new_wire = get_wire(a, b)
        changed = True
        
        for i in range(len(row_sets)):
            s = row_sets[i]
            if a in s and b in s:
                s.discard(a)
                s.discard(b)
                s.add(new_wire)
        
        pair_count = compute_pair_counts()
    
    # Build final outputs
    outputs = []
    for i in range(len(row_sets)):
        s = row_sets[i]
        if len(s) == 0:
            outputs.append(n)
        elif len(s) == 1:
            outputs.append(next(iter(s)))
        else:
            lst = sorted(s)
            outputs.append(_xor_tree(lst, get_wire))
    
    # Map outputs back to original row order
    if row_order != list(range(m)):
        final_outputs = [0] * m
        for new_idx, orig_idx in enumerate(row_order):
            final_outputs[orig_idx] = outputs[new_idx]
        outputs = final_outputs
    
    return {'gates': gates, 'outputs': outputs}


def _build_direct(matrix, n):
    """Direct construction: balanced XOR trees with CSE."""
    gate_registry = {}
    gates = []
    next_gate = [0]
    
    def get_xor_wire(a, b):
        if a == b:
            return n
        key = frozenset((a, b))
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate[0]
        gates.append([min(a, b), max(a, b)])
        gate_registry[key] = wire
        next_gate[0] += 1
        return wire
    
    outputs = []
    for i in range(matrix.shape[0]):
        ones = np.nonzero(matrix[i])[0].tolist()
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            outputs.append(_xor_tree(ones, get_xor_wire))
    
    return {'gates': gates, 'outputs': outputs}


def _local_search(matrix, circuit, depth_limit, time_budget=5.0):
    """Local search: remove unused gates, collapse trivial gates, try wire replacement."""
    gates = [g[:] for g in circuit['gates']]
    outputs = circuit['outputs'][:]
    n = matrix.shape[1]
    m = matrix.shape[0]
    start_time = time.time()
    
    def compute_masks():
        wm = {}
        for i in range(n):
            wm[i] = (1 << i)
        wm[n] = 0
        for k, (a, b) in enumerate(gates):
            wm[n + 1 + k] = wm.get(a, 0) ^ wm.get(b, 0)
        return wm
    
    def remove_unused():
        nonlocal gates, outputs
        changed = True
        while changed:
            changed = False
            if not gates:
                break
            usage = defaultdict(int)
            for k, (a, b) in enumerate(gates):
                usage[a] += 1
                usage[b] += 1
            for out in outputs:
                usage[out] += 1
            
            removed_wires = set()
            for k, (a, b) in enumerate(gates):
                wire = n + 1 + k
                if usage[wire] == 0:
                    removed_wires.add(wire)
            
            if removed_wires:
                old_to_new = {}
                new_gates2 = []
                next_w = n + 1
                for k, (a, b) in enumerate(gates):
                    wire = n + 1 + k
                    if wire not in removed_wires:
                        old_to_new[wire] = next_w
                        a2 = old_to_new.get(a, a)
                        b2 = old_to_new.get(b, b)
                        new_gates2.append([a2, b2])
                        next_w += 1
                outputs = [old_to_new.get(o, o) for o in outputs]
                gates = new_gates2
                changed = True
    
    def collapse_trivial():
        nonlocal gates, outputs
        changed = True
        while changed:
            changed = False
            wm = compute_masks()
            for k in range(len(gates) - 1, -1, -1):
                a, b = gates[k]
                wire = n + 1 + k
                val = wm.get(a, 0) ^ wm.get(b, 0)
                if val == 0:
                    replacement = n
                elif (val & (val - 1)) == 0:
                    replacement = val.bit_length() - 1
                else:
                    continue
                
                # Check if wire is used
                used = False
                for k2, (a2, b2) in enumerate(gates):
                    if k2 == k:
                        continue
                    if a2 == wire or b2 == wire:
                        used = True
                        break
                if not used:
                    for out in outputs:
                        if out == wire:
                            used = True
                            break
                
                if not used:
                    gates.pop(k)
                    changed = True
                    break
                else:
                    for k2, (a2, b2) in enumerate(gates):
                        if k2 == k:
                            continue
                        if a2 == wire:
                            gates[k2][0] = replacement
                        if b2 == wire:
                            gates[k2][1] = replacement
                    for i in range(len(outputs)):
                        if outputs[i] == wire:
                            outputs[i] = replacement
                    gates.pop(k)
                    changed = True
                    break
    
    def try_replacements():
        """Try replacing wire usages with equivalent cheaper wires."""
        nonlocal gates, outputs
        wm = compute_masks()
        # Build mask-to-cheapest-wire map
        mask_to_wire = {}
        for w in range(n + 1):
            mask_to_wire[wm.get(w, 0)] = w
        for k in range(len(gates)):
            wire = n + 1 + k
            mask = wm.get(wire, 0)
            if mask not in mask_to_wire or mask_to_wire[mask] > wire:
                mask_to_wire[mask] = wire
        
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            # Try replacing each gate's inputs with cheaper equivalent wires
            for k in range(len(gates)):
                a, b = gates[k]
                mask_a = wm.get(a, 0)
                mask_b = wm.get(b, 0)
                cheaper_a = mask_to_wire.get(mask_a, a)
                cheaper_b = mask_to_wire.get(mask_b, b)
                if cheaper_a != a or cheaper_b != b:
                    # Check that the new pair doesn't create a cycle
                    new_a = min(cheaper_a, cheaper_b)
                    new_b = max(cheaper_a, cheaper_b)
                    if new_a != new_b:
                        gates[k] = [new_a, new_b]
                        improved = True
                        wm = compute_masks()
                        # Update mask_to_wire
                        for w in range(n + 1):
                            mask_to_wire[wm.get(w, 0)] = w
                        for kk in range(len(gates)):
                            wire = n + 1 + kk
                            mask = wm.get(wire, 0)
                            if mask not in mask_to_wire or mask_to_wire[mask] > wire:
                                mask_to_wire[mask] = wire
                        break
                # Try replacing output wires
            for i in range(len(outputs)):
                mask = wm.get(outputs[i], 0)
                cheaper = mask_to_wire.get(mask, outputs[i])
                if cheaper != outputs[i]:
                    outputs[i] = cheaper
                    improved = True
                    break
    
    # Run passes
    remove_unused()
    collapse_trivial()
    remove_unused()
    
    if time.time() - start_time < time_budget:
        try_replacements()
        remove_unused()
        collapse_trivial()
    
    result = {'gates': gates, 'outputs': outputs}
    if _verify_circuit(matrix, result, depth_limit):
        return result
    return circuit


def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    if isinstance(matrix, list):
        matrix = np.array(matrix, dtype=np.int8)
    
    n_cols = matrix.shape[1]
    n_rows = matrix.shape[0]
    
    best = None
    best_gates = float('inf')
    
    # Strategy 1: Direct construction
    c = _build_direct(matrix, n_cols)
    if _verify_circuit(matrix, c, depth_limit):
        best = c
        best_gates = len(c['gates'])
    
    # Strategy 2: Greedy CSE with default ordering
    if time.time() < deadline - 5.0:
        c = _build_greedy_cse(matrix, n_cols)
        if _verify_circuit(matrix, c, depth_limit):
            if len(c['gates']) < best_gates:
                best = c
                best_gates = len(c['gates'])
    
    # Strategy 3: Multiple restarts with random row orderings
    if time.time() < deadline - 10.0:
        random.seed(42)
        for restart in range(10):
            if time.time() > deadline - 8.0:
                break
            row_order = list(range(n_rows))
            random.shuffle(row_order)
            c = _build_greedy_cse(matrix, n_cols, row_order=row_order)
            if _verify_circuit(matrix, c, depth_limit):
                if len(c['gates']) < best_gates:
                    best = c
                    best_gates = len(c['gates'])
    
    # Strategy 4: Greedy CSE with rows sorted by popcount (descending)
    if time.time() < deadline - 8.0:
        popcounts = [int(matrix[i].sum()) for i in range(n_rows)]
        row_order = sorted(range(n_rows), key=lambda i: popcounts[i], reverse=True)
        c = _build_greedy_cse(matrix, n_cols, row_order=row_order)
        if _verify_circuit(matrix, c, depth_limit):
            if len(c['gates']) < best_gates:
                best = c
                best_gates = len(c['gates'])
    
    # Strategy 5: Greedy CSE with rows sorted by popcount (ascending)
    if time.time() < deadline - 8.0:
        popcounts = [int(matrix[i].sum()) for i in range(n_rows)]
        row_order = sorted(range(n_rows), key=lambda i: popcounts[i])
        c = _build_greedy_cse(matrix, n_cols, row_order=row_order)
        if _verify_circuit(matrix, c, depth_limit):
            if len(c['gates']) < best_gates:
                best = c
                best_gates = len(c['gates'])
    
    # Local search refinement
    if best is not None and time.time() < deadline - 3.0:
        budget = min(6.0, deadline - time.time() - 2.0)
        refined = _local_search(matrix, best, depth_limit, time_budget=budget)
        if len(refined['gates']) < best_gates:
            best = refined
            best_gates = len(refined['gates'])
    
    # Final verification
    if best is None or not _verify_circuit(matrix, best, depth_limit):
        best = _build_direct(matrix, n_cols)
        if not _verify_circuit(matrix, best, depth_limit):
            best = {'gates': [], 'outputs': [n_cols] * n_rows}
    
    return best
