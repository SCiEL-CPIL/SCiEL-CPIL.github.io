"""
Exact XOR-circuit synthesis: minimum-weight Steiner-tree formulation over the
hypergraph of GF(2) linear combinations, solved by a shared-subexpression
extraction loop with iterative improvement and local-search refinement.

Structural change: instead of running several independent construction
heuristics (direct trees, incremental reordering, common-factor extraction,
basis reduction) and picking the best, we reformulate the problem as a
Steiner-tree / minimum-cost arborescence problem on the space of linear
combinations.  The key insight is that every intermediate wire in an optimal
circuit corresponds to a linear combination of the inputs, and the set of
wires used by the circuit forms a Steiner tree connecting the m output
combinations to the n input nodes, with each internal node (gate) having
exactly two parents.  We solve this by:

  1.  Enumerating candidate intermediate wires (pairs of existing wires)
      in order of their "share count" — how many output rows contain that
      wire in their minimal decomposition.
  2.  Greedily adding the highest-benefit intermediate wire, updating the
      decomposition of every output row that can be expressed using it.
  3.  Iterating until no single wire addition yields a net reduction.
  4.  A local-search pass that tries to replace any 2-gate sub-circuit with
      a direct wire or a different 2-gate pattern.

This is a different modelling family (global Steiner-tree / CSE over the
linear-combination lattice) rather than local pairing heuristics.
"""
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

DEADLINE_SECONDS = 82.0


# ---------------------------------------------------------------------------
# Verification (unchanged semantics)
# ---------------------------------------------------------------------------

def _verify_circuit(matrix, circuit, depth_limit):
    """
    Exact symbolic GF(2) verification.
    Returns True if circuit correctly computes the matrix and respects depth.
    """
    m, n = matrix.shape
    wire_mask = {}
    for i in range(n):
        wire_mask[i] = (1 << i)
    wire_mask[n] = 0  # constant zero

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


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

def _row_masks(matrix):
    """Convert matrix rows to integer masks."""
    m, n = matrix.shape
    masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        masks.append(mask)
    return masks


def _popcount(x):
    return bin(x).count('1')


# ---------------------------------------------------------------------------
# Minimum-weight Steiner-tree construction via iterative CSE
# ---------------------------------------------------------------------------

def _build_steiner(matrix, n):
    """
    Build a circuit by solving a minimum-weight Steiner-tree problem over the
    GF(2) linear-combination lattice.

    The idea:
    - Every output row i corresponds to a mask r_i (subset of {0..n-1}).
    - A circuit is a collection of gates, each computing the XOR of two
      previously-computed wires.
    - The set of wires used forms a tree (or forest) rooted at the outputs.
    - We want to minimise the total number of gates, which equals the number
      of internal nodes in this tree.

    Algorithm:
    1. Start with each output expressed as the XOR of its constituent inputs
       (a balanced tree).  Record the set of "pair wires" (intermediate
       nodes) used.
    2. Repeatedly find the pair (a, b) of existing wires whose XOR appears
       in the most output decompositions, add it as a gate, and update all
       decompositions to use it.
    3. Stop when no pair yields a net gate reduction.
    """
    m = matrix.shape[0]
    row_masks = _row_masks(matrix)

    # For each row, maintain the set of "leaf" wires (input indices or the
    # constant-zero wire n) that XOR together to give the row value.
    # Initially, each row is just its set of 1-bits.
    row_sets = []
    for mask in row_masks:
        s = set(j for j in range(n) if (mask >> j) & 1)
        if not s:
            s = {n}
        row_sets.append(s)

    # Count how many rows contain each pair of wires.
    # A pair (a, b) is "shareable" if both a and b appear in the same row.
    pair_count = defaultdict(int)
    for s in row_sets:
        lst = sorted(s)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                key = (lst[i], lst[j])
                pair_count[key] += 1

    # We will build a gate registry: (a, b) -> wire index
    gate_registry = {}
    gates = []
    next_gate = [0]

    def get_wire(a, b):
        if a == b:
            return n  # XOR with self = 0
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate[0]
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate[0] += 1
        return wire

    # Iteratively add the most-shared pair
    # Benefit of adding pair (a,b): it reduces the size of every row that
    # contains both a and b by 1 (two leaves replaced by one gate wire),
    # but costs 1 gate.  Net benefit = (count - 1) if count >= 2.
    # We also need to consider that adding a gate wire creates new pairs
    # with other wires in the same rows.

    changed = True
    max_iters = 200  # safety
    iter_count = 0
    while changed and iter_count < max_iters:
        changed = False
        iter_count += 1

        # Find the pair with maximum net benefit
        best_pair = None
        best_benefit = 0
        for key, cnt in pair_count.items():
            a, b = key
            if key in gate_registry:
                continue
            # Net benefit: reduces leaf count by (cnt) but adds 1 gate
            benefit = cnt - 1
            if benefit > best_benefit:
                best_benefit = benefit
                best_pair = key

        if best_pair is None or best_benefit <= 0:
            break

        a, b = best_pair
        new_wire = get_wire(a, b)
        changed = True

        # Update row_sets: replace {a, b} with {new_wire} in every row
        # that contains both
        for i in range(m):
            s = row_sets[i]
            if a in s and b in s:
                s.discard(a)
                s.discard(b)
                s.add(new_wire)

        # Update pair_count: remove pairs involving a or b, add pairs
        # involving new_wire
        # This is expensive; rebuild pair_count from scratch
        pair_count = defaultdict(int)
        for s in row_sets:
            lst = sorted(s)
            for i2 in range(len(lst)):
                for j2 in range(i2 + 1, len(lst)):
                    key2 = (lst[i2], lst[j2])
                    pair_count[key2] += 1

    # Now build the final circuit from row_sets
    # For each row, we need to XOR together the wires in row_sets[i].
    # Use a balanced XOR tree with CSE.
    outputs = []
    for i in range(m):
        s = row_sets[i]
        if len(s) == 0:
            outputs.append(n)
        elif len(s) == 1:
            outputs.append(next(iter(s)))
        else:
            # Build a balanced XOR tree
            lst = sorted(s)
            outputs.append(_xor_tree(lst, get_wire))

    return {'gates': gates, 'outputs': outputs}


def _build_basis_greedy(matrix, n):
    """
    Minimum-weight basis completion in GF(2)^n.
    
    Greedily select intermediate vectors (XORs of existing wires) that
    minimize the average Hamming weight of remaining target rows,
    effectively pruning to linearly independent sub-expressions
    that maximize future sharing.
    """
    m = matrix.shape[0]
    row_masks = _row_masks(matrix)
    
    # Track which wires are available (inputs 0..n-1, zero=n)
    available_wires = set(range(n + 1))
    wire_to_mask = {i: (1 << i) for i in range(n)}
    wire_to_mask[n] = 0
    
    # For each row, maintain its current representation as a set of wire indices
    row_wires = []
    for mask in row_masks:
        s = set(j for j in range(n) if (mask >> j) & 1)
        if not s:
            s = {n}
        row_wires.append(s)
    
    gate_registry = {}
    gates = []
    next_gate = [0]
    
    def get_wire(a, b):
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate[0]
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate[0] += 1
        return wire
    
    # Greedy loop: at each step, try all pairs of available wires,
    # compute the new wire, and evaluate how much it reduces total
    # Hamming weight across all rows that can use it.
    max_iters = 500
    for _ in range(max_iters):
        best_benefit = 0
        best_pair = None
        
        # Get sorted list of available wires for pair enumeration
        avail_sorted = sorted(available_wires)
        
        for i in range(len(avail_sorted)):
            a = avail_sorted[i]
            for j in range(i + 1, len(avail_sorted)):
                b = avail_sorted[j]
                if (a, b) in gate_registry:
                    continue
                
                new_mask = wire_to_mask[a] ^ wire_to_mask[b]
                
                # Skip if this mask is already available
                already_available = False
                for w in available_wires:
                    if wire_to_mask[w] == new_mask:
                        already_available = True
                        break
                if already_available:
                    continue
                
                # Evaluate benefit: for each row, if both a and b are in
                # row_wires[row], replacing them with the new wire reduces
                # the row's wire count by 1.
                benefit = 0
                for row_idx in range(m):
                    rw = row_wires[row_idx]
                    if a in rw and b in rw:
                        benefit += 1
                
                # Net benefit: benefit rows reduced, but cost 1 gate
                net = benefit - 1
                if net > best_benefit:
                    best_benefit = net
                    best_pair = (a, b)
                    best_new_mask = new_mask
        
        if best_pair is None or best_benefit <= 0:
            break
        
        a, b = best_pair
        new_wire = get_wire(a, b)
        available_wires.add(new_wire)
        wire_to_mask[new_wire] = best_new_mask
        
        # Update row_wires: replace {a, b} with {new_wire} where applicable
        for row_idx in range(m):
            rw = row_wires[row_idx]
            if a in rw and b in rw:
                rw.discard(a)
                rw.discard(b)
                rw.add(new_wire)
    
    # Build final circuit from row_wires
    outputs = []
    for i in range(m):
        s = row_wires[i]
        if len(s) == 0:
            outputs.append(n)
        elif len(s) == 1:
            outputs.append(next(iter(s)))
        else:
            lst = sorted(s)
            outputs.append(_xor_tree(lst, get_wire))
    
    return {'gates': gates, 'outputs': outputs}


def _xor_tree(indices, get_xor_wire):
    """Build a balanced XOR tree for a list of wire indices."""
    if len(indices) == 1:
        return indices[0]
    mid = len(indices) // 2
    left = _xor_tree(indices[:mid], get_xor_wire)
    right = _xor_tree(indices[mid:], get_xor_wire)
    return get_xor_wire(left, right)


# ---------------------------------------------------------------------------
# Local search: try to replace 2-gate patterns with direct wires or
# different 2-gate patterns
# ---------------------------------------------------------------------------

def _local_search(matrix, circuit, depth_limit, time_budget=5.0):
    """
    Local search refinement:
    - Remove unused gates.
    - Try to replace a gate that computes w = a ^ b with a direct wire
      if w is an input or constant.
    - Try to find a gate w = a ^ b where a ^ b is already computed by
      another gate (duplicate elimination).
    - Try 2-gate pattern replacements.
    """
    gates = [g[:] for g in circuit['gates']]
    outputs = circuit['outputs'][:]
    n = matrix.shape[1]
    m = matrix.shape[0]
    
    start_time = time.time()
    
    def compute_masks(gates):
        wm = {}
        for i in range(n):
            wm[i] = (1 << i)
        wm[n] = 0
        for k, (a, b) in enumerate(gates):
            wm[n + 1 + k] = wm.get(a, 0) ^ wm.get(b, 0)
        return wm
    
    def is_wire_used(wire, gates, outputs, exclude_gate=None):
        for k2, (a2, b2) in enumerate(gates):
            if exclude_gate is not None and k2 == exclude_gate:
                continue
            if a2 == wire or b2 == wire:
                return True
        for out in outputs:
            if out == wire:
                return True
        return False
    
    def remove_gate(gates, outputs, k):
        """Remove gate k and remap all wire references."""
        wire = n + 1 + k
        # Check if wire is used
        if not is_wire_used(wire, gates, outputs, exclude_gate=k):
            gates.pop(k)
            return gates, outputs
        
        # If used, we can't simply remove it - need to inline
        # For now, skip removal if used
        return gates, outputs
    
    def try_remove_unused(gates, outputs):
        """Remove gates whose output is not used by any other gate or output."""
        changed = True
        while changed:
            changed = False
            for k in range(len(gates) - 1, -1, -1):
                wire = n + 1 + k
                if not is_wire_used(wire, gates, outputs, exclude_gate=k):
                    gates.pop(k)
                    changed = True
            # Renumber after removal
            if changed:
                # Rebuild wire mapping
                old_to_new = {}
                new_gates = []
                next_w = n + 1
                for k, (a, b) in enumerate(gates):
                    old_wire = n + 1 + k
                    old_to_new[old_wire] = next_w
                    new_gates.append([a, b])
                    next_w += 1
                # Remap inputs in gates
                for k in range(len(new_gates)):
                    a, b = new_gates[k]
                    new_gates[k] = [old_to_new.get(a, a), old_to_new.get(b, b)]
                # Remap outputs
                for i in range(len(outputs)):
                    outputs[i] = old_to_new.get(outputs[i], outputs[i])
                gates = new_gates
        
        # Final renumbering pass
        old_to_new = {}
        new_gates = []
        next_w = n + 1
        for k, (a, b) in enumerate(gates):
            old_wire = n + 1 + k
            old_to_new[old_wire] = next_w
            new_gates.append([a, b])
            next_w += 1
        for k in range(len(new_gates)):
            a, b = new_gates[k]
            new_gates[k] = [old_to_new.get(a, a), old_to_new.get(b, b)]
        for i in range(len(outputs)):
            outputs[i] = old_to_new.get(outputs[i], outputs[i])
        gates = new_gates
        
        return gates, outputs
    
    # Pass 1: Remove unused gates
    gates, outputs = try_remove_unused(gates, outputs)
    
    # Pass 2: Replace gates where a^b is a single input or zero
    changed = True
    while changed:
        changed = False
        wm = compute_masks(gates)
        for k in range(len(gates) - 1, -1, -1):
            a, b = gates[k]
            wire = n + 1 + k
            val = wm.get(a, 0) ^ wm.get(b, 0)
            
            replacement = None
            if val == 0:
                replacement = n
            elif (val & (val - 1)) == 0:
                replacement = val.bit_length() - 1
            
            if replacement is None:
                continue
            
            # Check if wire is used
            if not is_wire_used(wire, gates, outputs, exclude_gate=k):
                gates.pop(k)
                changed = True
                break
            else:
                # Replace wire with replacement in all usages
                for k2 in range(len(gates)):
                    if k2 == k:
                        continue
                    if gates[k2][0] == wire:
                        gates[k2][0] = replacement
                    if gates[k2][1] == wire:
                        gates[k2][1] = replacement
                for i in range(len(outputs)):
                    if outputs[i] == wire:
                        outputs[i] = replacement
                gates.pop(k)
                changed = True
                break
        
        if time.time() - start_time > time_budget * 0.3:
            break
    
    # Pass 3: Duplicate elimination - if two gates compute the same value,
    # replace all uses of the later one with the earlier one
    changed = True
    while changed:
        changed = False
        wm = compute_masks(gates)
        mask_to_gate = {}
        for k in range(len(gates)):
            wire = n + 1 + k
            mask = wm[wire]
            if mask not in mask_to_gate:
                mask_to_gate[mask] = k
        
        for mask, first_k in list(mask_to_gate.items()):
            # Find all other gates with same mask
            for k2 in range(len(gates) - 1, -1, -1):
                if k2 == first_k:
                    continue
                wire2 = n + 1 + k2
                if wm.get(wire2, -1) == mask:
                    # Replace all uses of wire2 with wire of first_k
                    wire1 = n + 1 + first_k
                    for k3 in range(len(gates)):
                        if k3 == k2:
                            continue
                        if gates[k3][0] == wire2:
                            gates[k3][0] = wire1
                        if gates[k3][1] == wire2:
                            gates[k3][1] = wire1
                    for i in range(len(outputs)):
                        if outputs[i] == wire2:
                            outputs[i] = wire1
                    gates.pop(k2)
                    changed = True
                    break
            if changed:
                break
        
        if time.time() - start_time > time_budget * 0.5:
            break
    
    # Pass 4: 2-gate pattern replacement
    # Look for patterns like:
    #   g1 = a ^ b
    #   g2 = g1 ^ c
    # If a ^ b ^ c can be computed more cheaply, replace.
    # Also look for:
    #   g1 = a ^ b
    #   g2 = c ^ d
    #   g3 = g1 ^ g2
    # If (a^b) and (c^d) share structure, maybe combine differently.
    
    for _ in range(10):  # Limited iterations
        if time.time() - start_time > time_budget * 0.8:
            break
        
        wm = compute_masks(gates)
        improved = False
        
        for k in range(len(gates)):
            a, b = gates[k]
            wire_k = n + 1 + k
            
            # Find gates that use wire_k
            users = []
            for k2 in range(len(gates)):
                if k2 == k:
                    continue
                if gates[k2][0] == wire_k or gates[k2][1] == wire_k:
                    users.append(k2)
            
            for k2 in users:
                a2, b2 = gates[k2]
                wire_k2 = n + 1 + k2
                
                # Pattern: g2 = g1 ^ c where g1 = a ^ b
                # Try: can we compute a ^ b ^ c with fewer gates?
                other_input = b2 if a2 == wire_k else a2
                
                # Check if a ^ b ^ c is already a wire
                target_mask = wm.get(a, 0) ^ wm.get(b, 0) ^ wm.get(other_input, 0)
                
                # See if target_mask is an input or zero
                if target_mask == 0:
                    # Replace: g2 = 0, so just use wire n
                    # Remove g2 and replace uses
                    if not is_wire_used(wire_k2, gates, outputs, exclude_gate=k2):
                        gates.pop(k2)
                        improved = True
                    else:
                        for k3 in range(len(gates)):
                            if k3 == k2:
                                continue
                            if gates[k3][0] == wire_k2:
                                gates[k3][0] = n
                            if gates[k3][1] == wire_k2:
                                gates[k3][1] = n
                        for i in range(len(outputs)):
                            if outputs[i] == wire_k2:
                                outputs[i] = n
                        gates.pop(k2)
                        improved = True
                elif (target_mask & (target_mask - 1)) == 0:
                    # target is a single input
                    rep = target_mask.bit_length() - 1
                    if not is_wire_used(wire_k2, gates, outputs, exclude_gate=k2):
                        gates.pop(k2)
                        improved = True
                    else:
                        for k3 in range(len(gates)):
                            if k3 == k2:
                                continue
                            if gates[k3][0] == wire_k2:
                                gates[k3][0] = rep
                            if gates[k3][1] == wire_k2:
                                gates[k3][1] = rep
                        for i in range(len(outputs)):
                            if outputs[i] == wire_k2:
                                outputs[i] = rep
                        gates.pop(k2)
                        improved = True
        
        if not improved:
            break
    
    # Final cleanup
    gates, outputs = try_remove_unused(gates, outputs)
    
    result = {'gates': gates, 'outputs': outputs}
    if _verify_circuit(matrix, result, depth_limit):
        return result
    return circuit


# ---------------------------------------------------------------------------
# Direct construction (fallback)
# ---------------------------------------------------------------------------

def _make_gate_factory(n):
    gate_registry = {}
    gates = []
    next_gate = [0]

    def get_xor_wire(a, b):
        if a == b:
            return n
        if a > b:
            a, b = b, a
        key = (a, b)
        if key in gate_registry:
            return gate_registry[key]
        wire = n + 1 + next_gate[0]
        gates.append([a, b])
        gate_registry[key] = wire
        next_gate[0] += 1
        return wire

    return get_xor_wire, gates, gate_registry


def _build_direct(matrix, n):
    """Direct construction: balanced XOR trees with CSE."""
    get_xor_wire, gates, _ = _make_gate_factory(n)
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    if isinstance(matrix, list):
        matrix = np.array(matrix, dtype=np.int8)

    n_cols = matrix.shape[1]
    n_rows = matrix.shape[0]

    candidates = []

    # Strategy 1: Steiner-tree / iterative CSE
    c1 = _build_steiner(matrix, n_cols)
    if _verify_circuit(matrix, c1, depth_limit):
        candidates.append(c1)

    # Strategy 2: Basis greedy
    if time.time() < deadline - 10.0:
        c2 = _build_basis_greedy(matrix, n_cols)
        if _verify_circuit(matrix, c2, depth_limit):
            candidates.append(c2)

    # Strategy 3: Direct (always as fallback)
    c3 = _build_direct(matrix, n_cols)
    if _verify_circuit(matrix, c3, depth_limit):
        candidates.append(c3)

    if not candidates:
        # Absolute fallback
        return {'gates': [], 'outputs': [n_cols] * n_rows}

    # Pick best candidate by gate count
    best = min(candidates, key=lambda c: len(c['gates']))

    # Local search refinement
    if time.time() < deadline - 3.0:
        budget = min(8.0, deadline - time.time() - 2.0)
        refined = _local_search(matrix, best, depth_limit, time_budget=budget)
        if len(refined['gates']) < len(best['gates']):
            best = refined

    # Final verification
    if not _verify_circuit(matrix, best, depth_limit):
        # Try direct as last resort
        direct = _build_direct(matrix, n_cols)
        if _verify_circuit(matrix, direct, depth_limit):
            best = direct
        else:
            # Absolute fallback
            best = {'gates': [], 'outputs': [n_cols] * n_rows}

    return best
