import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    mat = np.array(matrix, dtype=np.int8)
    m, n = mat.shape
    zero_wire = n

    row_weights = np.sum(mat, axis=1)
    baseline_gates = int(np.sum(np.maximum(0, row_weights - 1)))

    # Trivial: no gates needed
    if baseline_gates == 0:
        gates = []
        outputs = []
        for i in range(m):
            if row_weights[i] == 0:
                outputs.append(zero_wire)
            else:
                idx = np.where(mat[i] == 1)[0][0]
                outputs.append(int(idx))
        return {'gates': gates, 'outputs': outputs}

    # Convert rows to integer bitmasks
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if mat[i, j] == 1:
                mask |= (1 << j)
        target_masks.append(mask)

    # --- Algebraic CSE strategy ---
    # 1. Find a minimal-weight generating set (basis) for the span of targets.
    #    We want basis vectors that are "cheap" to compute from inputs.
    # 2. Express each target as XOR of basis vectors.
    # 3. Build a DAG computing all basis vectors (with their own sub-decompositions)
    #    and then combine to produce each target.

    # Step 1: Find a low-weight basis for the target space.
    # Use Gaussian elimination but prefer low-weight pivots.
    # Represent vectors as Python ints (bitmasks).
    
    # Collect unique non-zero targets
    unique_targets = list(set(mask for mask in target_masks if mask != 0))
    
    if not unique_targets:
        gates = []
        outputs = [zero_wire] * m
        return {'gates': gates, 'outputs': outputs}

    # Find basis via Gaussian elimination, preferring lower-weight vectors
    # Sort by weight to get low-weight pivots first
    sorted_targets = sorted(unique_targets, key=lambda x: bin(x).count('1'))
    
    basis = []  # list of (mask, pivot_bit)
    basis_masks = set()
    
    for mask in sorted_targets:
        if mask == 0:
            continue
        # Reduce by existing basis
        reduced = mask
        for b_mask, b_pivot in basis:
            if (reduced >> b_pivot) & 1:
                reduced ^= b_mask
        if reduced == 0:
            continue
        # Find pivot: lowest set bit
        pivot = (reduced & -reduced).bit_length() - 1
        # Normalize: make pivot the lowest set bit (already is)
        basis.append((reduced, pivot))
        basis_masks.add(reduced)
    
    # Now basis spans all targets. Express each target as XOR of basis vectors.
    # Build representation: for each target, which basis vectors XOR to give it.
    # We'll use a dict: mask -> list of basis indices (XOR of those basis vectors)
    
    # Precompute: for each basis vector, its decomposition into inputs is just itself
    # (a basis vector is a subset of input wires). But we can also decompose basis vectors
    # further if they have high weight, by finding cheaper intermediate XORs.
    
    # Actually, let's think differently. The key insight:
    # If we have a set of "available" masks (initially inputs + zero), we can build up.
    # But the algebraic approach: find a small set of "intermediate" masks such that
    # every target is an XOR of at most k intermediates, and the intermediates themselves
    # are cheap to compute.
    
    # Better approach: Recursive decomposition with global memoization.
    # For each mask we need to compute, try to express it as XOR of two "simpler" masks
    # that are already computed or cheaper to compute.
    
    # Let's use a different strategy:
    # 1. Find a basis for the target space.
    # 2. For each basis vector, if it has weight > 1, try to decompose it into
    #    XOR of two other vectors that are "closer" to inputs (lower weight or already computed).
    # 3. Use memoization: wire_map[mask] = wire_index
    # 4. Process in order of increasing "complexity" (weight).
    
    # Actually, the cleanest approach: 
    # - Start with all input wires available.
    # - Use a priority queue of "needed" masks (targets + their sub-decompositions).
    # - For each needed mask, find the cheapest way to compute it from available masks.
    # - "Cheapest" = minimum number of new gates, where a gate computes XOR of two available.
    
    # Let me implement a cleaner version:
    # Global CSE via recursive decomposition with memoization.
    
    wire_map = {}  # mask -> wire_index
    gates = []
    next_wire = n + 1
    
    # Initialize with inputs and zero
    for i in range(n):
        wire_map[1 << i] = i
    wire_map[0] = zero_wire
    
    # We need to compute all target masks.
    # Use a worklist: for each target, recursively ensure it's computed.
    
    def ensure_computed(mask):
        """Ensure mask is in wire_map. Returns wire_index."""
        if mask in wire_map:
            return wire_map[mask]
        
        # Try to find two available masks whose XOR is mask
        # Iterate over available masks (excluding 0 and single bits for efficiency)
        best_pair = None
        best_cost = float('inf')
        
        avail_keys = list(wire_map.keys())
        
        for a in avail_keys:
            if a == 0:
                continue
            b = mask ^ a
            if b in wire_map:
                # Cost: 1 gate (if both a and b are "simple" i.e., inputs or already cheap)
                # Actually cost is always 1 gate here
                # Prefer pairs where both are inputs (weight 1) for depth reasons
                cost = 1
                if best_cost > cost:
                    best_cost = cost
                    best_pair = (a, b)
                    # Prefer input-input pairs
                    if a.bit_count() == 1 and b.bit_count() == 1:
                        break
        
        if best_pair is not None:
            a, b = best_pair
            if a == 0:
                wire = wire_map[b]
            elif b == 0:
                wire = wire_map[a]
            else:
                gates.append([wire_map[a], wire_map[b]])
                wire = next_wire
                next_wire += 1
            wire_map[mask] = wire
            return wire
        
        # No direct XOR of two available. Need to decompose further.
        # Try: mask = input_i XOR rest, where rest = mask ^ (1<<i)
        # Pick the input i that makes rest have lower weight or be already available
        best_input = None
        best_rest_weight = float('inf')
        
        for i in range(n):
            bit = 1 << i
            if (mask & bit) == 0:
                continue
            rest = mask ^ bit
            if rest in wire_map:
                # rest is already computed
                gates.append([i, wire_map[rest]])
                wire = next_wire
                next_wire += 1
                wire_map[mask] = wire
                return wire
            # Otherwise, prefer rest with lower weight
            w = rest.bit_count()
            if w < best_rest_weight:
                best_rest_weight = w
                best_input = i
        
        if best_input is not None:
            # Recursively compute rest, then XOR with input
            bit = 1 << best_input
            rest = mask ^ bit
            ensure_computed(rest)
            gates.append([best_input, wire_map[rest]])
            wire = next_wire
            next_wire += 1
            wire_map[mask] = wire
            return wire
        
        # Shouldn't get here for non-zero mask
        wire = zero_wire
        wire_map[mask] = wire
        return wire
    
    # Process targets in order of increasing weight to build up complexity gradually
    # This helps because simpler targets get computed first, making them available
    # for more complex targets.
    target_indices = sorted(range(m), key=lambda i: row_weights[i])
    
    for idx in target_indices:
        if time.time() > deadline:
            break
        mask = target_masks[idx]
        if mask == 0:
            continue
        ensure_computed(mask)
    
    # Build outputs
    outputs = []
    for i in range(m):
        mask = target_masks[i]
        if mask == 0:
            outputs.append(zero_wire)
        elif mask in wire_map:
            outputs.append(wire_map[mask])
        else:
            # Fallback: shouldn't happen, but compute directly
            ones = [j for j in range(n) if (mask >> j) & 1]
            if len(ones) == 0:
                outputs.append(zero_wire)
            elif len(ones) == 1:
                outputs.append(ones[0])
            else:
                current = ones[0]
                for j in range(1, len(ones)):
                    gates.append([current, ones[j]])
                    current = next_wire
                    next_wire += 1
                outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}
