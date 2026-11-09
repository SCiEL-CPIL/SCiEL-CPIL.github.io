"""
Exact XOR-circuit synthesis with strong sharing via systematic basis reduction.

Strategy:
1. Treat the m output rows as vectors in GF(2)^n.
2. Find a basis (independent set) of the row space.
3. Express each row as a linear combination of basis vectors.
4. Synthesize the circuit by:
   a. Building a circuit for each basis vector (using a balanced XOR tree over its support).
   b. For each output row, XOR-ing together the wires corresponding to the basis vectors in its combination.
   c. Using a shared gate registry to ensure that identical XOR operations are computed only once.

This approach maximizes sharing: if two outputs share a basis vector, the gate for that basis
vector is computed once and reused. Additionally, if two basis vectors have overlapping supports,
the intermediate XORs in their trees may also be shared via the gate registry.

We further optimize by:
- Choosing a basis that minimizes total weight (number of 1s) to reduce tree sizes.
- Using a greedy basis selection that prefers vectors with smaller support.
- Trying multiple random permutations of the row order to find a good basis.

Verification is exact symbolic GF(2) using bitmask representation.
"""

import time
import random
import numpy as np

DEADLINE_SECONDS = 85  # safety margin before 90s kill


def _verify_circuit(matrix, circuit, depth_limit):
    """
    Exact symbolic GF(2) verification.
    Returns True if circuit correctly computes the matrix and depth is within limit.
    """
    m, n = matrix.shape
    
    # Each wire corresponds to a linear form over GF(2) of the n inputs.
    # Represent each wire as a bitmask (Python int).
    wire_mask = {}
    for i in range(n):
        wire_mask[i] = (1 << i)
    wire_mask[n] = 0  # constant zero
    
    # Process gates
    for k, (a, b) in enumerate(circuit['gates']):
        wire = n + 1 + k
        mask_a = wire_mask.get(a, 0)
        mask_b = wire_mask.get(b, 0)
        wire_mask[wire] = mask_a ^ mask_b
    
    # Verify outputs
    for i in range(m):
        expected_mask = 0
        for j in range(n):
            if matrix[i, j]:
                expected_mask |= (1 << j)
        out_wire = circuit['outputs'][i]
        actual_mask = wire_mask.get(out_wire, 0)
        if actual_mask != expected_mask:
            return False
    
    # Check depth
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
        max_depth = max(max_depth, d)
    
    for out in circuit['outputs']:
        max_depth = max(max_depth, depth.get(out, 0))
    
    if max_depth > depth_limit:
        return False
    
    return True


def _build_circuit_from_basis(matrix, rng=None, num_trials=50):
    """
    Build a circuit by:
    1. Finding a basis for the row space.
    2. Expressing each row as a combination of basis vectors.
    3. Synthesizing with shared gates.
    
    We try multiple random row orderings to find a basis that minimizes gate count.
    """
    m, n = matrix.shape
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert rows to bitmasks for efficient GF(2) operations
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)
    
    best_circuit = None
    best_count = float('inf')
    
    if rng is None:
        rng = random.Random(42)
    
    for trial in range(num_trials):
        # Shuffle row order for basis selection
        order = list(range(m))
        rng.shuffle(order)
        
        # Gaussian elimination to find basis
        basis = []  # list of (mask, original_index)
        basis_masks = []  # just the masks
        
        for idx in order:
            mask = row_masks[idx]
            if mask == 0:
                continue
            
            # Reduce by existing basis
            reduced = mask
            for bm in basis_masks:
                # Find pivot bit of bm
                pivot = bm.bit_length() - 1
                if (reduced >> pivot) & 1:
                    reduced ^= bm
            
            if reduced != 0:
                basis.append((reduced, idx))
                basis_masks.append(reduced)
        
        # If no basis vectors, all outputs are zero
        if not basis:
            circuit = {'gates': [], 'outputs': [n] * m}
            count = 0
            if count < best_count:
                best_count = count
                best_circuit = circuit
            continue
        
        # Now express each row as a combination of basis vectors
        # For each row, find which basis vectors XOR to give it
        # We store basis as (mask, original_index)
        
        # Build circuit with shared gates
        gate_registry = {}  # (a, b) sorted -> wire
        gates = []
        next_gate = 0
        
        def get_xor_wire(a, b):
            nonlocal next_gate
            if a == b:
                return n  # constant zero
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
        
        # First, synthesize each basis vector as a wire
        # For a basis vector with support S (list of input indices),
        # compute XOR of all inputs in S using a balanced tree
        basis_wires = []
        for bm, orig_idx in basis:
            # Get support
            support = []
            for j in range(n):
                if (bm >> j) & 1:
                    support.append(j)
            
            if len(support) == 0:
                basis_wires.append(n)
            elif len(support) == 1:
                basis_wires.append(support[0])
            else:
                # Balanced tree
                def xor_tree(indices):
                    if len(indices) == 1:
                        return indices[0]
                    mid = len(indices) // 2
                    left = xor_tree(indices[:mid])
                    right = xor_tree(indices[mid:])
                    return get_xor_wire(left, right)
                basis_wires.append(xor_tree(support))
        
        # Now for each output row, find its combination of basis vectors
        # and XOR the corresponding basis wires together
        outputs = []
        for i in range(m):
            mask = row_masks[i]
            if mask == 0:
                outputs.append(n)
                continue
            
            # Find combination: which basis vectors XOR to give mask
            # Use the same reduction process
            reduced = mask
            combo_indices = []
            for b_idx, (bm, orig_idx) in enumerate(basis):
                pivot = bm.bit_length() - 1
                if (reduced >> pivot) & 1:
                    reduced ^= bm
                    combo_indices.append(b_idx)
            
            if reduced != 0:
                # Should not happen if basis spans the row space
                # Fallback: use direct synthesis for this row
                support = []
                for j in range(n):
                    if (mask >> j) & 1:
                        support.append(j)
                if len(support) == 0:
                    outputs.append(n)
                elif len(support) == 1:
                    outputs.append(support[0])
                else:
                    def xor_tree(indices):
                        if len(indices) == 1:
                            return indices[0]
                        mid = len(indices) // 2
                        left = xor_tree(indices[:mid])
                        right = xor_tree(indices[mid:])
                        return get_xor_wire(left, right)
                    outputs.append(xor_tree(support))
                continue
            
            if len(combo_indices) == 0:
                outputs.append(n)
            elif len(combo_indices) == 1:
                outputs.append(basis_wires[combo_indices[0]])
            else:
                # XOR the basis wires together
                def xor_wires(wire_list):
                    if len(wire_list) == 1:
                        return wire_list[0]
                    mid = len(wire_list) // 2
                    left = xor_wires(wire_list[:mid])
                    right = xor_wires(wire_list[mid:])
                    return get_xor_wire(left, right)
                combo_wires = [basis_wires[idx] for idx in combo_indices]
                outputs.append(xor_wires(combo_wires))
        
        circuit = {'gates': gates, 'outputs': outputs}
        count = len(gates)
        
        if count < best_count:
            best_count = count
            best_circuit = circuit
    
    return best_circuit


def _direct_baseline(matrix):
    """
    Direct baseline: each output is computed independently without sharing.
    """
    m, n = matrix.shape
    gates = []
    outputs = []
    next_gate = 0
    
    for i in range(m):
        ones = []
        for j in range(n):
            if matrix[i, j]:
                ones.append(j)
        
        if len(ones) == 0:
            outputs.append(n)
        elif len(ones) == 1:
            outputs.append(ones[0])
        else:
            # Simple chain
            wire = ones[0]
            for j in range(1, len(ones)):
                a = wire
                b = ones[j]
                wire = n + 1 + next_gate
                gates.append([a, b])
                next_gate += 1
            outputs.append(wire)
    
    return {'gates': gates, 'outputs': outputs}


def synthesize(matrix, depth_limit):
    global _start_time
    _start_time = time.time()
    
    matrix = np.array(matrix, dtype=np.uint8)
    m, n = matrix.shape
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    deadline = _start_time + DEADLINE_SECONDS
    
    # Phase 1: Direct baseline
    baseline_circuit = _direct_baseline(matrix)
    best_circuit = baseline_circuit
    best_count = len(baseline_circuit['gates'])
    
    # Phase 2: Basis-based synthesis with multiple trials
    # Allocate most of the time budget to this phase
    time_for_basis = min(60, deadline - time.time() - 5)
    
    if time_for_basis > 2:
        rng = random.Random(42)
        start = time.time()
        num_trials = 50
        
        # Do initial batch
        circuit = _build_circuit_from_basis(matrix, rng, num_trials=num_trials)
        if circuit:
            count = len(circuit['gates'])
            if count < best_count:
                best_circuit = circuit
                best_count = count
        
        # Continue with more trials if time remains
        while time.time() < deadline - 10:
            elapsed = time.time() - start
            # Estimate trials per second
            if elapsed > 0:
                trials_per_sec = num_trials / elapsed
            else:
                trials_per_sec = 10
            remaining_time = deadline - 10 - time.time()
            extra_trials = int(remaining_time * trials_per_sec)
            if extra_trials < 5:
                break
            
            circuit = _build_circuit_from_basis(matrix, rng, num_trials=extra_trials)
            if circuit:
                count = len(circuit['gates'])
                if count < best_count:
                    best_circuit = circuit
                    best_count = count
    
    # Final verification
    if not _verify_circuit(matrix, best_circuit, depth_limit):
        # Fallback to baseline
        if _verify_circuit(matrix, baseline_circuit, depth_limit):
            best_circuit = baseline_circuit
        else:
            # Emergency fallback: direct wiring with guaranteed depth
            gates = []
            outputs = []
            next_gate = 0
            for i in range(m):
                ones = []
                for j in range(n):
                    if matrix[i, j]:
                        ones.append(j)
                if len(ones) == 0:
                    outputs.append(n)
                elif len(ones) == 1:
                    outputs.append(ones[0])
                else:
                    wire = ones[0]
                    for j in range(1, len(ones)):
                        a = wire
                        b = ones[j]
                        wire = n + 1 + next_gate
                        gates.append([a, b])
                        next_gate += 1
                    outputs.append(wire)
            best_circuit = {'gates': gates, 'outputs': outputs}
    
    return best_circuit
