"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
import random

def synthesize(matrix, depth_limit):
    """
    Synthesize an XOR circuit for the given binary matrix.
    
    Args:
        matrix: 2D binary array (n_rows x n_cols) where each row represents an output function
                of the input variables (0..n_cols-1).
        depth_limit: Maximum allowed depth of the circuit.
                     
    Returns:
        dict with keys:
            'gates': list of [wire1, wire2] pairs, where each gate computes XOR of wire1 and wire2.
            'outputs': list of wire indices corresponding to the outputs of the circuit.
    """
    start_time = time.time()
    deadline = start_time + 55.0
    
    A = np.array(matrix, dtype=np.int8)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    n_rows, n_cols = A.shape
    n_inputs = n_cols
    zero_wire = n_inputs
    
    if n_rows == 0 or n_cols == 0:
        return {'gates': [], 'outputs': [zero_wire] * n_rows}
    
    row_popcounts = A.sum(axis=1)
    zero_rows = set(np.where(row_popcounts == 0)[0])
    
    def elapsed_ok():
        return time.time() < deadline
    
    def greedy_optimize(row_sets, depth_limit, max_iter=40000):
        """
        Greedily optimize computation of a set of row representations.
        Each row is a set of wire indices. Returns (gates, final_row_wires, next_wire, wire_set, wire_depth).
        """
        gates = []
        next_wire = zero_wire + 1
        
        ws = {}
        wd = {}
        for i in range(n_inputs):
            ws[i] = frozenset([i])
            wd[i] = 0
        ws[zero_wire] = frozenset()
        wd[zero_wire] = 0
        
        # Canonical form: map frozenset -> wire index (for merging identical wires)
        canonical = {}
        for i in range(n_inputs):
            canonical[frozenset([i])] = i
        canonical[frozenset()] = zero_wire
        
        # Work with mutable sets
        rows = [set(s) for s in row_sets]
        
        # Track usage: wire -> set of row indices containing it
        wire_to_rows = defaultdict(set)
        for i, rs in enumerate(rows):
            for w in rs:
                wire_to_rows[w].add(i)
        
        for _ in range(max_iter):
            if not elapsed_ok():
                break
            
            # Count pairs efficiently
            pair_count = defaultdict(int)
            pair_rows = defaultdict(set)
            
            for i, rs in enumerate(rows):
                if len(rs) < 2:
                    continue
                elems = list(rs)
                for x in range(len(elems)):
                    a = elems[x]
                    if a == zero_wire:
                        continue
                    for y in range(x + 1, len(elems)):
                        b = elems[y]
                        if b == zero_wire:
                            continue
                        key = (min(a, b), max(a, b))
                        if ws[key[0]] == ws[key[1]]:
                            continue
                        pair_count[key] += 1
                        pair_rows[key].add(i)
            
            if not pair_count:
                break
            
            # Find best pair: maximize benefit, tie-break by lower depth
            best_pair = None
            best_benefit = 0
            best_depth = 999
            for (a, b), cnt in pair_count.items():
                if cnt < 2:
                    continue
                benefit = cnt - 1
                new_depth = max(wd[a], wd[b]) + 1
                if new_depth > depth_limit:
                    continue
                if benefit > best_benefit or (benefit == best_benefit and new_depth < best_depth):
                    best_benefit = benefit
                    best_depth = new_depth
                    best_pair = (a, b)
            
            if best_pair is None:
                break
            
            a, b = best_pair
            new_set = ws[a] ^ ws[b]
            
            if len(new_set) == 0:
                # XOR produces zero; just remove both from rows
                for i in pair_rows[(a, b)]:
                    rs = rows[i]
                    if a in rs and b in rs:
                        rs.discard(a)
                        rs.discard(b)
                        wire_to_rows[a].discard(i)
                        wire_to_rows[b].discard(i)
                continue
            
            # Check if this wire already exists (canonicalization)
            if new_set in canonical:
                existing = canonical[new_set]
                # Replace a and b with existing wire in all rows
                for i in pair_rows[(a, b)]:
                    rs = rows[i]
                    if a in rs and b in rs:
                        rs.discard(a)
                        rs.discard(b)
                        if existing not in rs:
                            rs.add(existing)
                            wire_to_rows[existing].add(i)
                        wire_to_rows[a].discard(i)
                        wire_to_rows[b].discard(i)
                continue
            
            new_wire = next_wire
            next_wire += 1
            ws[new_wire] = new_set
            wd[new_wire] = max(wd[a], wd[b]) + 1
            canonical[new_set] = new_wire
            gates.append([a, b])
            
            for i in pair_rows[(a, b)]:
                rs = rows[i]
                if a in rs and b in rs:
                    rs.discard(a)
                    rs.discard(b)
                    rs.add(new_wire)
                    wire_to_rows[a].discard(i)
                    wire_to_rows[b].discard(i)
                    wire_to_rows[new_wire].add(i)
        
        # Finalize: for each row, if it has multiple wires, XOR them
        final_wires = []
        for i, rs in enumerate(rows):
            if len(rs) == 0:
                final_wires.append(zero_wire)
            elif len(rs) == 1:
                final_wires.append(list(rs)[0])
            else:
                elems = list(rs)
                elems.sort(key=lambda w: wd.get(w, 0))
                current = elems[0]
                for e in elems[1:]:
                    new_depth = max(wd.get(current, 0), wd.get(e, 0)) + 1
                    if new_depth > depth_limit:
                        break
                    gates.append([current, e])
                    current = next_wire
                    next_wire += 1
                    wd[current] = new_depth
                final_wires.append(current)
        
        return gates, final_wires, next_wire, ws, wd
    
    def eliminate_dead_gates(gates, outputs, zero_wire, n_inputs, depth_limit):
        """
        Eliminate gates whose outputs are never used.
        Returns a new list of gates with dead code removed.
        """
        if not gates:
            return gates
        
        # Determine which wires are needed
        needed = set(outputs)
        
        # Build gate index: wire -> (gate_idx, input1, input2)
        wire_to_gate = {}
        for idx, (a, b) in enumerate(gates):
            wire_to_gate[n_inputs + 1 + idx] = (idx, a, b)
        
        # Iteratively mark needed wires
        changed = True
        while changed:
            changed = False
            for wire in list(needed):
                if wire in wire_to_gate:
                    idx, a, b = wire_to_gate[wire]
                    if a not in needed:
                        needed.add(a)
                        changed = True
                    if b not in needed:
                        needed.add(b)
                        changed = True
        
        # Keep only gates whose output is needed
        new_gates = []
        wire_remap = {}
        for idx, (a, b) in enumerate(gates):
            out_wire = n_inputs + 1 + idx
            if out_wire in needed:
                new_gates.append([wire_remap.get(a, a), wire_remap.get(b, b)])
                wire_remap[out_wire] = out_wire
        
        # Remap outputs
        new_outputs = [wire_remap.get(w, w) for w in outputs]
        
        return new_gates, new_outputs
    
    def independent_baseline():
        """Baseline: each row independently synthesized."""
        gates = []
        next_wire = zero_wire + 1
        outputs = []
        for i in range(n_rows):
            if i in zero_rows:
                outputs.append(zero_wire)
                continue
            rs = set(np.where(A[i] == 1)[0].tolist())
            if len(rs) == 0:
                outputs.append(zero_wire)
            elif len(rs) == 1:
                outputs.append(list(rs)[0])
            else:
                elems = list(rs)
                current = elems[0]
                for e in elems[1:]:
                    gates.append([current, e])
                    current = next_wire
                    next_wire += 1
                outputs.append(current)
        return {'gates': gates, 'outputs': outputs}
    
    # Strategy 1: Pure greedy on all rows
    def try_pure_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        row_sets = []
        for i in range(n_rows):
            if i in zero_rows:
                row_sets.append(set())
            else:
                row_sets.append(set(np.where(A[i] == 1)[0].tolist()))
        
        gates, final_wires, _, _, _ = greedy_optimize(row_sets, depth_limit)
        return {'gates': gates, 'outputs': final_wires}
    
    # Strategy 2: Basis decomposition with greedy optimization
    def try_basis_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        
        # Gaussian elimination to find basis
        A_float = A.astype(np.float64)
        n = n_cols
        rank = 0
        pivot_cols = []
        basis_rows = []  # indices of independent rows
        row_reduced = [A_float[i].copy() for i in range(n_rows)]
        
        for i in range(n_rows):
            if i in zero_rows:
                continue
            v = row_reduced[i].copy()
            # Reduce by existing pivots
            for pc in pivot_cols:
                if v[pc] != 0:
                    v = (v + row_reduced[basis_rows[pivot_cols.index(pc)]]) % 2
            row_reduced[i] = v
            if v.sum() > 0:
                pc = int(np.argmax(v))
                pivot_cols.append(pc)
                basis_rows.append(i)
                rank += 1
        
        if rank == 0:
            return try_pure_greedy(seed)
        
        # Compute basis rows with greedy optimization
        basis_row_sets = []
        for i in basis_rows:
            basis_row_sets.append(set(np.where(A[i] == 1)[0].tolist()))
        
        gates, basis_wires, next_wire, ws, wd = greedy_optimize(basis_row_sets, depth_limit)
        
        # Express each output row
        outputs = []
        # Build mapping: for each row, find which basis rows it's a combination of
        # Re-do elimination to get coefficients
        pivot_to_basis_idx = {}
        for j, pc in enumerate(pivot_cols):
            pivot_to_basis_idx[pc] = j
        
        for i in range(n_rows):
            if i in zero_rows:
                outputs.append(zero_wire)
                continue
            
            v = A_float[i].copy()
            coeffs = []
            for j, pc in enumerate(pivot_cols):
                if v[pc] != 0:
                    coeffs.append(j)
                    v = (v + row_reduced[basis_rows[j]]) % 2
            
            if not coeffs:
                outputs.append(zero_wire)
            elif len(coeffs) == 1:
                outputs.append(basis_wires[coeffs[0]])
            else:
                # XOR all needed basis wires
                current = basis_wires[coeffs[0]]
                for ci in coeffs[1:]:
                    gates.append([current, basis_wires[ci]])
                    current = next_wire
                    next_wire += 1
                outputs.append(current)
        
        return {'gates': gates, 'outputs': outputs}
    
    # Strategy 3: Greedy with row ordering (process dense rows first for better sharing)
    def try_ordered_greedy(seed=None):
        if seed is not None:
            random.seed(seed)
        
        # Sort rows by density (descending) for better early sharing
        order = sorted(range(n_rows), key=lambda i: -row_popcounts[i])
        
        row_sets = []
        for i in range(n_rows):
            if i in zero_rows:
                row_sets.append(set())
            else:
                row_sets.append(set(np.where(A[i] == 1)[0].tolist()))
        
        # Reorder
        ordered_row_sets = [row_sets[i] for i in order]
        
        gates, final_wires, _, _, _ = greedy_optimize(ordered_row_sets, depth_limit)
        
        # Map back to original order
        outputs = [0] * n_rows
        for new_idx, orig_idx in enumerate(order):
            outputs[orig_idx] = final_wires[new_idx]
        
        return {'gates': gates, 'outputs': outputs}
    
    # Strategy 4: Two-phase: first greedy, then eliminate dead gates
    def try_two_phase(seed=None):
        if seed is not None:
            random.seed(seed)
        
        result = try_pure_greedy(seed)
        if result is None:
            return None
        
        # Eliminate dead gates
        gates = result['gates']
        outputs = result['outputs']
        if gates:
            new_gates, new_outputs = eliminate_dead_gates(gates, outputs, zero_wire, n_inputs, depth_limit)
            result['gates'] = new_gates
            result['outputs'] = new_outputs
        
        return result
    
    # Strategy 5: Greedy with improved benefit scoring (consider future sharing potential)
    def try_greedy_improved(row_sets, depth_limit, max_iter=40000):
        """
        Improved greedy: score pairs by (count-1) * (1 + 0.1*len(new_set)) to favor
        creating wires that are more useful for future sharing.
        """
        gates = []
        next_wire = zero_wire + 1
        
        ws = {}
        wd = {}
        for i in range(n_inputs):
            ws[i] = frozenset([i])
            wd[i] = 0
        ws[zero_wire] = frozenset()
        wd[zero_wire] = 0
        
        canonical = {}
        for i in range(n_inputs):
            canonical[frozenset([i])] = i
        canonical[frozenset()] = zero_wire
        
        rows = [set(s) for s in row_sets]
        
        wire_to_rows = defaultdict(set)
        for i, rs in enumerate(rows):
            for w in rs:
                wire_to_rows[w].add(i)
        
        for _ in range(max_iter):
            if not elapsed_ok():
                break
            
            pair_count = defaultdict(int)
            pair_rows = defaultdict(set)
            
            for i, rs in enumerate(rows):
                if len(rs) < 2:
                    continue
                elems = list(rs)
                for x in range(len(elems)):
                    a = elems[x]
                    if a == zero_wire:
                        continue
                    for y in range(x + 1, len(elems)):
                        b = elems[y]
                        if b == zero_wire:
                            continue
                        key = (min(a, b), max(a, b))
                        if ws[key[0]] == ws[key[1]]:
                            continue
                        pair_count[key] += 1
                        pair_rows[key].add(i)
            
            if not pair_count:
                break
            
            best_pair = None
            best_score = 0
            best_depth = 999
            for (a, b), cnt in pair_count.items():
                if cnt < 2:
                    continue
                new_set = ws[a] ^ ws[b]
                if len(new_set) == 0:
                    continue
                new_depth = max(wd[a], wd[b]) + 1
                if new_depth > depth_limit:
                    continue
                # Score: benefit * (1 + small bonus for larger new_set)
                base_benefit = cnt - 1
                size_bonus = 1 + 0.05 * len(new_set)
                score = base_benefit * size_bonus
                if score > best_score or (score == best_score and new_depth < best_depth):
                    best_score = score
                    best_depth = new_depth
                    best_pair = (a, b)
            
            if best_pair is None:
                break
            
            a, b = best_pair
            new_set = ws[a] ^ ws[b]
            
            if len(new_set) == 0:
                for i in pair_rows[(a, b)]:
                    rs = rows[i]
                    if a in rs and b in rs:
                        rs.discard(a)
                        rs.discard(b)
                        wire_to_rows[a].discard(i)
                        wire_to_rows[b].discard(i)
                continue
            
            if new_set in canonical:
                existing = canonical[new_set]
                for i in pair_rows[(a, b)]:
                    rs = rows[i]
                    if a in rs and b in rs:
                        rs.discard(a)
                        rs.discard(b)
                        if existing not in rs:
                            rs.add(existing)
                            wire_to_rows[existing].add(i)
                        wire_to_rows[a].discard(i)
                        wire_to_rows[b].discard(i)
                continue
            
            new_wire = next_wire
            next_wire += 1
            ws[new_wire] = new_set
            wd[new_wire] = max(wd[a], wd[b]) + 1
            canonical[new_set] = new_wire
            gates.append([a, b])
            
            for i in pair_rows[(a, b)]:
                rs = rows[i]
                if a in rs and b in rs:
                    rs.discard(a)
                    rs.discard(b)
                    rs.add(new_wire)
                    wire_to_rows[a].discard(i)
                    wire_to_rows[b].discard(i)
                    wire_to_rows[new_wire].add(i)
        
        final_wires = []
        for i, rs in enumerate(rows):
            if len(rs) == 0:
                final_wires.append(zero_wire)
            elif len(rs) == 1:
                final_wires.append(list(rs)[0])
            else:
                elems = list(rs)
                elems.sort(key=lambda w: wd.get(w, 0))
                current = elems[0]
                for e in elems[1:]:
                    new_depth = max(wd.get(current, 0), wd.get(e, 0)) + 1
                    if new_depth > depth_limit:
                        break
                    gates.append([current, e])
                    current = next_wire
                    next_wire += 1
                    wd[current] = new_depth
                final_wires.append(current)
        
        return gates, final_wires, next_wire, ws, wd
    
    # Run multiple strategies
    best_result = None
    best_gate_count = float('inf')
    
    # Try pure greedy with a few seeds
    for seed in [0, 1, 42, 7]:
        if not elapsed_ok():
            break
        result = try_pure_greedy(seed)
        if result is not None:
            gc = len(result['gates'])
            if gc < best_gate_count:
                best_gate_count = gc
                best_result = result
    
    # Try ordered greedy
    for seed in [0, 1]:
        if not elapsed_ok():
            break
        result = try_ordered_greedy(seed)
        if result is not None:
            gc = len(result['gates'])
            if gc < best_gate_count:
                best_gate_count = gc
                best_result = result
    
    # Try basis greedy
    for seed in [0, 1, 42]:
        if not elapsed_ok():
            break
        result = try_basis_greedy(seed)
        if result is not None:
            gc = len(result['gates'])
            if gc < best_gate_count:
                best_gate_count = gc
                best_result = result
    
    # Try improved greedy with dead code elimination
    for seed in [0, 1, 42]:
        if not elapsed_ok():
            break
        row_sets = []
        for i in range(n_rows):
            if i in zero_rows:
                row_sets.append(set())
            else:
                row_sets.append(set(np.where(A[i] == 1)[0].tolist()))
        
        gates, final_wires, _, _, _ = try_greedy_improved(row_sets, depth_limit)
        result = {'gates': gates, 'outputs': final_wires}
        
        # Eliminate dead gates
        if result['gates']:
            new_gates, new_outputs = eliminate_dead_gates(result['gates'], result['outputs'], zero_wire, n_inputs, depth_limit)
            result['gates'] = new_gates
            result['outputs'] = new_outputs
        
        gc = len(result['gates'])
        if gc < best_gate_count:
            best_gate_count = gc
            best_result = result
    
    # Apply dead code elimination to the best result
    if best_result is not None and best_result['gates']:
        new_gates, new_outputs = eliminate_dead_gates(best_result['gates'], best_result['outputs'], zero_wire, n_inputs, depth_limit)
        best_result['gates'] = new_gates
        best_result['outputs'] = new_outputs
    
    if best_result is None:
        best_result = independent_baseline()
    
    return best_result

# EVOLVE-BLOCK-END
