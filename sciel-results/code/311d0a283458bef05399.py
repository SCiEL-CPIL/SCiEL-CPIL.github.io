import time
import random
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    if m == 0:
        return {'gates': [], 'outputs': []}

    # Convert rows to bitsets for efficient GF(2) manipulation
    row_bits = []
    for i in range(m):
        bits = 0
        for j in range(n):
            if matrix[i][j]:
                bits |= (1 << j)
        row_bits.append(bits)

    # If all zeros
    if not any(row_bits):
        return {'gates': [], 'outputs': [n] * m}

    def _verify(circuit):
        gates = circuit['gates']
        outputs = circuit['outputs']
        num_wires = n + 1 + len(gates)
        wire_val = [0] * num_wires
        for i in range(n):
            wire_val[i] = 1 << i
        wire_val[n] = 0

        for k, (a, b) in enumerate(gates):
            if a >= num_wires or b >= num_wires:
                return False
            wire_val[n + 1 + k] = wire_val[a] ^ wire_val[b]

        for i in range(m):
            expected = row_bits[i]
            if outputs[i] >= num_wires:
                return False
            if wire_val[outputs[i]] != expected:
                return False
        return True

    def _compute_depth(gates, outputs):
        if not gates:
            return 0
        num_wires = n + 1 + len(gates)
        wire_depth = [0] * num_wires
        for k, (a, b) in enumerate(gates):
            wire_depth[n + 1 + k] = max(wire_depth[a], wire_depth[b]) + 1
        max_depth = 0
        for w in outputs:
            if w < num_wires:
                max_depth = max(max_depth, wire_depth[w])
        return max_depth

    def _baseline_circuit():
        """Independent XOR of inputs for each output."""
        gates = []
        outputs = []
        next_wire = n + 1
        for i in range(m):
            bits = row_bits[i]
            if bits == 0:
                outputs.append(n)
            else:
                indices = [j for j in range(n) if (bits >> j) & 1]
                if len(indices) == 1:
                    outputs.append(indices[0])
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        return {'gates': gates, 'outputs': outputs}

    def _steiner_mst(time_budget, seed):
        """
        Steiner tree / MST approach over the Boolean hypercube.
        Terminals: zero vector (wire n) and all row vectors.
        We build a minimum spanning tree over the set of reachable vectors
        using Prim's algorithm, where edge weight is the number of gates
        needed to compute one vector from another (i.e., popcount of XOR difference).
        
        Actually, in GF(2), any vector can be computed from any other by XOR-ing
        with the difference. The cost to compute vector v from vector u is
        the number of input wires that differ, which is popcount(v XOR u).
        But that's not quite right for gate count - we need to think about it differently.
        
        Better approach: Use the classic CSE greedy but with a global MST-inspired
        ordering. Start from the zero vector, and greedily add vectors that are
        "closest" (fewest gates to compute from already-computed vectors).
        """
        rng = random.Random(seed)
        
        # We'll maintain a set of computed vectors and their wire assignments
        # Start with all input wires and the zero wire
        # Wire mapping: wire_id -> vector (bitset)
        wire_to_vec = {}
        vec_to_wire = {}
        
        # Initialize with inputs and zero
        for i in range(n):
            vec = 1 << i
            wire_to_vec[i] = vec
            vec_to_wire[vec] = i
        wire_to_vec[n] = 0
        vec_to_wire[0] = n
        
        next_wire = n + 1
        gates = []
        
        # Target vectors that need to be reachable
        targets = list(set(row_bits))
        
        # Remove zero if present (already have it)
        targets = [t for t in targets if t != 0]
        
        # If no targets, done
        if not targets:
            return {'gates': [], 'outputs': [n] * m}
        
        # Greedy Prim-like: repeatedly pick the target vector that can be
        # computed with the fewest new gates from already-computed vectors.
        # A vector v can be computed from any pair (u, w) where u XOR w = v,
        # or from a single wire if v is already in vec_to_wire.
        
        # For efficiency, we'll track which targets are still pending
        pending = list(targets)
        pending_set = set(pending)
        
        max_iter = 500
        for _ in range(max_iter):
            if time.time() > time_budget:
                break
            if not pending:
                break
            
            best_target = None
            best_cost = float('inf')
            best_pair = None
            
            # For each pending target, find the cheapest way to compute it
            # from already-computed vectors
            for t in pending:
                # Check if t is already computed
                if t in vec_to_wire:
                    best_target = t
                    best_cost = 0
                    best_pair = None
                    break
                
                # Try to compute t from two existing vectors
                # t = u XOR w => u = t XOR w
                # Iterate over existing vectors w, check if t XOR w is also existing
                # This is O(|computed|) per target, which could be slow
                
                # Instead, use a heuristic: try XOR-ing t with each existing wire
                # and see if the result is also an existing wire
                # But that's still expensive. Let's use a different approach.
                
                # Alternative: compute t as XOR of a subset of input wires
                # The cost is popcount(t) - 1 gates (XOR all 1-bits together)
                # But we might do better by using intermediate wires.
                
                # For now, use the simple cost: popcount(t) - 1
                # and see if we can do better with existing intermediates.
                
                pc = bin(t).count('1')
                cost_direct = max(0, pc - 1)
                
                if cost_direct < best_cost:
                    best_cost = cost_direct
                    best_target = t
                    best_pair = 'direct'
            
            if best_target is None:
                break
            
            t = best_target
            
            if t in vec_to_wire:
                # Already computed
                pending.remove(t)
                pending_set.discard(t)
                continue
            
            # Compute t
            if best_pair == 'direct':
                # XOR all input wires that are 1 in t
                indices = [j for j in range(n) if (t >> j) & 1]
                if len(indices) == 0:
                    wire_id = n
                elif len(indices) == 1:
                    wire_id = indices[0]
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    wire_id = prev
            else:
                # Should not reach here with current logic
                continue
            
            wire_to_vec[wire_id] = t
            vec_to_wire[t] = wire_id
            pending.remove(t)
            pending_set.discard(t)
        
        # Build outputs
        outputs = []
        for i in range(m):
            bits = row_bits[i]
            if bits == 0:
                outputs.append(n)
            elif bits in vec_to_wire:
                outputs.append(vec_to_wire[bits])
            else:
                # Fallback: compute directly
                indices = [j for j in range(n) if (bits >> j) & 1]
                if len(indices) == 1:
                    outputs.append(indices[0])
                else:
                    prev = indices[0]
                    for idx in indices[1:]:
                        gates.append([prev, idx])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        
        return {'gates': gates, 'outputs': outputs}

    def _greedy_cse_v3(time_budget, seed):
        """
        Improved greedy CSE with better pair selection.
        Uses frequency-weighted co-occurrence and considers the impact
        on reducing total popcount across all expressions.
        """
        rng = random.Random(seed)
        exprs = list(row_bits)
        gates = []
        next_wire = n + 1
        
        max_iter = 400
        for _ in range(max_iter):
            if time.time() > time_budget:
                break
            
            # Count total popcount to measure improvement
            total_pc = sum(bin(e).count('1') for e in exprs)
            
            # Count frequency of each wire
            freq = {}
            for e in exprs:
                temp = e
                w = 0
                while temp:
                    if temp & 1:
                        freq[w] = freq.get(w, 0) + 1
                    temp >>= 1
                    w += 1
            
            # Only consider wires with freq >= 2
            candidates = [w for w, c in freq.items() if c >= 2]
            if len(candidates) < 2:
                break
            
            # Sort by frequency descending
            candidates.sort(key=lambda w: freq[w], reverse=True)
            top_k = min(len(candidates), 40)
            top_candidates = candidates[:top_k]
            
            best_pair = None
            best_benefit = 0
            
            for i in range(len(top_candidates)):
                a = top_candidates[i]
                a_bit = 1 << a
                for j in range(i + 1, len(top_candidates)):
                    b = top_candidates[j]
                    b_bit = 1 << b
                    
                    # Count how many exprs contain both a and b
                    benefit = 0
                    for e in exprs:
                        if (e & a_bit) and (e & b_bit):
                            benefit += 1
                    
                    if benefit > best_benefit:
                        best_benefit = benefit
                        best_pair = (a, b)
            
            if best_pair is None or best_benefit <= 1:
                break
            
            a, b = best_pair
            g = next_wire
            next_wire += 1
            gates.append([a, b])
            
            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g
            
            for i in range(m):
                e = exprs[i]
                if (e & a_bit) and (e & b_bit):
                    e = (e & ~a_bit & ~b_bit) | g_bit
                    exprs[i] = e
        
        # Build outputs
        outputs = []
        for i in range(m):
            e = exprs[i]
            if e == 0:
                outputs.append(n)
            else:
                wires = []
                temp = e
                w = 0
                while temp:
                    if temp & 1:
                        wires.append(w)
                    temp >>= 1
                    w += 1
                
                if len(wires) == 1:
                    outputs.append(wires[0])
                elif len(wires) == 0:
                    outputs.append(n)
                else:
                    prev = wires[0]
                    for w in wires[1:]:
                        gates.append([prev, w])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        
        return {'gates': gates, 'outputs': outputs}

    def _pairwise_greedy(time_budget, seed):
        """
        Alternative: start from each pair of rows, find common subexpressions,
        and build up a shared structure.
        """
        rng = random.Random(seed)
        exprs = list(row_bits)
        gates = []
        next_wire = n + 1
        
        max_iter = 300
        for _ in range(max_iter):
            if time.time() > time_budget:
                break
            
            # For each pair of expressions, find common XOR subterms
            # and extract them as shared gates.
            
            best_extraction = None
            best_saving = 0
            
            # Try extracting XOR of pairs of inputs that appear together
            # in multiple expressions
            freq = {}
            for e in exprs:
                temp = e
                w = 0
                while temp:
                    if temp & 1:
                        freq[w] = freq.get(w, 0) + 1
                    temp >>= 1
                    w += 1
            
            candidates = [w for w, c in freq.items() if c >= 2]
            if len(candidates) < 2:
                break
            
            candidates.sort(key=lambda w: freq[w], reverse=True)
            top_k = min(len(candidates), 30)
            top_candidates = candidates[:top_k]
            
            for i in range(len(top_candidates)):
                a = top_candidates[i]
                a_bit = 1 << a
                for j in range(i + 1, len(top_candidates)):
                    b = top_candidates[j]
                    b_bit = 1 << b
                    
                    benefit = 0
                    for e in exprs:
                        if (e & a_bit) and (e & b_bit):
                            benefit += 1
                    
                    if benefit > best_saving:
                        best_saving = benefit
                        best_extraction = (a, b)
            
            if best_extraction is None or best_saving <= 1:
                break
            
            a, b = best_extraction
            g = next_wire
            next_wire += 1
            gates.append([a, b])
            
            a_bit = 1 << a
            b_bit = 1 << b
            g_bit = 1 << g
            
            for i in range(m):
                e = exprs[i]
                if (e & a_bit) and (e & b_bit):
                    e = (e & ~a_bit & ~b_bit) | g_bit
                    exprs[i] = e
        
        # Build outputs
        outputs = []
        for i in range(m):
            e = exprs[i]
            if e == 0:
                outputs.append(n)
            else:
                wires = []
                temp = e
                w = 0
                while temp:
                    if temp & 1:
                        wires.append(w)
                    temp >>= 1
                    w += 1
                
                if len(wires) == 1:
                    outputs.append(wires[0])
                elif len(wires) == 0:
                    outputs.append(n)
                else:
                    prev = wires[0]
                    for w in wires[1:]:
                        gates.append([prev, w])
                        prev = next_wire
                        next_wire += 1
                    outputs.append(prev)
        
        return {'gates': gates, 'outputs': outputs}

    # First, get a baseline
    baseline = _baseline_circuit()
    best_result = baseline
    best_gate_count = len(baseline['gates'])

    # Try multiple strategies with different seeds
    strategies = [
        ('greedy_cse_v3', _greedy_cse_v3),
        ('pairwise_greedy', _pairwise_greedy),
        ('steiner_mst', _steiner_mst),
    ]

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    for strategy_name, strategy_func in strategies:
        for seed in seeds:
            if time.time() > deadline - 3.0:
                break

            try:
                result = strategy_func(deadline - 3.0, seed)
                if _verify(result):
                    d = _compute_depth(result['gates'], result['outputs'])
                    if d <= depth_limit:
                        gc = len(result['gates'])
                        if gc < best_gate_count:
                            best_gate_count = gc
                            best_result = result
            except Exception:
                pass

    return best_result
