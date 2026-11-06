"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 55.0  # safety margin for 60s limit
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0:
        return {'gates': [], 'outputs': []}
    
    # Parse rows as frozensets of column indices
    rows = []
    for i in range(m):
        row_set = frozenset(j for j in range(n) if matrix[i][j] % 2 == 1)
        rows.append(row_set)
    
    # Wire assignment: inputs are 0..n-1, const-zero is n
    # New gates get wires n+1, n+2, ...
    next_wire = n + 1
    gates = []
    
    # Memo: frozenset -> wire_id
    memo = {}
    for j in range(n):
        memo[frozenset([j])] = j
    memo[frozenset()] = n
    
    def elapsed():
        return time.time() - start_time
    
    def add_gate(a, b):
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w
    
    def compute_set(s):
        """Compute wire for frozenset s, using memo and adding gates as needed."""
        if s in memo:
            return memo[s]
        
        if len(s) == 0:
            memo[s] = n
            return n
        if len(s) == 1:
            j = next(iter(s))
            memo[s] = j
            return j
        
        # Strategy 1: Find W in memo such that s ^ W is also in memo
        for W, wire_w in memo.items():
            r = s ^ W
            if r in memo:
                wire_r = memo[r]
                new_w = add_gate(wire_w, wire_r)
                memo[s] = new_w
                return new_w
        
        # Strategy 2: Find W in memo that maximizes |W ∩ s| (to minimize remaining work)
        # The remaining set is s \ W (as a set difference, but we need XOR)
        # Actually s ^ W = (s \ W) ∪ (W \ s), so we want W such that s ^ W has small popcount
        best_cost = float('inf')
        best_W = None
        best_r = None
        for W, wire_w in memo.items():
            r = s ^ W
            cost = len(r)
            if cost < best_cost:
                best_cost = cost
                best_W = W
                best_r = r
                if cost <= 2:
                    break
        
        if best_W is not None:
            wire_w = memo[best_W]
            if best_r in memo:
                wire_r = memo[best_r]
            else:
                wire_r = compute_set(best_r)
            new_w = add_gate(wire_w, wire_r)
            memo[s] = new_w
            return new_w
        
        # Fallback: balanced XOR tree
        elems = list(s)
        wires = [j for j in elems]
        while len(wires) > 1:
            new_wires = []
            for i in range(0, len(wires), 2):
                if i + 1 < len(wires):
                    new_w = add_gate(wires[i], wires[i+1])
                    new_wires.append(new_w)
                else:
                    new_wires.append(wires[i])
            wires = new_wires
        result_wire = wires[0]
        memo[s] = result_wire
        return result_wire
    
    # Pre-compute useful intermediate sets to maximize sharing
    # Collect all pairwise XORs of rows that might be useful
    # First, compute all outputs with the greedy approach
    outputs = []
    for i in range(m):
        if elapsed() > DEADLINE:
            while len(outputs) < m:
                outputs.append(n)
            break
        w = compute_set(rows[i])
        outputs.append(w)
    
    while len(outputs) < m:
        outputs.append(n)
    
    # Optimization pass: try alternative orderings and pre-computed intermediates
    if elapsed() < DEADLINE - 5.0:
        # Reset for optimization attempt
        next_wire = n + 1
        gates = []
        memo = {}
        for j in range(n):
            memo[frozenset([j])] = j
        memo[frozenset()] = n
        
        # Build a list of candidate intermediate sets:
        # 1. All rows
        # 2. XOR of pairs of rows (if popcount is reasonable)
        candidate_sets = list(rows)
        
        # Add pairwise XORs of rows with moderate popcount
        if m <= 32:
            for i in range(m):
                for j in range(i+1, m):
                    xor_set = rows[i] ^ rows[j]
                    if 0 < len(xor_set) <= max(4, len(rows[i]) // 2):
                        candidate_sets.append(xor_set)
        
        # Sort candidates by popcount ascending to build up shared sub-expressions
        candidate_sets = sorted(set(candidate_sets), key=lambda s: (len(s), hash(s)))
        
        # Pre-compute candidate sets that are cheap
        for cs in candidate_sets:
            if elapsed() > DEADLINE - 3.0:
                break
            if cs in memo:
                continue
            if len(cs) == 0:
                memo[cs] = n
                continue
            if len(cs) == 1:
                j = next(iter(cs))
                memo[cs] = j
                continue
            # Try to compute using existing memo
            found = False
            for W, wire_w in memo.items():
                r = cs ^ W
                if r in memo:
                    wire_r = memo[r]
                    new_w = add_gate(wire_w, wire_r)
                    memo[cs] = new_w
                    found = True
                    break
            if not found:
                # Only pre-compute if it's small enough
                if len(cs) <= 6:
                    compute_set(cs)
        
        # Now compute all rows
        opt_outputs = [None] * m
        for i in range(m):
            if elapsed() > DEADLINE - 2.0:
                break
            s = rows[i]
            opt_outputs[i] = compute_set(s)
        
        # Check if optimization helped
        if all(opt_outputs[i] is not None for i in range(m)) and len(gates) < len(outputs) + 50:
            outputs = opt_outputs
    
    while len(outputs) < m:
        outputs.append(n)
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
