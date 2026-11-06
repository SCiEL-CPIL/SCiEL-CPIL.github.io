"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 85.0  # 5s safety margin
    
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
    gates = []  # list of [wire_a, wire_b]
    
    # Memo: frozenset -> wire_id
    memo = {}
    # Inputs are always available
    for j in range(n):
        memo[frozenset([j])] = j
    # Const-zero wire
    memo[frozenset()] = n
    
    def elapsed():
        return time.time() - start_time
    
    def add_gate(a, b):
        """Add a gate XORing wires a and b, return new wire id."""
        nonlocal next_wire
        gates.append([a, b])
        w = next_wire
        next_wire += 1
        return w
    
    def compute_set(s):
        """Compute wire for frozenset s, using memo and adding gates as needed."""
        if s in memo:
            return memo[s]
        
        # Try to find existing wire W such that s △ W is in memo
        # This gives a 1-gate solution
        for W, wire_w in memo.items():
            r = s ^ W  # symmetric difference
            if r in memo:
                wire_r = memo[r]
                new_w = add_gate(wire_w, wire_r)
                memo[s] = new_w
                return new_w
        
        # Try: find W in memo such that r = s ^ W has small popcount
        # and r can be computed cheaply
        best_cost = float('inf')
        best_W = None
        best_r = None
        for W, wire_w in memo.items():
            r = s ^ W
            cost = len(r) - 1  # gates to compute r from inputs (if r not in memo)
            if r in memo:
                cost = 0  # already have it, just need 1 gate for XOR with W
            if cost < best_cost:
                best_cost = cost
                best_W = W
                best_r = r
                if cost == 0:
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
        
        # Fallback: compute from scratch using balanced tree
        if len(s) == 0:
            memo[s] = n
            return n
        if len(s) == 1:
            j = next(iter(s))
            memo[s] = j
            return j
        
        # Balanced XOR tree
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
    
    outputs = []
    for i in range(m):
        if elapsed() > DEADLINE:
            # Return best result so far
            # Pad outputs with whatever we have
            while len(outputs) < m:
                outputs.append(n)  # const-zero as fallback
            break
        w = compute_set(rows[i])
        outputs.append(w)
    
    # Pad if we hit deadline early
    while len(outputs) < m:
        outputs.append(n)
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
