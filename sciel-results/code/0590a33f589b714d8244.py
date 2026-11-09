"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 85  # safety margin before 90s kill
    
    mat = np.array(matrix, dtype=np.int32)
    n_rows, n_cols = mat.shape
    n = n_cols
    zero_wire = n  # constant-zero wire
    
    # For each row, get the set of input indices where value is 1
    row_sets = []
    for i in range(n_rows):
        cols = np.nonzero(mat[i])[0]
        row_sets.append(frozenset(cols.tolist()))
    
    # Group rows by their input set
    set_to_rows = defaultdict(list)
    for i, s in enumerate(row_sets):
        set_to_rows[s].append(i)
    
    # We'll assign wires. Input wires: 0..n-1, zero_wire = n
    # Gate wires start at n+1
    # wire_cache: frozenset -> wire index (for XOR of all elements in the set)
    # We also cache intermediate results during tree building
    
    # Strategy: for each unique set, build a balanced XOR tree.
    # To maximize sharing, process sets in order of increasing popcount.
    # When building the tree for a set S, split S into two halves S1, S2.
    # If S1 or S2 (or any subset used in the tree) is already cached, reuse it.
    
    # Actually, the most effective sharing: 
    # 1. All singleton sets {i} map to wire i (input wires)
    # 2. For a set S with |S| >= 2, we need to XOR |S| elements.
    #    Balanced tree: split into two roughly equal parts, recurse.
    #    Cache every intermediate subset.
    
    # Let's use a recursive approach with memoization.
    # get_wire(subset) returns a wire that computes XOR of all elements in subset.
    # If subset is empty: return zero_wire
    # If subset has 1 element: return that element (input wire)
    # Otherwise: split into two parts, get wires for each, XOR them.
    
    # To minimize total gates, we want to maximize reuse of cached intermediate results.
    # Processing order: smaller sets first, so their results are available for larger sets.
    
    unique_sets = list(set_to_rows.keys())
    # Sort by popcount ascending
    unique_sets.sort(key=lambda s: len(s))
    
    wire_cache = {}  # frozenset -> wire index
    gates = []  # list of [wire1, wire2]
    next_wire = n + 1  # first gate wire
    
    def get_wire(subset):
        nonlocal next_wire
        if len(subset) == 0:
            return zero_wire
        if len(subset) == 1:
            return next(iter(subset))
        if subset in wire_cache:
            return wire_cache[subset]
        
        # Split into two roughly equal parts
        elems = sorted(subset)
        mid = len(elems) // 2
        s1 = frozenset(elems[:mid])
        s2 = frozenset(elems[mid:])
        
        w1 = get_wire(s1)
        w2 = get_wire(s2)
        
        if w1 == w2:
            # XOR of same wire is 0
            result = zero_wire
        else:
            gates.append([w1, w2])
            result = next_wire
            next_wire += 1
        
        wire_cache[subset] = result
        return result
    
    # Process all unique sets
    for s in unique_sets:
        get_wire(s)
    
    # Build outputs
    outputs = []
    for i in range(n_rows):
        s = row_sets[i]
        outputs.append(get_wire(s))
    
    # Check deadline
    if time.time() > DEADLINE:
        # Should not happen for small instances, but guard anyway
        pass
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
