def synthesize(matrix, depth_limit):
    import numpy as np
    import time
    
    start_time = time.time()
    DEADLINE = 85.0  # safety margin
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    # Convert to list of frozensets
    rows = []
    for i in range(m):
        s = frozenset(j for j in range(n) if matrix[i][j])
        rows.append(s)
    
    # Available wires: wire index -> frozenset
    # Wires 0..n-1 are inputs: {i}
    # Wire n is constant zero: empty set
    wire_to_set = {}
    for i in range(n):
        wire_to_set[i] = frozenset([i])
    wire_to_set[n] = frozenset()
    
    set_to_wire = {}
    for i in range(n):
        set_to_wire[frozenset([i])] = i
    set_to_wire[frozenset()] = n
    
    gates = []  # list of [wire_a, wire_b]
    next_wire = n + 1
    
    def get_wire_for_set(s):
        """Get wire for set s, creating gates if needed. Returns (wire, depth_used)."""
        nonlocal next_wire
        if s in set_to_wire:
            return set_to_wire[s], 0
        
        # Try to find best decomposition
        # Option 1: if |s| == 1, shouldn't happen (handled above)
        # Option 2: find two existing sets whose symmetric difference is s
        # This is expensive; instead, greedily split
        
        # Greedy: pick an element, split into {elem} and s\{elem}
        # But better: try to use existing wires to minimize new gates
        
        # For now, simple approach: split s into two parts, compute each, XOR them
        # Choose split to maximize reuse
        
        best_wire = None
        best_depth = float('inf')
        
        # Try all existing sets S such that S is a proper subset of s or overlaps
        # Actually, try: for each existing wire with set S, compute s Δ S, then XOR
        # But s Δ S might not be in cache either
        
        # Simpler: just split s into two roughly equal parts
        elems = list(s)
        if len(elems) == 0:
            return n, 0
        
        # Try to find a cached set that is a subset of s with size >= 2
        # to reduce the size of the remaining part
        best_split = None
        best_remaining_size = len(s)
        
        for existing_set in set_to_wire:
            if not existing_set:
                continue
            # Check if existing_set is subset of s
            if existing_set <= s:
                remaining = s - existing_set
                if len(remaining) < best_remaining_size and len(remaining) > 0:
                    best_remaining_size = len(remaining)
                    best_split = existing_set
        
        if best_split is not None:
            # s = best_split XOR (s - best_split)
            wire1 = set_to_wire[best_split]
            wire2, depth2 = get_wire_for_set(s - best_split)
            # Create new gate
            gates.append([wire1, wire2])
            new_wire = next_wire
            next_wire += 1
            wire_to_set[new_wire] = s
            set_to_wire[s] = new_wire
            return new_wire, depth2 + 1
        else:
            # No useful split found; build from scratch
            # Split into two halves
            half = len(elems) // 2
            part1 = frozenset(elems[:half])
            part2 = frozenset(elems[half:])
            
            if len(part1) == 0:
                wire1 = n
                depth1 = 0
            elif len(part1) == 1:
                wire1 = list(part1)[0]
                depth1 = 0
            else:
                wire1, depth1 = get_wire_for_set(part1)
            
            if len(part2) == 0:
                wire2 = n
                depth2 = 0
            elif len(part2) == 1:
                wire2 = list(part2)[0]
                depth2 = 0
            else:
                wire2, depth2 = get_wire_for_set(part2)
            
            gates.append([wire1, wire2])
            new_wire = next_wire
            next_wire += 1
            wire_to_set[new_wire] = s
            set_to_wire[s] = new_wire
            return new_wire, max(depth1, depth2) + 1
    
    outputs = []
    for row in rows:
        wire, depth = get_wire_for_set(row)
        outputs.append(wire)
    
    return {'gates': gates, 'outputs': outputs}
