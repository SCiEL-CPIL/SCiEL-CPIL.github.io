def synthesize(matrix, depth_limit):
    import numpy as np
    from collections import defaultdict
    
    m, n = matrix.shape
    
    # Convert matrix to list of row bitmasks
    row_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i, j]:
                mask |= (1 << j)
        row_masks.append(mask)
    
    # Expressions: list of sets of wire IDs
    expressions = [set() for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if row_masks[i] & (1 << j):
                expressions[i].add(j)
    
    # Wire management
    wire_masks = [1 << i for i in range(n)]
    wire_masks.append(0)  # wire n is zero
    mask_to_wire = {0: n}
    for i in range(n):
        mask_to_wire[1 << i] = i
    
    gates = []
    next_wire = n + 1
    
    def extract_pair(u, v):
        """Extract u^v as a new wire. Returns new wire ID."""
        nonlocal next_wire, wire_masks, mask_to_wire, gates
        
        new_mask = wire_masks[u] ^ wire_masks[v]
        
        if new_mask in mask_to_wire:
            existing = mask_to_wire[new_mask]
            for i in range(m):
                expr = expressions[i]
                if u in expr and v in expr:
                    expr.remove(u)
                    expr.remove(v)
                    expr.add(existing)
            return existing
        
        wire_masks.append(new_mask)
        mask_to_wire[new_mask] = next_wire
        new_id = next_wire
        next_wire += 1
        gates.append([u, v])
        
        for i in range(m):
            expr = expressions[i]
            if u in expr and v in expr:
                expr.remove(u)
                expr.remove(v)
                expr.add(new_id)
        
        return new_id
    
    # Iteratively extract the most beneficial pairs
    while True:
        pair_count = defaultdict(int)
        for expr in expressions:
            if len(expr) < 2:
                continue
            lst = list(expr)
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    u, v = lst[i], lst[j]
                    if u > v:
                        u, v = v, u
                    pair_count[(u, v)] += 1
        
        if not pair_count:
            break
        
        max_count = max(pair_count.values())
        if max_count < 2:
            break
        
        # Among max-count pairs, prefer those with simpler resulting masks
        best_pair = None
        best_metric = None
        for (u, v), count in pair_count.items():
            if count == max_count:
                new_mask = wire_masks[u] ^ wire_masks[v]
                bc = bin(new_mask).count('1')
                metric = (bc, u, v)
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    best_pair = (u, v)
        
        u, v = best_pair
        extract_pair(u, v)
    
    # Finalize outputs
    outputs = []
    for expr in expressions:
        if not expr:
            outputs.append(n)
        elif len(expr) == 1:
            outputs.append(list(expr)[0])
        else:
            lst = list(expr)
            current = lst[0]
            for w in lst[1:]:
                nm = wire_masks[current] ^ wire_masks[w]
                if nm in mask_to_wire:
                    current = mask_to_wire[nm]
                else:
                    gates.append([current, w])
                    wire_masks.append(nm)
                    mask_to_wire[nm] = next_wire
                    current = next_wire
                    next_wire += 1
            outputs.append(current)
    
    return {'gates': gates, 'outputs': outputs}
