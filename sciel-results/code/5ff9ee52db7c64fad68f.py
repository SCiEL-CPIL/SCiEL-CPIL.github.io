def synthesize(matrix, depth_limit):
    import numpy as np
    from collections import defaultdict

    m, n = matrix.shape
    
    # Initial wires: 0..n-1 are inputs, n is zero
    wire_supports = [frozenset([i]) for i in range(n)]
    wire_supports.append(frozenset())
    
    # Expressions: list of sets of wire IDs
    expressions = []
    for i in range(m):
        expr = set()
        for j in range(n):
            if matrix[i, j]:
                expr.add(j)
        expressions.append(expr)
        
    gates = []
    next_wire = n + 1
    
    # Iterative CSE
    while True:
        pair_count = defaultdict(int)
        
        # Count pairs in current expressions
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
            
        best_pair = None
        for (u, v), c in pair_count.items():
            if c == max_count:
                best_pair = (u, v)
                break
                
        if best_pair is None:
            break
            
        u, v = best_pair
        
        # Create new wire
        new_support = wire_supports[u] ^ wire_supports[v]
        wire_supports.append(new_support)
        new_id = next_wire
        next_wire += 1
        
        # Add gate
        gates.append([u, v])
        
        # Update expressions: replace u and v with new_id
        for i in range(m):
            expr = expressions[i]
            if u in expr and v in expr:
                expr.remove(u)
                expr.remove(v)
                expr.add(new_id)
                
    # Final stage: combine remaining wires in each expression
    outputs = []
    for expr in expressions:
        if not expr:
            outputs.append(n)
        elif len(expr) == 1:
            outputs.append(list(expr)[0])
        else:
            # Calculate target support
            target_support = frozenset()
            for w in expr:
                target_support = target_support ^ wire_supports[w]
            
            # Check if already exists
            found_wire = None
            for k, s in enumerate(wire_supports):
                if s == target_support:
                    found_wire = k
                    break
            
            if found_wire is not None:
                outputs.append(found_wire)
            else:
                # Build it
                lst = list(expr)
                current = lst[0]
                for w in lst[1:]:
                    gates.append([current, w])
                    new_s = wire_supports[current] ^ wire_supports[w]
                    wire_supports.append(new_s)
                    current = len(wire_supports) - 1
                outputs.append(current)
                
    return {'gates': gates, 'outputs': outputs}
