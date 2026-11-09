def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    # For the specific case of k=3, g=16, we use an optimized LUW construction.
    # The standard LUW construction with parameter d has girth >= 2d-1.
    # To ensure girth >= g, we need 2d - 1 >= g => d >= (g+1)/2.
    # For k=3, q=3.
    # N = q * (q^d - 1).
    
    if k != 3:
        # Fallback to original logic for other k if needed, 
        # but the task is specifically for k=3, g=16.
        # We'll just use the generic LUW with a safe d.
        d = max(3, (g + 1) // 2)
        return _luw_construct(k, d)
        
    # Optimized for k=3
    d = (g + 1) // 2
    if 2 * d - 1 < g:
        d += 1
        
    return _luw_construct(3, d)

def _luw_construct(q, d):
    """
    Lazebnik-Ustimenko-Woldar construction.
    Returns (n, edges).
    """
    # The LUW construction is defined for prime powers q.
    # If q is not a prime power, we might need to adjust or use a different method.
    # For k=3, q=3 is prime.
    
    # Check if q is a prime power
    # For simplicity, we assume q is a prime power or use the generic modular arithmetic
    # which works for any q >= 2 if we interpret the field operations correctly.
    # The original code used modulo q arithmetic, which effectively creates a cyclic group structure,
    # which is valid for the "generalized" LUW or similar Cayley graph constructions.
    
    # The specific LUW paper uses finite fields.
    # For q=3, it's GF(3).
    
    # Implementation details from the original _ludw_fallback:
    # It used `add, sub, mul` based on modulo q.
    # This constructs a graph on the group (Z/qZ)^d.
    # This is a valid construction for regular graphs of large girth.
    
    # Let's reuse the logic from _ludw_fallback but with the correct d.
    
    # The original code had a specific `rule` generation.
    # Let's ensure that logic is preserved.
    
    # Original rule logic:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in range(5, d + 1):
    #     r, c0 = (c - 5) % 4, c - (c - 5) % 4
    #     rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # This logic seems to generate a specific set of linear forms.
    # We will copy this logic into _luw_construct.
    
    # Note: The original _ludw_fallback handled q=4 specially with XOR.
    # For q=3, it used modulo 3.
    
    add = (lambda x, y: (x + y) % q)
    sub = (lambda x, y: (x - y) % q)
    mul = (lambda x, y: (x * y) % q)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
