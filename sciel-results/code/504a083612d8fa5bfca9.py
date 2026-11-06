# EVOLVE-BLOCK-START
def construct():
    MOD = 1009
    MAX_VAL = 1008
    
    # Compute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Identify self-inverse elements (forbidden)
    forbidden = set()
    for x in range(1, MOD):
        if (x * x) % MOD == 1:
            forbidden.add(x)
    
    # Build conflict edges
    edges = []
    seen = set()
    for x in range(1, MOD):
        y = inv[x]
        if y != x:
            a, b = min(x, y), max(x, y)
            edge = (a, b)
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)
    
    # Sort edges by b
    edges.sort(key=lambda e: e[1])
    
    best_start = 1
    best_length = 0
    current_min_start = 1
    edge_idx = 0
    n_edges = len(edges)
    last_forbidden = 0
    
    for e in range(1, MOD):
        # Update based on edges ending at e
        while edge_idx < n_edges and edges[edge_idx][1] == e:
            a, b = edges[edge_idx]
            current_min_start = max(current_min_start, a + 1)
            edge_idx += 1
        
        # Update based on forbidden elements
        if e in forbidden:
            last_forbidden = e
        
        if last_forbidden >= current_min_start:
            current_min_start = last_forbidden + 1
            
        if current_min_start <= e:
            length = e - current_min_start + 1
            if length > best_length:
                best_length = length
                best_start = current_min_start
    
    return [best_start, best_length]
# EVOLVE-BLOCK-END
