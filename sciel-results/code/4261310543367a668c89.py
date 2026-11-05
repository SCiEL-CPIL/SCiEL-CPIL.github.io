def is_qr(x, p):
    """Check if x is a quadratic residue mod p (x != 0)."""
    return pow(x, (p - 1) // 2, p) == 1

def construct():
    # Use Paley graph P(37) and K3
    p = 37
    n_p = p
    n_total = n_p + 3
    
    edges = []
    
    # Paley graph P(37)
    # Vertices 0..36
    # Edge (i, j) if (j - i) % p is a quadratic residue
    for i in range(n_p):
        for j in range(i + 1, n_p):
            diff = (j - i) % p
            if is_qr(diff, p):
                edges.append([i, j])
                
    # K3 on vertices 37, 38, 39
    edges.append([37, 38])
    edges.append([38, 39])
    edges.append([37, 39])
    
    return {'n': n_total, 'edges': edges}
