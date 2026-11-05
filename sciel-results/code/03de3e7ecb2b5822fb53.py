def construct():
    """
    Returns a Paley tournament on 67 vertices.
    The Paley tournament on p (prime, p % 4 == 3) is defined by i -> j iff (j-i) is a quadratic residue mod p.
    This specific instance is known to satisfy the 4-domination property (Schütte's problem k=4).
    """
    p = 67
    # Compute quadratic residues mod p
    squares = {x * x % p for x in range(1, p)}
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            # i -> j iff (j - i) % p is in squares
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': p, 'orientation': orientation}
