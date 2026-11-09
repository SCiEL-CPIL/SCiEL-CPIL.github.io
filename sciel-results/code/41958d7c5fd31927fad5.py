def construct():
    """
    Constructs a tournament with the minimum known number of vertices 
    such that every set of 4 vertices has a common dominator.
    
    This uses the Paley tournament construction over the prime field F_67.
    p = 67 is a prime congruent to 3 mod 4.
    The orientation is defined such that i -> j if (j - i) mod p is a quadratic residue.
    """
    p = 67
    
    # Precompute quadratic residues modulo p
    # Squares of 1..p-1
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    # Build the orientation list
    # One bit per pair (i, j) with i < j
    # 0 means i -> j, 1 means j -> i
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': p, 'orientation': orientation}
