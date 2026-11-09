import time

def construct():
    # The smallest known tournament satisfying the condition is the Paley tournament
    # on p = 67 vertices.
    # p = 67 is prime and 67 % 4 == 3.
    # In a Paley tournament, i -> j iff (j - i) % p is a quadratic residue.
    # The orientation list requires one bit per pair (i, j) with i < j in lexicographic order.
    # 0 means i -> j, 1 means j -> i.
    
    p = 67
    
    # Precompute quadratic residues modulo p
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                # i -> j
                orientation.append(0)
            else:
                # j -> i
                orientation.append(1)
    
    return {'n': p, 'orientation': orientation}
