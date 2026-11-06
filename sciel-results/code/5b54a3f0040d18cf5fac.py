import time

def construct():
    # Use Paley graph of order 29
    # 29 is a prime, 29 % 4 == 1
    p = 29
    
    # Calculate quadratic residues mod p
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    # Generate colors in lexicographic order of pairs (i, j) with i < j
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': p, 'colors': colors}
