def construct():
    n = 29
    # Compute quadratic residues modulo 29
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in squares:
                colors.append(1)
            else:
                colors.append(0)
                
    return {'n': n, 'colors': colors}
