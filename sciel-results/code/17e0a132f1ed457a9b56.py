def construct():
    n = 29
    # Compute quadratic residues modulo 29
    # x^2 % 29 for x in 1..28
    residues = set()
    for x in range(1, n):
        residues.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            # If diff is a quadratic residue, color 0, else 1
            # Note: The choice of 0/1 mapping doesn't matter for validity, 
            # only that it's consistent. 
            # Paley graphs are self-complementary, so both colorings are valid.
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
