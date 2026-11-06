import time

def construct():
    # Deadline guard: 150s limit, use 120s for safety margin
    start_time = time.time()
    deadline = start_time + 115.0
    
    # The Paley graph of order 29 is a strongly regular graph (29, 14, 6, 7).
    # Both it and its complement are K_{2,2,2}-free.
    # This provides a valid coloring for n=29, which is the current record.
    # The seed used n=17. This is an improvement.
    
    n = 29
    q = n
    
    # Check if q is a valid prime power for Paley graph (q % 4 == 1)
    # 29 is prime and 29 % 4 == 1.
    
    # Calculate quadratic residues modulo q
    # Residues are {x^2 mod q for x in 1..q-1}
    squares = set()
    for x in range(1, q):
        squares.add((x * x) % q)
    
    # Construct colors in lexicographic order (i, j) with i < j
    # Color 0 if edge exists in Paley graph (difference is a square)
    # Color 1 otherwise
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % q
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    # Verify time
    if time.time() > deadline:
        # Should not happen for n=29, but just in case
        return {'n': 0, 'colors': []}
        
    return {'n': n, 'colors': colors}
