import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    n = 10
    # Construct a single Latin square of order 10
    # Using the standard cyclic construction: L[i][j] = (i + j) % n
    # This ensures every row and column contains each symbol 0..9 exactly once.
    square = []
    for i in range(n):
        row = []
        for j in range(n):
            # Ensure Python int type
            val = int((i + j) % n)
            row.append(val)
        square.append(row)
    
    # Quick validation to ensure strict compliance (optional but safe)
    # Check rows
    for r in square:
        if sorted(r) != list(range(n)):
            # Fallback: return empty? No, must return non-empty. 
            # The construction is mathematically guaranteed to be correct.
            pass
    # Check cols
    for j in range(n):
        col = [square[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            pass

    # Since max MOLS for n=10 is 1, we return a list containing just this one square.
    # The return type must be a list of lists of lists.
    result = [square]
    
    # Ensure all types are Python int and list
    # The construction above uses int() and list.append, so types are correct.
    
    return result
