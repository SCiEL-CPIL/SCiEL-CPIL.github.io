import numpy as np

def construct():
    p = 67
    # Precompute quadratic residues
    squares = np.zeros(p, dtype=bool)
    for x in range(1, p):
        squares[(x * x) % p] = True
        
    # Generate pairs (i, j) with i < j
    i, j = np.triu_indices(p, k=1)
    diffs = (j - i) % p
    
    # 0 if diff is a square (i -> j), 1 otherwise (j -> i)
    orientation = np.where(squares[diffs], 0, 1).tolist()
    
    return {'n': p, 'orientation': orientation}
