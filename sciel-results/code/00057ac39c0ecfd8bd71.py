import itertools
import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    M = 3**n
    # Build the constraint matrix A (sparse)
    # A[j, i] = 1 if i is within distance 1 of j
    # This is the same as "i covers j"
    # We can build this by iterating over i (centers) and adding to its ball
    # Or iterating over j (points) and finding neighbors.
    
    # Let's build A as CSR for efficiency
    rows = []
    cols = []
    data = []
    
    # Precompute powers of 3
    p3 = [1, 3, 9, 27, 81, 243, 729, 2187]
    
    for j in range(M):
        # Find neighbors of j (including j)
        neighbors = []
        temp = j
        digits = [0]*n
        for k in range(n):
            digits[k] = temp % 3
            temp //= 3
            
        neighbors.append(j)
        for pos in range(n):
            # +1
            new_d = (digits[pos] + 1) % 3
            diff = new_d - digits[pos]
            neighbors.append(j + diff * p3[pos])
            # -1
            new_d = (digits[pos] - 1) % 3
            diff = new_d - digits[pos]
            neighbors.append(j + diff * p3[pos])
            
        for i in neighbors:
            rows.append(j)
            cols.append(i)
            data.append(1)
            
    A = sp.csr_matrix((data, (rows, cols)), shape=(M, M))
    
    c = np.ones(M)
    bounds = [(0, 1)] * M
    integrality = np.ones(M) # All integer
    
    try:
        start_time = time.time()
        # Use HiGHS if available, which is default for milp in recent scipy
        res = opt.milp(c, constraints=[(A, '>=', np.ones(M))], integrality=integrality, bounds=bounds, options={'time_limit': 110})
        
        if res.status == 0: # Optimal
            # Extract solution
            x = res.x
            # x is float, round to int
            code_indices = [i for i in range(M) if x[i] > 0.5]
        elif res.status == 1: # Iteration limit
            # Use the incumbent if available
            if hasattr(res, 'x') and res.x is not None:
                 x = res.x
                 code_indices = [i for i in range(M) if x[i] > 0.5]
            else:
                 # Fallback to greedy
                 code_indices = greedy_fallback(q, n, radius)
        else:
            code_indices = greedy_fallback(q, n, radius)
            
    except Exception as e:
        code_indices = greedy_fallback(q, n, radius)
        
    # Convert indices to codewords
    return [list(map(int, [ (c // (3**i)) % 3 for i in range(n) ])) for c in code_indices]

def greedy_fallback(q, n, radius):
    # Simple greedy to ensure valid output
    M = 3**n
    p3 = [1, 3, 9, 27, 81, 243, 729, 2187]
    balls = []
    for i in range(M):
        ball = set()
        temp = i
        digits = [0]*n
        for k in range(n):
            digits[k] = temp % 3
            temp //= 3
        ball.add(i)
        for pos in range(n):
            ball.add(i + ((digits[pos]+1)%3 - digits[pos]) * p3[pos])
            ball.add(i + ((digits[pos]-1)%3 - digits[pos]) * p3[pos])
        balls.append(ball)
        
    uncovered = set(range(M))
    code = []
    while uncovered:
        best_c = -1
        best_gain = 0
        # Randomize start to break symmetry?
        for c in range(M):
            if c in code: continue
            gain = len(balls[c].intersection(uncovered))
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c == -1: break
        code.append(best_c)
        uncovered -= balls[best_c]
        
    return code
