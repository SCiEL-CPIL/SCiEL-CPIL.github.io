import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 12
    m = 40
    N = 1 << n  # 4096
    
    # Helper to check if a is strict subset of b
    def is_subset(a, b):
        return (a & b) == a and a != b

    # Helper to check if a and b are comparable (one is subset of other)
    def is_comparable(a, b):
        if a == b:
            return False
        return (a & b) == a or (a & b) == b

    # Greedy Initialization
    F = []
    in_F = [False] * N
    
    for _ in range(m):
        best_gain = -1
        best_mask = -1
        for mask in range(N):
            if in_F[mask]:
                continue
            gain = 0
            for existing in F:
                if is_comparable(mask, existing):
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
        if best_mask == -1:
            # Should not happen if N > m
            break
        F.append(best_mask)
        in_F[best_mask] = True
        
    # Local Search (Hill Climbing)
    # We represent F as a list of masks.
    # in_F is a boolean array.
    
    def get_score():
        score = 0
        for i in range(len(F)):
            for j in range(i + 1, len(F)):
                if is_comparable(F[i], F[j]):
                    score += 1
        return score

    current_score = get_score()
    
    # Precompute comparability for all pairs of masks? 
    # N=4096, N^2 = 16M. A list of lists of booleans is too big/slow in Python.
    # We will compute on the fly but optimize the loops.
    
    # To speed up hill climbing:
    # 1. Precompute G(y) for all y not in F: number of elements in F comparable to y.
    # 2. Precompute loss(x) for all x in F: number of elements in F \ {x} comparable to x.
    
    while time.time() < deadline:
        # Precompute gain for all y not in F
        # G[y] = count of z in F such that is_comparable(y, z)
        G = [0] * N
        for y in range(N):
            if in_F[y]:
                continue
            count = 0
            for z in F:
                if is_comparable(y, z):
                    count += 1
            G[y] = count
            
        # Precompute loss for all x in F
        # loss[x] = count of z in F, z != x, such that is_comparable(x, z)
        loss = {}
        for x in F:
            count = 0
            for z in F:
                if x == z:
                    continue
                if is_comparable(x, z):
                    count += 1
            loss[x] = count
            
        best_delta = 0
        best_swap = None # (x, y)
        
        # Find best swap
        for x in F:
            lx = loss[x]
            for y in range(N):
                if in_F[y]:
                    continue
                # Delta = Gain(y, F \ {x}) - Loss(x, F \ {x})
                # Gain(y, F \ {x}) = G[y] - (1 if is_comparable(y, x) else 0)
                comp_yx = 1 if is_comparable(y, x) else 0
                delta = (G[y] - comp_yx) - lx
                
                if delta > best_delta:
                    best_delta = delta
                    best_swap = (x, y)
                    
        if best_swap is None:
            break
            
        x, y = best_swap
        # Perform swap
        F.remove(x)
        in_F[x] = False
        F.append(y)
        in_F[y] = True
        current_score += best_delta
        
    # Convert masks back to lists of integers
    result = []
    for mask in F:
        subset = [i for i in range(n) if (mask >> i) & 1]
        result.append(subset)
        
    return result
