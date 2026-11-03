import math
import time

N = 11

def construct():
    start_time = time.time()
    deadline = 115.0  # 5s safety margin
    
    # Lattice parameters
    gap_factor = 1.001
    d = math.sqrt(3) * gap_factor
    
    u = (d, 0.0)
    v = (d / 2.0, d * math.sqrt(3) / 2.0)
    
    # Generate candidates
    # For N=11, max_k=3 is sufficient.
    # We can dynamically increase max_k if needed, but 3 is safe for 11.
    max_k = 3
    candidates = []
    
    for i in range(-max_k, max_k + 1):
        for j in range(-max_k, max_k + 1):
            if max(abs(i), abs(j), abs(i+j)) <= max_k:
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                
                n0 = abs(x)
                n1 = abs(x * 0.5 + y * math.sqrt(3) / 2.0)
                n2 = abs(-x * 0.5 + y * math.sqrt(3) / 2.0)
                
                norm = max(n0, n1, n2)
                candidates.append((norm, x, y))
                
                if time.time() > deadline:
                    return [[0.0, 0.0, 0.0]], (0.0, 0.0), 100.0, 0.0
                    
    candidates.sort(key=lambda item: item[0])
    
    if len(candidates) < N:
        # Fallback: just place them in a line? Or increase max_k?
        # For N=11, max_k=3 gives 19 candidates, so this shouldn't happen.
        pass

    selected = candidates[:N]
    max_norm = max(c[0] for c in selected)
    
    S = 1.0 + (2.0 / math.sqrt(3.0)) * max_norm
    S += 1e-5  # Safety margin
    
    inner = [[x, y, 0.0] for _, x, y in selected]
    
    return inner, (0.0, 0.0), S, 0.0
