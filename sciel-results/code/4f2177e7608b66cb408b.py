import time
import math

def compute_score(U):
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    if len(U) < 2:
        return 0.0
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
            
    ratio = len(diffset) / len(sumset)
    # If ratio is 1, score is 1.
    # If ratio < 1, score < 1.
    
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / denom

def generate_sidon(n):
    if n <= 0:
        return []
    U = [0]
    for i in range(1, n):
        candidate = 1
        while True:
            U.append(candidate)
            sums = set()
            is_sidon = True
            for j in range(len(U)):
                for k in range(j, len(U)):
                    s = U[j] + U[k]
                    if s in sums:
                        is_sidon = False
                        break
                    sums.add(s)
                if not is_sidon:
                    break
            if is_sidon:
                break
            else:
                U.pop()
                candidate += 1
    return U

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_score = 0.0
    best_U = [0, 1, 3]
    
    # Search n from 3 to 100
    for n in range(3, 100):
        if time.time() > deadline:
            break
        U = generate_sidon(n)
        score = compute_score(U)
        if score > best_score:
            best_score = score
            best_U = U
            
    return best_U
