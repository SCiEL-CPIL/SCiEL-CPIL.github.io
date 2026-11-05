import time
import random
from itertools import combinations

def _is_sidon_fast(arr):
    n = len(arr)
    seen = set()
    for i in range(n):
        ai = arr[i]
        for j in range(i, n):
            s = ai + arr[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(arr):
    n = len(arr)
    # Check from largest k down
    for k in range(n, 0, -1):
        for subset in combinations(arr, k):
            if _is_sidon_fast(subset):
                return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = None
    best_size = 12
    
    random.seed(42)
    
    candidates = []
    
    # 1. APs
    for step in range(1, 15):
        for start in range(0, 128 - 11*step + 1):
            arr = [start + i*step for i in range(12)]
            candidates.append(arr)
            
    # 2. Quadratics
    for a in range(1, 10):
        for b in range(0, 20):
            arr = [a*i*i + b*i for i in range(12)]
            if max(arr) <= 128:
                candidates.append(sorted(arr))
                
    # 3. Random perturbations of APs
    base_ap = list(range(12))
    for _ in range(200):
        arr = base_ap[:]
        # Swap a few elements
        for _ in range(3):
            i, j = random.sample(range(12), 2)
            arr[i], arr[j] = arr[j], arr[i]
        # Shift
        shift = random.randint(0, 100)
        arr = sorted([x + shift for x in arr])
        if max(arr) <= 128:
             candidates.append(arr)
             
    # 4. Dense clusters
    for _ in range(200):
        lo = random.randint(0, 100)
        hi = min(lo + random.randint(12, 20), 128)
        if hi - lo + 1 >= 12:
            candidates.append(sorted(random.sample(range(lo, hi+1), 12)))
            
    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            unique.append(c)
            
    # Evaluate
    for arr in unique:
        if time.time() > deadline - 5.0:
            break
        size = _max_sidon_size(arr)
        if size < best_size:
            best_size = size
            best_set = arr[:]
            if best_size <= 3:
                break
                
    return best_set
