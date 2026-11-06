import numpy as np
import time
import random

def _evaluate(a):
    """Compute max(|A+A|, |A*A|) for set a (array of 20 distinct ints)."""
    n = len(a)
    # A+A: all pairwise sums (with replacement)
    sums = a[:, None] + a[None, :]
    s_distinct = len(np.unique(sums.ravel()))
    
    # A*A: all pairwise products (with replacement)
    prods = a[:, None] * a[None, :]
    p_distinct = len(np.unique(prods.ravel()))
    
    return max(s_distinct, p_distinct)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_a = None
    best_score = float('inf')
    
    # Multiple diverse initializations
    inits = []
    
    # Powers of 2
    p2 = [2**i for i in range(20)]
    if p2[-1] <= 10000:
        inits.append(np.array(p2, dtype=np.int64))
    
    # Arithmetic progression
    inits.append(np.arange(1, 21, dtype=np.int64))
    
    # Geometric with ratio 3
    geo3 = [3**i for i in range(13)]  # 3^12 = 531441 > 10000, so up to 3^12
    geo3 = geo3[:13]
    while len(geo3) < 20:
        geo3.append(geo3[-1] * 2)
    geo3 = geo3[:20]
    if max(geo3) <= 10000:
        inits.append(np.array(geo3, dtype=np.int64))
    
    # Mixed: first 10 small, next 10 spread
    mixed1 = list(range(1, 11)) + [100, 200, 500, 1000, 2000, 4000, 6000, 8000, 9000, 9999]
    inits.append(np.array(mixed1, dtype=np.int64))
    
    # Primes
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71]
    inits.append(np.array(primes, dtype=np.int64))
    
    # Random inits
    random.seed(42)
    for _ in range(10):
        a = np.sort(random.sample(range(1, 10001), 20)).astype(np.int64)
        inits.append(a)
    
    # Evaluate all inits
    for a in inits:
        if a.max() > 10000 or a.min() < 1 or len(np.unique(a)) != 20:
            continue
        score = _evaluate(a)
        if score < best_score:
            best_score = score
            best_a = a.copy()
    
    # Local search with simulated annealing
    current_a = best_a.copy()
    current_score = best_score
    
    T = 50.0
    cooling = 0.999
    
    np.random.seed(123)
    iteration = 0
    
    while time.time() < deadline:
        iteration += 1
        
        # Choose move
        if np.random.random() < 0.6:
            # Swap
            i, j = np.random.choice(20, 2, replace=False)
            new_a = current_a.copy()
            new_a[i], new_a[j] = new_a[j], new_a[i]
            if len(np.unique(new_a)) != 20:
                continue
        elif np.random.random() < 0.8:
            # Replace
            i = np.random.choice(20)
            new_val = random.randint(1, 10000)
            new_a = current_a.copy()
            new_a[i] = new_val
            new_a = np.sort(new_a)
            if len(np.unique(new_a)) != 20 or new_a.max() > 10000:
                continue
        else:
            # Perturb: add random offset to random subset
            new_a = current_a.copy()
            k = np.random.choice(20, size=np.random.randint(1, 5), replace=False)
            for idx in k:
                offset = random.randint(-50, 50)
                new_a[idx] += offset
            new_a = np.sort(new_a)
            if len(np.unique(new_a)) != 20 or new_a.max() > 10000 or new_a.min() < 1:
                continue
        
        new_score = _evaluate(new_a)
        delta = new_score - current_score
        
        if delta <= 0 or np.random.random() < np.exp(-delta / max(T, 0.01)):
            current_a = new_a
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_a = new_a.copy()
        
        T *= cooling
        if T < 0.01:
            T = 20.0
            cooling = 0.999
    
    return list(best_a.astype(int))
