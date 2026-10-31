import numpy as np
import time
import random

N = 50

def is_sidon(set_list):
    """Check if a list of integers forms a Sidon set."""
    sums = set()
    lst = sorted(set_list)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            s = lst[i] + lst[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_max_sidon(n, max_time=10.0):
    """Find the largest Sidon set in range(n)."""
    best_set = set()
    start_time = time.time()
    
    # Greedy initialization
    for attempt in range(100):
        current = set()
        # Randomize the order slightly or start from different points
        # Standard greedy: try 0..n-1
        for i in range(n):
            if is_sidon(list(current) + [i]):
                current.add(i)
        
        if len(current) > len(best_set):
            best_set = set(current)
            
        if time.time() - start_time > max_time / 2:
            break
            
    # Local search: try to add elements or swap to increase size
    # Since we want max size, we just try to add more.
    # If greedy gets stuck at size K, maybe a different starting set allows K+1.
    
    # Try random restarts with different random subsets as seeds
    for _ in range(50):
        if time.time() - start_time > max_time:
            break
        current = set()
        # Pick a random subset of size ~5 to start
        # Actually, greedy from scratch is usually good.
        # Let's try greedy but skipping some small numbers
        skip_prob = 0.1
        for i in range(n):
            if random.random() < skip_prob and i > 0:
                continue
            if is_sidon(list(current) + [i]):
                current.add(i)
        
        if len(current) > len(best_set):
            best_set = set(current)
            
    return list(best_set)

def construct():
    deadline = time.time() + 140.0
    
    # Find the largest Sidon set
    sidon_set = find_max_sidon(N, max_time=10.0)
    
    if not sidon_set:
        # Fallback to delta
        h = np.zeros(N)
        h[0] = 1.0
        return list(h)
        
    K = len(sidon_set)
    
    # Construct heights
    h = np.zeros(N)
    for idx in sidon_set:
        h[idx] = 1.0 / K
        
    # Verify score
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    linf = np.max(conv)
    score = l2_sq / (l1 * linf)
    
    # Theoretically score should be K/2.
    # Let's print or assert? No, just return.
    
    return list(h)
