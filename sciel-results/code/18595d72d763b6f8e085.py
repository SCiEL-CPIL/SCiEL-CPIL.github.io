"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _compute_sizes(A):
    """Compute |A+A| and |A*A| for set A."""
    A_list = list(A)
    n = len(A_list)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A_list[i]
        for j in range(i, n):
            aj = A_list[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return len(sums), len(prods)

def _score(A):
    s, p = _compute_sizes(A)
    return max(s, p)

def _generate_candidates():
    """Generate a diverse pool of candidate integers."""
    candidates = set()
    # Small integers
    for i in range(1, 101):
        candidates.add(i)
    # Highly composite and structured numbers
    for i in range(1, 101):
        for j in range(1, 101):
            p = i * j
            if p <= 10000:
                candidates.add(p)
    # Add some random numbers to ensure coverage
    for _ in range(500):
        candidates.add(random.randint(1, 10000))
    # Add numbers that are powers or near powers
    for base in range(2, 100):
        val = base
        while val <= 10000:
            candidates.add(val)
            candidates.add(val + 1)
            candidates.add(val - 1)
            val *= base
    # Filter and convert to list
    candidates = [x for x in candidates if 1 <= x <= 10000]
    return candidates

def _greedy_initialization(candidates, n=20):
    """Build an initial set greedily."""
    A = set()
    # Start with 1, as it often helps with products (x*1=x)
    # But 1 creates many distinct products if other numbers are distinct.
    # Let's try starting with small numbers.
    # Sort candidates by "density" or just use small ones first?
    # Let's try a few different starting points.
    
    best_A = None
    best_score = float('inf')
    
    # Strategy 1: Start with smallest
    A = set(range(1, 21))
    score = _score(A)
    if score < best_score:
        best_score = score
        best_A = set(A)
        
    # Strategy 2: Start with highly composite small numbers
    # Numbers with many divisors
    def num_divisors(x):
        count = 0
        for i in range(1, int(math.isqrt(x)) + 1):
            if x % i == 0:
                count += 1
                if i != x // i:
                    count += 1
        return count
        
    sorted_cands = sorted([c for c in candidates if c <= 100], key=lambda x: -num_divisors(x))
    A = set(sorted_cands[:20])
    score = _score(A)
    if score < best_score:
        best_score = score
        best_A = set(A)
        
    return best_A

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    candidates = _generate_candidates()
    
    # Initialize with greedy
    best_A = _greedy_initialization(candidates)
    best_score = _score(best_A)
    
    # Simulated Annealing
    A = set(best_A)
    current_score = best_score
    
    T = 10.0
    cooling = 0.999
    
    while (time.time() - start_time) < time_limit:
        if T < 0.1:
            break
            
        # Pick a random element to remove
        A_list = list(A)
        old_val = random.choice(A_list)
        
        # Pick a random candidate to add
        new_val = random.choice(candidates)
        
        if new_val == old_val or new_val in A:
            continue
            
        new_A = A.copy()
        new_A.discard(old_val)
        new_A.add(new_val)
        
        new_score = _score(new_A)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            A = new_A
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_A = set(A)
        
        T *= cooling
        
        # Occasionally reset temperature to escape deep local minima
        if random.random() < 0.01 and (time.time() - start_time) < time_limit - 5:
            T = 5.0
            
    return sorted(best_A)
# EVOLVE-BLOCK-END
