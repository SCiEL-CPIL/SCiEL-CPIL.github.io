import time
import random
import numpy as np

def _score(A):
    """Compute max(|A+A|, |A*A|) for set A using NumPy vectorization."""
    A = np.array(A, dtype=np.int64)
    # A+A
    sums = A[:, None] + A[None, :]
    # A*A
    prods = A[:, None] * A[None, :]
    
    # Unique counts
    # np.unique on 400 elements is fast
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    
    return max(n_sums, n_prods)

def _is_valid(A, n=20, maximum=10000):
    """Check if A has n distinct positive integers <= maximum."""
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 for x in A):
        return False
    if any(x > maximum for x in A):
        return False
    return True

def _generate_smooth_numbers(n, max_val=10000):
    """Generate n smooth numbers (small prime factors) to minimize product set size."""
    primes = [2, 3, 5, 7, 11, 13]
    nums = set([1])
    # Simple sieve-like generation of smooth numbers
    for p in primes:
        for _ in range(10): # enough to generate plenty
            new_nums = set()
            for x in nums:
                y = x * p
                if y <= max_val:
                    new_nums.add(y)
            nums.update(new_nums)
            if len(nums) >= n * 2:
                break
        if len(nums) >= n * 2:
            break
    
    smooth_list = sorted(list(nums))
    if len(smooth_list) < n:
        # Fill with powers of 2 if not enough smooth numbers
        while len(smooth_list) < n:
            next_pow = 2 ** len(smooth_list)
            if next_pow > max_val:
                break
            if next_pow not in smooth_list:
                smooth_list.append(next_pow)
                smooth_list.sort()
    
    return smooth_list[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    rng = random.Random(42)
    
    # Initial candidates
    # 1. Powers of 2
    A_pow2 = [2**i for i in range(20)]
    # 2. Smooth numbers
    A_smooth = _generate_smooth_numbers(20)
    # 3. Random
    A_rand = sorted(rng.sample(range(1, 10001), 20))
    
    best_A = None
    best_score = float('inf')
    
    for A_init in [A_pow2, A_smooth, A_rand]:
        if not _is_valid(A_init):
            continue
        s = _score(A_init)
        if s < best_score:
            best_score = s
            best_A = A_init[:]
    
    if best_A is None:
        return sorted(A_pow2)
    
    # Simulated Annealing
    current_A = best_A[:]
    current_score = best_score
    T = 100.0
    cooling = 0.995
    min_T = 0.1
    
    while time.time() < deadline:
        if T < min_T:
            # Restart with lower T or break
            break
            
        # Generate neighbor
        new_A = current_A[:]
        
        # Strategy: Perturb one element or swap two
        choice = rng.random()
        if choice < 0.5:
            # Perturb one element
            i = rng.randint(0, 19)
            old_val = new_A[i]
            # Scale delta based on magnitude
            delta_range = max(10, int(old_val * 0.1))
            delta = rng.randint(-delta_range, delta_range)
            new_val = old_val + delta
            
            if 1 <= new_val <= 10000 and new_val not in new_A:
                new_A[i] = new_val
            else:
                continue
        else:
            # Swap two elements
            i, j = rng.sample(range(20), 2)
            new_A[i], new_A[j] = new_A[j], new_A[i]
        
        # Ensure validity (distinct, positive, <= 10000)
        if not _is_valid(new_A):
            continue
            
        new_score = _score(new_A)
        
        # Acceptance criterion
        if new_score < current_score:
            current_A = new_A
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = current_A[:]
        else:
            # Accept with probability exp(-delta / T)
            delta = new_score - current_score
            if delta == 0:
                prob = 1.0
            else:
                prob = np.exp(-delta / T)
            
            if rng.random() < prob:
                current_A = new_A
                current_score = new_score
        
        T *= cooling
        
    return sorted(best_A)
