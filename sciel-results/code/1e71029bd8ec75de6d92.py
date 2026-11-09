import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    # Precompute all pairs (i, j) with i <= j for n=20
    # There are n*(n+1)/2 = 210 pairs
    pairs_i = np.array([i for i in range(n) for j in range(i, n)], dtype=np.int32)
    pairs_j = np.array([j for i in range(n) for j in range(i, n)], dtype=np.int32)
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using NumPy vectorization."""
        A_arr = np.array(A, dtype=np.int64)
        sums = A_arr[pairs_i] + A_arr[pairs_j]
        prods = A_arr[pairs_i] * A_arr[pairs_j]
        sumset_size = len(np.unique(sums))
        prodset_size = len(np.unique(prods))
        return max(sumset_size, prodset_size)

    def is_valid_set(A):
        if len(A) != n:
            return False
        if len(set(A)) != n:
            return False
        if min(A) < 1 or max(A) > max_val:
            return False
        return True

    # Helper to generate a candidate set from parameters
    def get_smooth_pool():
        pool = set()
        for a in range(0, 14):
            v2 = 1 << a
            if v2 > max_val: break
            for b in range(0, 9):
                v3 = v2 * (3 ** b)
                if v3 > max_val: break
                for c in range(0, 6):
                    v5 = v3 * (5 ** c)
                    if v5 > max_val: break
                    for d in range(0, 5):
                        v7 = v5 * (7 ** d)
                        if v7 > max_val: break
                        pool.add(v7)
        return sorted(pool)
    
    smooth_pool = get_smooth_pool()
    
    # Generate initial candidates from structured families
    candidates = []
    
    # 1. Smallest smooth numbers
    if len(smooth_pool) >= n:
        candidates.append(smooth_pool[:n])
        
    # 2. Geometric progressions with integer steps
    for r in [1.1, 1.2, 1.5, 2.0, 3.0]:
        for start in [1, 2, 5, 10]:
            seq = []
            curr = float(start)
            for _ in range(n):
                val = int(round(curr))
                if val < 1: val = 1
                if val > max_val: break
                if val not in seq:
                    seq.append(val)
                curr *= r
            if len(seq) == n:
                candidates.append(seq)
                
    # 3. Arithmetic progressions
    for start in [1, 2, 5, 10, 50]:
        for step in [1, 2, 5, 10, 50]:
            seq = [start + i * step for i in range(n)]
            if max(seq) <= max_val:
                candidates.append(seq)
                
    # 4. Random candidates
    for _ in range(10):
        candidates.append(sorted(random.sample(range(1, max_val + 1), n)))
        
    # Evaluate initial candidates
    best_score = float('inf')
    best_set = None
    
    for cand in candidates:
        if not is_valid_set(cand):
            continue
        score = evaluate(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
            
    if best_set is None:
        best_set = random.sample(range(1, max_val + 1), n)
        best_score = evaluate(best_set)
        
    current_set = best_set[:]
    current_score = best_score
    
    # Precompute a list of all possible pairs of indices for swap moves
    # There are n*(n-1)/2 = 190 pairs
    index_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute a pool of "interesting" values to swap into the set
    # This pool should cover the range but be dense enough to find improvements
    # We use a mix of smooth numbers, a grid, and random samples
    swap_pool = set()
    swap_pool.update(smooth_pool)
    # Add a grid of values
    for v in range(1, max_val + 1, 10):
        swap_pool.add(v)
    # Add some random values
    for _ in range(2000):
        swap_pool.add(random.randint(1, max_val))
    swap_pool = list(swap_pool)
    
    # Simulated Annealing with Swap Moves
    # Temperature schedule
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.995
    
    T = T_start
    
    while T > T_end and time.time() < deadline - 5:
        # Perform a fixed number of moves per temperature step
        # We want to do as many as possible within the time budget
        # Let's estimate: 190 pairs * len(swap_pool) evaluations is too many.
        # Instead, we pick random pairs and random values from the pool.
        
        # Number of moves per T step: ~1000
        moves_per_step = 1000
        
        for _ in range(moves_per_step):
            if time.time() > deadline - 5:
                break
            
            # Pick a random pair of indices to swap
            i, j = random.choice(index_pairs)
            
            # Pick a random value from the pool to replace current_set[i]
            # And another to replace current_set[j]? 
            # Or just swap values between i and j? No, that doesn't change the set.
            # The move is: replace A[i] with x and A[j] with y, where x, y are from the pool
            # and distinct from each other and the rest of the set.
            
            # To keep it simple and fast, let's try replacing A[i] with a random value from the pool
            # and A[j] with another random value from the pool.
            
            old_i = current_set[i]
            old_j = current_set[j]
            
            # Sample two distinct values from the pool
            # Ensure they are not in the current set (except for the positions being replaced)
            # This check is expensive. Let's just try and validate.
            
            x = random.choice(swap_pool)
            y = random.choice(swap_pool)
            
            if x == y:
                continue
            if x == old_i or x == old_j:
                x = random.choice(swap_pool)
            if y == old_i or y == old_j:
                y = random.choice(swap_pool)
            if x == y:
                continue
                
            # Check if x or y are in the rest of the set
            # This is O(n).
            rest_set = set(current_set)
            rest_set.discard(old_i)
            rest_set.discard(old_j)
            
            if x in rest_set or y in rest_set:
                continue
                
            # Apply move
            current_set[i] = x
            current_set[j] = y
            
            new_score = evaluate(current_set)
            
            # Accept or reject based on Metropolis criterion
            delta = new_score - current_score
            if delta < 0:
                # Improvement
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current_set[:]
            else:
                # Worse
                # Accept with probability exp(-delta / T)
                if random.random() < math.exp(-delta / T):
                    current_score = new_score
                else:
                    # Revert
                    current_set[i] = old_i
                    current_set[j] = old_j
            
        T *= cooling_rate
        
    # Final polish with single-element moves if time permits
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(n):
                old_val = current_set[i]
                best_new_val = None
                best_new_score = current_score
                current_vals = set(current_set)
                
                # Try a subset of the pool for single replacement
                # To save time, try only a few random ones
                for _ in range(20):
                    new_val = random.choice(swap_pool)
                    if new_val == old_val or new_val in current_vals:
                        continue
                    current_set[i] = new_val
                    new_score = evaluate(current_set)
                    current_set[i] = old_val
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                
                if best_new_val is not None:
                    current_set[i] = best_new_val
                    current_score = best_new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_set = current_set[:]

    return sorted(best_set)
