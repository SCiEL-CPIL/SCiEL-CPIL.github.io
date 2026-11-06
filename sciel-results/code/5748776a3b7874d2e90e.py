import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # 45s out of 60s
    
    n = 20
    max_val = 10000
    
    def score(A):
        A = list(set(A))
        if len(A) != n:
            return float('inf')
        if max(A) > max_val:
            return float('inf')
        sums = set()
        prods = set()
        for i in range(n):
            for j in range(n):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return max(len(sums), len(prods))
    
    best_score = float('inf')
    best_set = list(range(1, n+1))
    
    # Helper to generate GP
    def gen_gp(ratio, start, length):
        A = []
        val = start
        for _ in range(length):
            A.append(val)
            val *= ratio
        return A
    
    # Generate 2,3-smooth numbers (Hamming numbers) up to max_val
    smooth_numbers = []
    a = b = c = 0
    while len(smooth_numbers) < 200:
        next_val = min(2**a if a < 20 else float('inf'),
                       3**b if b < 20 else float('inf'),
                       6**(a//2) * 2**(a%2) if a < 20 else float('inf'))
        # Better: generate all 2^i * 3^j
        break
    
    # Generate 2,3-smooth numbers properly
    smooth = set()
    for i in range(15):
        for j in range(15):
            val = (2**i) * (3**j)
            if val <= max_val:
                smooth.add(val)
    smooth_list = sorted(smooth)
    
    # Candidate 1: Single GP
    for ratio in [2, 3]:
        for start in [1, 2]:
            A = gen_gp(ratio, start, n)
            if max(A) <= max_val:
                s = score(A)
                if s < best_score:
                    best_score = s
                    best_set = A
    
    # Candidate 2: Two clusters GP
    for ratio in [2, 3]:
        for start1 in [1, 2]:
            for start2 in [3, 4, 5, 10]:
                A1 = gen_gp(ratio, start1, 10)
                A2 = gen_gp(ratio, start2, 10)
                A = A1 + A2
                if len(set(A)) == n and max(A) <= max_val:
                    s = score(A)
                    if s < best_score:
                        best_score = s
                        best_set = A
    
    # Candidate 3: Subsets of 2,3-smooth numbers
    # Pick 20 distinct smooth numbers with good structure
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        # Mix of low and high smooth numbers
        A = random.sample(smooth_list, n)
        s = score(A)
        if s < best_score:
            best_score = s
            best_set = list(A)
    
    # Also try structured subsets: consecutive powers
    for base_combo in [(2,3), (3,2), (2,2,3), (3,3,2)]:
        A = []
        for i in range(n):
            # Cycle through bases
            base = base_combo[i % len(base_combo)]
            exp = i // len(base_combo) + 1
            val = base ** exp
            if val <= max_val and val not in A:
                A.append(val)
        if len(A) == n and max(A) <= max_val:
            s = score(A)
            if s < best_score:
                best_score = s
                best_set = list(A)
    
    # Simulated Annealing for global optimization
    def simulated_annealing(A, time_limit):
        import math
        A = list(A)
        current_score = score(A)
        best_score_sa = current_score
        best_set_sa = list(A)
        
        T = 10.0  # Initial temperature
        cooling_rate = 0.9995
        min_T = 0.1
        
        # Pre-compute a list of "good" candidate values to pick from
        # Mixing smooth numbers and randoms helps explore the space
        candidate_pool = smooth_list[:]
        # Add some random values to ensure coverage
        for _ in range(100):
            candidate_pool.append(random.randint(1, max_val))
        
        while time.time() < time_limit - 2.0: # Keep 2s margin
            if T < min_T:
                T = min_T
                
            # Decide move type: 1-swap or 2-swap
            if random.random() < 0.3:
                # 2-swap move
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i == j:
                    continue
                old_i, old_j = A[i], A[j]
                
                # Generate new values
                # Bias towards smooth numbers
                if random.random() < 0.7:
                    new_i = random.choice(smooth_list)
                    new_j = random.choice(smooth_list)
                else:
                    new_i = random.randint(1, max_val)
                    new_j = random.randint(1, max_val)
                    
                # Ensure distinctness
                if new_i == new_j or new_i == old_j or new_j == old_i:
                    continue
                # Check if new values collide with other elements
                existing = set(A)
                existing.discard(old_i)
                existing.discard(old_j)
                if new_i in existing or new_j in existing:
                    continue
                
                A[i] = new_i
                A[j] = new_j
                new_score = score(A)
                
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if current_score < best_score_sa:
                        best_score_sa = current_score
                        best_set_sa = list(A)
                else:
                    A[i] = old_i
                    A[j] = old_j
            else:
                # 1-swap move
                i = random.randint(0, n-1)
                old_i = A[i]
                
                if random.random() < 0.7:
                    new_i = random.choice(smooth_list)
                else:
                    new_i = random.randint(1, max_val)
                    
                if new_i == old_i:
                    continue
                existing = set(A)
                existing.discard(old_i)
                if new_i in existing:
                    continue
                
                A[i] = new_i
                new_score = score(A)
                
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if current_score < best_score_sa:
                        best_score_sa = current_score
                        best_set_sa = list(A)
                else:
                    A[i] = old_i
            
            T *= cooling_rate
            
        return best_set_sa, best_score_sa

    # Run Simulated Annealing on best found so far
    if time.time() < deadline - 10:
        best_set, best_score = simulated_annealing(best_set, deadline - 10)
        
    return best_set
