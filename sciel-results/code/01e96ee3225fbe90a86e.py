import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A"""
        n = len(A)
        sums = set()
        prods = set()
        for i in range(n):
            for j in range(i, n):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return max(len(sums), len(prods))
    
    def evaluate_with_details(A):
        n = len(A)
        sums = set()
        prods = set()
        for i in range(n):
            for j in range(i, n):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return max(len(sums), len(prods)), len(sums), len(prods)
    
    # Initial candidates
    def geometric_init(r):
        A = [r**i for i in range(1, 21)]
        if max(A) <= 10000:
            return sorted(A)
        return None
    
    def arithmetic_init(a, d):
        A = [a + i*d for i in range(20)]
        if max(A) <= 10000 and min(A) >= 1:
            return sorted(A)
        return None
    
    # Try various initial structures
    candidates = []
    
    # Geometric progressions
    for r in range(2, 11):
        g = geometric_init(r)
        if g:
            candidates.append(g)
    
    # Arithmetic progressions
    for a in range(1, 50):
        for d in range(1, 500):
            ap = arithmetic_init(a, d)
            if ap:
                candidates.append(ap)
    
    # Mixed: powers of 2 times small factors
    for base in range(2, 20):
        A = sorted(set([base**i for i in range(1, 11)] + [base**i * 3 for i in range(1, 11)]))
        if len(A) >= 20:
            A = A[:20]
            if len(set(A)) == 20 and max(A) <= 10000:
                candidates.append(sorted(A))
    
    # Smooth numbers (only small prime factors)
    def get_smooth_numbers(limit=10000, n=20):
        primes = [2, 3, 5, 7]
        smooth = set()
        def gen(idx, val):
            if len(smooth) > 200:
                return
            if val > limit:
                return
            if idx == len(primes):
                if val >= 1:
                    smooth.add(val)
                return
            for p in primes[idx:]:
                gen(idx+1, val*p)
                val *= p
                if val <= limit:
                    gen(idx+1, val)
        gen(0, 1)
        smooth_list = sorted(smooth)
        if len(smooth_list) >= n:
            return smooth_list[:n]
        return None
    
    smooth = get_smooth_numbers()
    if smooth:
        candidates.append(smooth)
    
    # Evaluate all candidates and pick best
    if not candidates:
        return list(range(1, 21))
    
    best_A = None
    best_score = float('inf')
    for c in candidates:
        if len(c) != 20 or len(set(c)) != 20:
            continue
        if max(c) > 10000 or min(c) < 1:
            continue
        score = evaluate(c)
        if score < best_score:
            best_score = score
            best_A = list(c)
    
    if best_A is None:
        return list(range(1, 21))
    
    # Simulated annealing with nudge/swap moves
    current_A = best_A[:]
    current_score = best_score
    best_A_sa = best_A[:]
    best_score_sa = best_score
    
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    max_iter = 500000
    
    rng = random.Random(42)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        if T < T_min:
            break
        
        # Choose move type
        move_type = rng.random()
        new_A = current_A[:]
        
        if move_type < 0.5:
            # Nudge: change one element slightly
            idx = rng.randint(0, 19)
            delta = rng.randint(-50, 50)
            new_val = current_A[idx] + delta
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in current_A:
                continue
            new_A[idx] = new_val
        elif move_type < 0.8:
            # Swap: exchange two elements with different values
            i, j = rng.sample(range(20), 2)
            new_A[i], new_A[j] = new_A[j], new_A[i]
        else:
            # Replace one element with a new random value
            idx = rng.randint(0, 19)
            new_val = rng.randint(1, 10000)
            if new_val in current_A:
                continue
            new_A[idx] = new_val
        
        # Validate
        if len(set(new_A)) != 20 or max(new_A) > 10000 or min(new_A) < 1:
            continue
        
        new_score = evaluate(new_A)
        delta_score = new_score - current_score
        
        if delta_score < 0 or rng.random() < pow(2.718281828, -delta_score / T):
            current_A = new_A
            current_score = new_score
            if current_score < best_score_sa:
                best_score_sa = current_score
                best_A_sa = current_A[:]
        
        T *= cooling
    
    return sorted(best_A_sa)
