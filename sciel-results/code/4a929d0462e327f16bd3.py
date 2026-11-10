import random
import time
import math

def construct():
    n = 20
    max_val = 10000
    time_limit = 55.0
    
    def get_score(A):
        sums = set()
        prods = set()
        for i in range(n):
            ai = A[i]
            for j in range(n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))

    # Initialize with a few diverse candidates and pick the best
    candidates = []
    
    # Geometric Progressions
    for a in [1, 2, 3, 5]:
        for r in [1.2, 1.5, 1.8, 2.0, 2.5]:
            A = []
            val = float(a)
            for _ in range(n):
                if val > max_val:
                    break
                iv = int(val + 0.5)
                if iv not in A:
                    A.append(iv)
                val *= r
            while len(A) < n:
                A.append(random.randint(1, max_val))
            candidates.append(A[:n])

    # Arithmetic Progressions
    for a in [1, 2, 5]:
        for d in [1, 2, 5, 10, 20]:
            A = []
            for i in range(n):
                val = a + i * d
                if val > max_val:
                    break
                if val not in A:
                    A.append(val)
            while len(A) < n:
                A.append(random.randint(1, max_val))
            candidates.append(A[:n])
            
    # Random
    for _ in range(5):
        candidates.append(random.sample(range(1, max_val + 1), n))

    best_set = None
    best_score = float('inf')
    for cand in candidates:
        if len(set(cand)) != n or any(x < 1 or x > max_val for x in cand):
            continue
        score = get_score(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]

    current_set = best_set[:]
    current_score = best_score
    
    # Simulated Annealing
    T = 1000.0
    T_min = 1.0
    alpha = 0.999
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            T = T_min
            
        # Generate a neighbor
        move_type = random.random()
        new_set = current_set[:]
        
        if move_type < 0.6:
            # Replace one element
            i = random.randint(0, n - 1)
            if random.random() < 0.5:
                # Random value
                new_val = random.randint(1, max_val)
            else:
                # Derived value (sum or product of two existing)
                x, y = random.sample(range(n), 2)
                if random.random() < 0.5:
                    new_val = current_set[x] + current_set[y]
                else:
                    new_val = current_set[x] * current_set[y]
                    if new_val > max_val:
                        # Try a factor
                        p = new_val
                        factors = []
                        for f in range(1, int(math.isqrt(p)) + 1):
                            if p % f == 0:
                                factors.append(f)
                                if f != p // f:
                                    factors.append(p // f)
                        valid_factors = [f for f in factors if 1 <= f <= max_val]
                        if valid_factors:
                            new_val = random.choice(valid_factors)
                        else:
                            new_val = random.randint(1, max_val)
            
            if new_val != new_set[i]:
                if new_val in new_set:
                    continue
                new_set[i] = new_val
        elif move_type < 0.8:
            # Swap two elements
            i, j = random.sample(range(n), 2)
            new_set[i], new_set[j] = new_set[j], new_set[i]
        else:
            # Perturb one element slightly
            i = random.randint(0, n - 1)
            delta = random.randint(-5, 5)
            new_val = new_set[i] + delta
            if 1 <= new_val <= max_val and new_val not in new_set:
                new_set[i] = new_val
            else:
                continue

        if len(set(new_set)) != n:
            continue

        new_score = get_score(new_set)
        delta_score = new_score - current_score
        
        # Acceptance criterion
        if delta_score < 0 or random.random() < math.exp(-delta_score / T):
            current_set = new_set
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
        
        T *= alpha

    return best_set
