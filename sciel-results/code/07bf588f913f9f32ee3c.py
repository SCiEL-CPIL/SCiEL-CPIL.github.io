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

    def generate_gp(a, r, n):
        A = []
        val = a
        for _ in range(n):
            if val > max_val:
                break
            if val not in A:
                A.append(val)
            val = int(val * r + 0.5)
        while len(A) < n:
            A.append(random.randint(1, max_val))
        return A[:n]

    def generate_ap(a, d, n):
        A = []
        for i in range(n):
            val = a + i * d
            if val > max_val:
                break
            if val not in A:
                A.append(val)
        while len(A) < n:
            A.append(random.randint(1, max_val))
        return A[:n]

    def generate_log_dense(n):
        A = set()
        while len(A) < n:
            base = random.choice([2, 3, 4])
            exp = random.randint(0, 10)
            val = base ** exp
            if 1 <= val <= max_val:
                val = max(1, min(max_val, val + random.randint(-2, 2)))
                A.add(val)
        return list(A)

    candidates = []
    
    for a in [1, 2, 3]:
        for r in [1.1, 1.2, 1.5, 2.0]:
            candidates.append(generate_gp(a, r, n))
            
    for a in [1, 2, 5]:
        for d in [1, 2, 3, 5, 10]:
            candidates.append(generate_ap(a, d, n))
            
    for _ in range(10):
        candidates.append(generate_log_dense(n))
        
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
            
    if best_set is None:
        best_set = list(range(1, n+1))
        best_score = get_score(best_set)

    current_set = best_set[:]
    current_score = best_score
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            
            move_type = random.random()
            
            if move_type < 0.5:
                i, j = random.sample(range(n), 2)
                new_set = current_set[:]
                new_set[i], new_set[j] = new_set[j], new_set[i]
            else:
                i = random.randint(0, n - 1)
                if random.random() < 0.5:
                    if n > 1:
                        x, y = random.sample(range(n), 2)
                        candidate_val = current_set[x] + current_set[y]
                        if 1 <= candidate_val <= max_val and candidate_val not in current_set:
                            new_set = current_set[:]
                            new_set[i] = candidate_val
                        else:
                            continue
                    else:
                        continue
                else:
                    if n > 1:
                        x, y = random.sample(range(n), 2)
                        p = current_set[x] * current_set[y]
                        factors = []
                        for f in range(1, int(math.isqrt(p)) + 1):
                            if p % f == 0:
                                factors.append(f)
                                if f != p // f:
                                    factors.append(p // f)
                        valid_factors = [f for f in factors if 1 <= f <= max_val and f not in current_set]
                        if valid_factors:
                            candidate_val = random.choice(valid_factors)
                            new_set = current_set[:]
                            new_set[i] = candidate_val
                        else:
                            continue
                    else:
                        continue

            if len(set(new_set)) != n:
                continue
                
            new_score = get_score(new_set)
            if new_score < current_score:
                current_set = new_set
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_set = current_set[:]
        
        if not improved:
            for _ in range(10):
                i = random.randint(0, n - 1)
                new_val = random.randint(1, max_val)
                if new_val not in current_set:
                    current_set[i] = new_val
                    current_score = get_score(current_set)
                    break

    return best_set
