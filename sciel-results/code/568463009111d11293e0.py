import time
import random
import math

def calc_score(U):
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    n_sum = len(sum_set)
    n_diff = len(diff_set)
    
    if n_sum == 0:
        return 0.0
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0.0
        
    val = 1.0 + math.log(n_diff / n_sum) / denom
    return val

def normalize(U):
    if not U:
        return []
    min_u = min(U)
    return sorted([u - min_u for u in U])

def find_perfect_diff_set(p, k, trials=20000):
    for _ in range(trials):
        U = random.sample(range(p), k)
        diffs = set()
        valid = True
        for a in U:
            for b in U:
                if a != b:
                    d = (a - b) % p
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
            if not valid:
                break
        if valid and len(diffs) == p - 1:
            return U
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_set = [0, 1, 3]
    best_score = calc_score(best_set)
    
    # Try known perfect difference sets
    primes = [7, 13, 31, 43, 73, 109]
    for p in primes:
        k = 1
        while k*k - k + 1 < p:
            k += 1
        if k*k - k + 1 == p:
            U_p = find_perfect_diff_set(p, k, trials=10000)
            if U_p:
                U_norm = normalize(U_p)
                s = calc_score(U_norm)
                if s > best_score:
                    best_score = s
                    best_set = U_norm
        
        if time.time() > deadline:
            break
            
    # Hill Climbing
    current_set = list(best_set)
    current_score = best_score
    
    max_val = 200
    
    while time.time() < deadline:
        op = random.choice(['add', 'remove', 'change'])
        new_set = list(current_set)
        
        if op == 'add':
            candidate = random.randint(0, max_val)
            if candidate not in new_set:
                new_set.append(candidate)
            else:
                continue
        elif op == 'remove':
            if len(new_set) > 1:
                idx = random.randint(0, len(new_set) - 1)
                new_set.pop(idx)
            else:
                continue
        elif op == 'change':
            if len(new_set) > 0:
                idx = random.randint(0, len(new_set) - 1)
                new_val = random.randint(0, max_val)
                if new_val not in new_set:
                    new_set[idx] = new_val
                else:
                    continue
        
        new_set = normalize(new_set)
        new_score = calc_score(new_set)
        
        if new_score > current_score:
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
        
        if random.random() < 0.01:
            k = random.randint(3, 12)
            random_set = random.sample(range(max_val), k)
            random_set = normalize(random_set)
            random_score = calc_score(random_set)
            if random_score > current_score:
                current_set = random_set
                current_score = random_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)

    return best_set
