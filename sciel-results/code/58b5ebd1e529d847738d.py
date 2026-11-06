import random
import time
import math

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def get_violations(S):
        S_set = set(S)
        violations = 0
        items = list(S)
        for i in range(len(items)):
            a = items[i]
            for j in range(i, len(items)):
                b = items[j]
                c = a + b
                if c in S_set:
                    violations += 1
        return violations

    def repair_set(S):
        S = set(S)
        while True:
            # Find all violations
            violations = []
            items = list(S)
            for i in range(len(items)):
                a = items[i]
                for j in range(i, len(items)):
                    b = items[j]
                    c = a + b
                    if c in S:
                        violations.append((a, b, c))
            
            if not violations:
                break
            
            # Count frequency of each element in violations
            freq = {}
            for a, b, c in violations:
                freq[a] = freq.get(a, 0) + 1
                freq[b] = freq.get(b, 0) + 1
                freq[c] = freq.get(c, 0) + 1
            
            # Remove the element with the highest frequency
            worst_elem = max(freq.keys(), key=lambda k: freq[k])
            S.remove(worst_elem)
            
        return S

    def max_sum_free_sa(vals, time_limit=10.0):
        n = len(vals)
        if n == 0:
            return []
        
        rng = random.Random(12345)
        
        # Initial solution: Random subset of size n/2
        current_S = set(rng.sample(vals, n // 2))
        
        def score(S):
            v = get_violations(S)
            if v == 0:
                return len(S)
            else:
                return -10000 + len(S) - 10 * v

        current_score = score(current_S)
        best_S = set(current_S)
        best_score = current_score
        
        start_time = time.time()
        T = 100.0
        cooling = 0.995
        
        while T > 0.01 and (time.time() - start_time) < time_limit:
            # Generate a neighbor
            move_type = rng.random()
            new_S = set(current_S)
            
            if move_type < 0.33:
                # Add a random element
                candidates = [x for x in vals if x not in new_S]
                if candidates:
                    x = rng.choice(candidates)
                    new_S.add(x)
            elif move_type < 0.66:
                # Remove a random element
                if new_S:
                    x = rng.choice(list(new_S))
                    new_S.remove(x)
            else:
                # Swap
                if new_S:
                    to_remove = rng.choice(list(new_S))
                    candidates = [x for x in vals if x not in new_S]
                    if candidates:
                        to_add = rng.choice(candidates)
                        new_S.remove(to_remove)
                        new_S.add(to_add)
            
            new_score = score(new_S)
            delta = new_score - current_score
            
            if delta >= 0 or rng.random() < math.exp(delta / T):
                current_S = new_S
                current_score = new_score
                if current_score > best_score:
                    best_S = set(current_S)
                    best_score = current_score
            
            T *= cooling
            
        # Final repair to ensure validity
        best_S = repair_set(best_S)
        
        return list(best_S)

    results = []
    time_alloc = {60: 15.0, 120: 25.0, 240: 70.0}
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        t_limit = time_alloc.get(size, 10.0)
        subset = max_sum_free_sa(vals, t_limit)
        results.append(subset)
        
    return results
