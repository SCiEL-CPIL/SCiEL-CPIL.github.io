import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    def score(A):
        # A is a tuple or list of 20 distinct ints
        # Use efficient set building
        n = len(A)
        sum_set = set()
        prod_set = set()
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sum_set.add(ai + aj)
                prod_set.add(ai * aj)
        return max(len(sum_set), len(prod_set))
    
    def score_incremental(A, changed_idx):
        """Score with some optimization - still O(n^2) but cleaner"""
        n = len(A)
        sum_set = set()
        prod_set = set()
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sum_set.add(ai + aj)
                prod_set.add(ai * aj)
        return max(len(sum_set), len(prod_set))
    
    def generate_initial():
        candidates = []
        
        # 1. Small AP
        candidates.append(tuple(range(1, 21)))
        
        # 2. Denser AP
        candidates.append(tuple(range(2, 41, 2)))  # 2,4,6,...,40
        
        # 3. Multiplicative structure: powers of 2 up to limit, then fill
        p2 = [2**i for i in range(14)]  # 1,2,4,...,8192
        # Need 20 elements, have 14. Add some in between
        remaining = 20 - len(p2)
        # Add multiples that create collisions
        extra = []
        for k in range(3, 20):
            if k not in p2 and len(extra) < remaining:
                extra.append(k)
        candidates.append(tuple(sorted(p2 + extra)))
        
        # 4. Geometric-like with small ratio
        gp = [1]
        for i in range(19):
            next_val = int(gp[-1] * 1.25) + 1
            if next_val > 10000:
                next_val = 10000
            if next_val in gp:
                next_val = gp[-1] + 1
            gp.append(next_val)
        if len(set(gp)) == 20 and max(gp) <= 10000:
            candidates.append(tuple(gp))
        
        # 5. Random small numbers (encourage sum collisions)
        for _ in range(10):
            candidates.append(tuple(random.sample(range(1, 200), 20)))
        
        # 6. Random mixed
        for _ in range(10):
            small = random.sample(range(1, 100), 10)
            large = random.sample(range(100, 10001), 10)
            candidates.append(tuple(sorted(small + large)))
        
        # 7. Structured: multiples of a base
        for base in [2, 3, 5]:
            mults = random.sample(range(1, 10001 // base + 1), 20)
            cands = tuple(sorted(base * m for m in mults))
            if max(cands) <= 10000 and len(set(cands)) == 20:
                candidates.append(cands)
        
        # Evaluate and return best
        best_c = None
        best_s = float('inf')
        for c in candidates:
            if len(set(c)) != 20 or max(c) > 10000 or min(c) < 1:
                continue
            s = score(c)
            if s < best_s:
                best_s = s
                best_c = c
        return best_c, best_s
    
    current_A, current_score = generate_initial()
    best_A = current_A
    best_score = current_score
    
    # Simulated Annealing with better moves
    T = 15.0
    T_min = 0.5
    alpha = 0.9995
    
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random index to modify
        idx = random.randint(0, 19)
        old_val = current_A[idx]
        
        # Generate new value with better strategy
        r = random.random()
        if r < 0.3:
            # Small numbers for sum collisions
            new_val = random.randint(1, 50)
        elif r < 0.5:
            # Medium range
            new_val = random.randint(50, 500)
        elif r < 0.7:
            # Large range
            new_val = random.randint(500, 10000)
        elif r < 0.85:
            # Near existing values (for product collisions)
            other_idx = random.randint(0, 19)
            if other_idx == idx:
                other_idx = (idx + 1) % 20
            base = current_A[other_idx]
            new_val = base + random.randint(-10, 10)
            new_val = max(1, min(10000, new_val))
        else:
            # Multiplicative perturbation of existing value
            factor = random.choice([0.5, 0.75, 1.25, 1.5, 2.0, 0.8, 1.2])
            new_val = int(old_val * factor)
            new_val = max(1, min(10000, new_val))
            
        if new_val == old_val:
            continue
            
        # Check distinctness
        if new_val in current_A:
            continue
            
        # Create new tuple
        new_A_list = list(current_A)
        new_A_list[idx] = new_val
        new_A = tuple(new_A_list)
        
        new_score = score(new_A)
        
        # Acceptance criterion
        if new_score < current_score:
            current_A = new_A
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_A = current_A
        else:
            no_improve += 1
            # Metropolis criterion
            delta = new_score - current_score
            if T > T_min:
                # Use log for numerical stability
                log_accept = -delta / T
                if log_accept > -700:  # Avoid underflow
                    if random.random() < 2.718281828459045 ** log_accept:
                        current_A = new_A
                        current_score = new_score
        
        T *= alpha
        if T < T_min:
            # Reheat if stuck
            if no_improve > 2000:
                T = 10.0
                no_improve = 0
            else:
                T = T_min
    
    # Final polish: try small perturbations around best
    if time.time() < deadline - 5:
        polish_A = list(best_A)
        polish_score = best_score
        for _ in range(2000):
            if time.time() >= deadline - 2:
                break
            idx = random.randint(0, 19)
            old_val = polish_A[idx]
            new_val = old_val + random.randint(-5, 5)
            new_val = max(1, min(10000, new_val))
            if new_val == old_val or new_val in polish_A:
                continue
            polish_A[idx] = new_val
            new_score = score(tuple(polish_A))
            if new_score < polish_score:
                polish_score = new_score
                if polish_score < best_score:
                    best_score = polish_score
                    best_A = tuple(polish_A)
            else:
                polish_A[idx] = old_val
    
    return list(best_A)
