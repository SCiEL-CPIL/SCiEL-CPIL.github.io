import time
import random

def _compute_objective(A):
    """Compute max(|A+A|, |A*A|) for set A."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return max(len(sums), len(prods))

def _generate_pool(max_val=10000):
    """Generate a pool of smooth numbers."""
    pool = set()
    pool.add(1)
    # 2^a * 3^b * 5^c * 7^d * 11^e * 13^f
    for a in range(14):
        v = 1
        for b in range(9):
            v2 = v
            for c in range(6):
                v3 = v2
                for d in range(5):
                    v4 = v3
                    for e in range(4):
                        v5 = v4
                        for f in range(3):
                            val = v5
                            if val <= max_val:
                                pool.add(val)
                            v5 *= 13
                        v4 *= 11
                    v3 *= 7
                v2 *= 5
            v *= 3
        v *= 2
    
    # Add small numbers for diversity
    for x in range(1, 101):
        pool.add(x)
    
    # Add pairwise and triple products of small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    for i, p in enumerate(small_primes):
        for q in small_primes[i:]:
            if p * q <= max_val:
                pool.add(p * q)
                for r in small_primes:
                    if p * q * r <= max_val:
                        pool.add(p * q * r)
    
    pool = sorted([x for x in pool if 1 <= x <= max_val])
    return pool

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    pool = _generate_pool(10000)
    pool_list = pool
    pool_set = set(pool_list)
    
    # Initial: small integers
    best_A = list(range(1, 21))
    best_score = _compute_objective(best_A)
    
    # Try some structured initial sets
    init_candidates = [
        list(range(1, 21)),
        sorted(pool_list[:20]),
    ]
    # Add a geometric-ish set
    geo = []
    x = 1
    while len(geo) < 20 and x <= 10000:
        geo.append(x)
        x = int(x * 1.2) + 1
    if len(geo) == 20:
        init_candidates.append(sorted(geo))
    
    for init_set in init_candidates:
        if len(init_set) != 20 or len(set(init_set)) != 20:
            continue
        s = _compute_objective(init_set)
        if s < best_score:
            best_score = s
            best_A = list(init_set)
    
    # Local search
    random.seed(42)
    current_A = list(best_A)
    current_set = set(current_A)
    current_score = best_score
    
    def fast_obj(A):
        n = len(A)
        sums = set()
        prods = set()
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                sums.add(ai + A[j])
                prods.add(ai * A[j])
        return max(len(sums), len(prods))
    
    no_improve = 0
    stall_limit = 1500
    
    while time.time() < deadline:
        improved_this_round = False
        
        # Try 30 random swap attempts per round
        for _ in range(30):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, 19)
            old_val = current_A[idx]
            
            # Pick a replacement
            new_val = random.choice(pool_list)
            if new_val in current_set:
                continue
            
            new_A = current_A[:]
            new_A[idx] = new_val
            new_score = fast_obj(new_A)
            
            if new_score < current_score:
                current_set.discard(old_val)
                current_set.add(new_val)
                current_A[idx] = new_val
                current_score = new_score
                improved_this_round = True
                no_improve = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_A = list(current_A)
        
        if not improved_this_round:
            no_improve += 1
            if no_improve > stall_limit:
                # Perturb and restart
                if time.time() > deadline - 5:
                    break
                current_A = list(best_A)
                current_set = set(current_A)
                current_score = best_score
                # Random perturbation
                for _ in range(random.randint(3, 6)):
                    idx = random.randint(0, 19)
                    old_val = current_A[idx]
                    new_val = random.choice(pool_list)
                    if new_val not in current_set:
                        current_set.discard(old_val)
                        current_set.add(new_val)
                        current_A[idx] = new_val
                current_score = fast_obj(current_A)
                no_improve = 0
    
    # Final polish: single swaps from best_A (limited)
    if time.time() < deadline - 2:
        best_set = set(best_A)
        # Sample a subset of pool for polishing
        sample_pool = random.sample(pool_list, min(200, len(pool_list)))
        for i in range(20):
            if time.time() > deadline - 2:
                break
            old_val = best_A[i]
            for new_val in sample_pool:
                if new_val in best_set:
                    continue
                test_A = list(best_A)
                test_A[i] = new_val
                s = fast_obj(test_A)
                if s < best_score:
                    best_score = s
                    best_A = test_A
                    best_set = set(best_A)
    
    return best_A
