import time
import random
import numpy as np

def construct():
    deadline = time.time() + 55.0  # Safety margin
    n = 20
    max_val = 10000
    best_set = None
    best_score = float('inf')

    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A."""
        a = np.array(A, dtype=np.int64)
        # Vectorized sum and product sets
        sumset = a[:, None] + a[None, :]
        prodset = a[:, None] * a[None, :]
        
        # Use np.unique which is efficient for small arrays
        s_size = len(np.unique(sumset))
        p_size = len(np.unique(prodset))
        return max(s_size, p_size)

    def gen_structured_set(k, m, a, d, b, r):
        """
        Generate a set from AP of length k and GP of length m.
        AP: a, a+d, ..., a+(k-1)d
        GP: b, b*r, ..., b*r^(m-1)
        """
        ap = [a + i * d for i in range(k)]
        gp = [b * (r ** i) for i in range(m)]
        
        A = set(ap) | set(gp)
        # Filter valid values
        A = [x for x in A if 1 <= x <= max_val]
        
        # Check distinctness and size
        if len(A) < n:
            return None
        
        # If we have more than n, we need to trim? 
        # The task requires exactly n distinct integers.
        # We can just take the first n? Or optimize selection?
        # For now, if len(A) > n, we might have collisions or we can just pick a subset.
        # But the structure is what matters.
        # If len(A) == n, return it.
        # If len(A) > n, it's not a valid candidate for "exactly 20" unless we drop some.
        # Dropping some breaks the structure.
        # So we should aim for k+m = 20 and no collisions.
        
        if len(A) != n:
            return None
            
        return sorted(A)

    def local_search_structured(start_params, max_iter=5000):
        # start_params: (k, m, a, d, b, r)
        k, m, a, d, b, r = start_params
        current_set = gen_structured_set(k, m, a, d, b, r)
        if current_set is None:
            return None, float('inf')
            
        current_score = evaluate(current_set)
        local_best = current_set[:]
        local_best_score = current_score
        
        # Parameters to perturb
        # k, m are lengths (sum to 20)
        # a, d for AP
        # b, r for GP
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
                
            # Perturb one parameter
            new_k, new_m, new_a, new_d, new_b, new_r = k, m, a, d, b, r
            
            move_type = random.randint(0, 5)
            if move_type == 0:
                # Change k, m (keeping sum 20)
                delta = random.choice([-2, -1, 1, 2])
                new_k = k + delta
                new_m = m - delta
                if new_k < 5 or new_m < 5:
                    continue
            elif move_type == 1:
                # Change a (AP start)
                new_a = a + random.randint(-10, 10)
                if new_a < 1: new_a = 1
            elif move_type == 2:
                # Change d (AP step)
                new_d = d + random.choice([-2, -1, 1, 2])
                if new_d < 1: new_d = 1
            elif move_type == 3:
                # Change b (GP start)
                # Multiplicative or additive change
                if random.random() < 0.5:
                    new_b = b * random.choice([2, 3, 1//2, 1//3]) # Integer division issues
                    if new_b < 1: new_b = 1
                else:
                    new_b = b + random.randint(-50, 50)
                    if new_b < 1: new_b = 1
            elif move_type == 4:
                # Change r (GP ratio)
                new_r = r + random.choice([-1, 1])
                if new_r < 2: new_r = 2
            else:
                # Random reset of one component
                if random.random() < 0.5:
                    new_a = random.randint(1, 50)
                    new_d = random.randint(1, 5)
                else:
                    new_b = random.randint(1, 100)
                    new_r = random.randint(2, 4)

            # Ensure distinctness of AP and GP internally
            # AP is distinct if d > 0.
            # GP is distinct if r > 1.
            
            new_set = gen_structured_set(new_k, new_m, new_a, new_d, new_b, new_r)
            if new_set is None:
                continue
                
            new_score = evaluate(new_set)
            
            if new_score < current_score:
                current_set = new_set
                current_score = new_score
                k, m, a, d, b, r = new_k, new_m, new_a, new_d, new_b, new_r
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best = current_set[:]
            else:
                # Simulated annealing
                if random.random() < 0.1:
                    if new_score < current_score * 1.2:
                        current_set = new_set
                        current_score = new_score
                        k, m, a, d, b, r = new_k, new_m, new_a, new_d, new_b, new_r

        return local_best, local_best_score

    # Initial Candidates
    candidates = []
    
    # 1. AP + GP with small numbers
    # k=10, m=10
    # AP: 1..10 (a=1, d=1)
    # GP: 11, 22, 44... (b=11, r=2)
    candidates.append((10, 10, 1, 1, 11, 2))
    
    # 2. AP + GP with larger GP start to avoid collision
    # AP: 1..10
    # GP: 100, 200, 400...
    candidates.append((10, 10, 1, 1, 100, 2))
    
    # 3. Smaller AP, Larger GP
    # k=8, m=12
    # AP: 1..8
    # GP: 10, 20, 40...
    candidates.append((8, 12, 1, 1, 10, 2))
    
    # 4. k=12, m=8
    # AP: 1..12
    # GP: 13, 26, 52...
    candidates.append((12, 8, 1, 1, 13, 2))
    
    # 5. Different ratios
    candidates.append((10, 10, 1, 1, 11, 3))
    candidates.append((10, 10, 1, 2, 20, 2)) # AP: 1,3,5..
    
    # Random starts for structured search
    for _ in range(10):
        k = random.randint(8, 12)
        m = 20 - k
        a = random.randint(1, 10)
        d = random.randint(1, 5)
        b = random.randint(10, 100)
        r = random.randint(2, 4)
        candidates.append((k, m, a, d, b, r))

    # Run local search from each candidate
    for params in candidates:
        if time.time() > deadline:
            break
            
        res_set, res_score = local_search_structured(params)
        
        if res_set is not None and res_score < best_score:
            best_score = res_score
            best_set = res_set[:]

    # Fallback if structured search fails
    if best_set is None:
        best_set = list(range(1, 21))

    return best_set
