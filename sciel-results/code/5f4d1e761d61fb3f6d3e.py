def construct():
    import time
    import random
    import math
    
    deadline = time.time() + 110.0
    N = 511

    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    pindex = {p: i for i, p in enumerate(primes)}

    mult_mask = {}
    for p in primes:
        m = 0
        for b in range(p, N + 1, p):
            m |= (1 << b)
        mult_mask[p] = m

    pmask = [0] * (N + 1)
    conflict = [0] * (N + 1)
    for a in range(2, N + 1):
        x = a
        pm = 0
        cm = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                pm |= (1 << pindex[p])
                cm |= mult_mask[p]
                while x % p == 0:
                    x //= p
        if x > 1:
            pm |= (1 << pindex[x])
            cm |= mult_mask[x]
        pmask[a] = pm
        conflict[a] = cm

    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (512 - a)

    # Greedy initialization: pick highest weight non-conflicting elements
    S = set()
    S_mask = 0
    for a in range(N, 0, -1):
        if (conflict[a] & S_mask) == 0:
            S_mask |= (1 << a)
            S.add(a)

    def score():
        return sum(w[a] for a in S)

    best_score = score()
    best_S = S.copy()
    best_S_mask = S_mask

    # Local search with simulated annealing
    current_score = best_score
    T = 0.1  # initial temperature
    T_min = 1e-6
    cooling = 0.9997
    max_iter = 80000
    
    random.seed(42)
    
    # Precompute available candidates for faster lookup
    all_elements = list(range(1, N + 1))
    
    for iteration in range(max_iter):
        if time.time() >= deadline - 5.0:
            break
        if T < T_min:
            break
            
        # Random move: add, remove, or swap
        move_type = random.random()
        new_S_mask = S_mask
        delta = 0.0
        valid_move = False
        
        if move_type < 0.35:
            # Try to add a random non-conflicting element
            # Sample a few random candidates to avoid full scan
            for _ in range(20):
                a = random.choice(all_elements)
                if not (S_mask >> a) & 1 and (conflict[a] & S_mask) == 0:
                    new_S_mask |= (1 << a)
                    delta = w[a]
                    valid_move = True
                    break
        elif move_type < 0.55:
            # Try to remove a random element
            if S:
                x = random.choice(list(S))
                new_S_mask ^= (1 << x)
                delta = -w[x]
                valid_move = True
        else:
            # Try to swap: remove one, add another
            if S:
                x = random.choice(list(S))
                rest = S_mask ^ (1 << x)
                for _ in range(20):
                    a = random.choice(all_elements)
                    if not (S_mask >> a) & 1 and (conflict[a] & rest) == 0:
                        new_S_mask = rest | (1 << a)
                        delta = w[a] - w[x]
                        valid_move = True
                        break
        
        if not valid_move:
            continue
        
        # Acceptance criterion
        if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
            S_mask = new_S_mask
            # Reconstruct S from S_mask
            S = set()
            for a in range(1, N + 1):
                if (S_mask >> a) & 1:
                    S.add(a)
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_S = S.copy()
                best_S_mask = S_mask
        
        T *= cooling
    
    # Final greedy polish: try to add any remaining non-conflicting elements
    final_S = best_S.copy()
    final_S_mask = best_S_mask
    for a in range(N, 0, -1):
        if not (final_S_mask >> a) & 1 and (conflict[a] & final_S_mask) == 0:
            final_S_mask |= (1 << a)
            final_S.add(a)
    
    # Verify and return using final_S
    final_score = sum(w[a] for a in final_S)
    if final_score > best_score:
        best_S = final_S
        best_score = final_score
    
    # Second polish: try removing low-weight elements that block high-weight additions
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        current_mask = best_S_mask
        current_set = best_S
        current_sc = best_score
        
        for a in sorted(current_set, key=lambda x: w[x]):  # Try removing lowest weight first
            if time.time() >= deadline - 2.0:
                break
            test_mask = current_mask ^ (1 << a)
            test_sc = current_sc - w[a]
            
            # Try to add any element that was previously blocked
            for b in range(N, 0, -1):
                if not (test_mask >> b) & 1 and (conflict[b] & test_mask) == 0:
                    new_mask = test_mask | (1 << b)
                    new_sc = test_sc + w[b]
                    if new_sc > current_sc:
                        # Check if we can add more
                        for c in range(N, 0, -1):
                            if not (new_mask >> c) & 1 and (conflict[c] & new_mask) == 0:
                                new_mask |= (1 << c)
                                new_sc += w[c]
                        
                        new_set = set()
                        for x in range(1, N + 1):
                            if (new_mask >> x) & 1:
                                new_set.add(x)
                        
                        best_S = new_set
                        best_S_mask = new_mask
                        best_score = new_sc
                        improved = True
                        break
                break  # Only check highest b for this a
            
            if improved:
                break
    
    return list(best_S)
