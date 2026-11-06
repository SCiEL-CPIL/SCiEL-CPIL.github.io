import math
import time
import random

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    N = 511
    C = 512
    
    # Sieve for primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Weights: w[a] = 1/(512 - a)
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (C - a)
    
    # Precompute prime factor masks for each number
    # Use a list of lists for prime factors
    pf = [[] for _ in range(N + 1)]
    for p in primes:
        for multiple in range(p, N + 1, p):
            pf[multiple].append(p)
    
    # Build conflict masks using bitsets
    # conflict_mask[a] has bits set for all b that conflict with a
    conflict_mask = [0] * (N + 1)
    
    for a in range(1, N + 1):
        if a == 1:
            continue
        for b in range(a + 1, N + 1):
            if not pf[a] or not pf[b]:
                continue
            # Check if they share a prime factor
            common = False
            for p in pf[a]:
                if p in pf[b]:
                    common = True
                    break
            if common:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    def get_score(selected):
        """selected is a set of vertices"""
        return sum(w[a] for a in selected)
    
    # Greedy initialization with multiple strategies
    def greedy_init(order):
        S = set()
        S.add(1)  # 1 is coprime to everything
        for a in order:
            if a in S:
                continue
            # Check if a is compatible with all in S
            can_add = True
            for b in S:
                if (conflict_mask[b] >> a) & 1:
                    can_add = False
                    break
            if can_add:
                S.add(a)
        return S
    
    # Try different greedy orders
    orders = [
        list(range(N, 0, -1)),  # descending
        list(range(2, N + 1)),  # ascending (excluding 1)
    ]
    
    best_S = None
    best_score = -1
    for order in orders:
        S_greedy = greedy_init(order)
        score = get_score(S_greedy)
        if score > best_score:
            best_score = score
            best_S = S_greedy.copy()
    
    # Also try a randomized greedy
    rng = random.Random(42)
    for _ in range(50):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S_greedy = greedy_init(order)
        score = get_score(S_greedy)
        if score > best_score:
            best_score = score
            best_S = S_greedy.copy()
    
    # Local search: simulated annealing
    all_vertices = list(range(1, N + 1))
    
    S = best_S.copy()
    current_score = best_score
    best_S_sa = S.copy()
    best_score_sa = current_score
    
    T = 0.1
    T_min = 1e-8
    cooling = 0.9995
    
    def can_add_to_set(S_set, a):
        """Check if a can be added to S_set"""
        for b in S_set:
            if (conflict_mask[b] >> a) & 1:
                return False
        return True
    
    step = 0
    while time.time() < deadline - 10.0 and T > T_min:
        step += 1
        
        # Random swap: remove one, add one
        if S:
            # Pick a vertex to remove
            x = rng.choice(list(S))
            new_S = S.copy()
            new_S.discard(x)
            
            # Find best vertex to add
            best_add = None
            best_add_w = -1
            for a in all_vertices:
                if a in new_S:
                    continue
                if can_add_to_set(new_S, a):
                    if w[a] > best_add_w:
                        best_add_w = w[a]
                        best_add = a
            
            if best_add is not None:
                new_S.add(best_add)
                delta = w[best_add] - w[x]
                
                if delta > 0 or rng.random() < math.exp(delta / T):
                    S = new_S
                    current_score += delta
                    if current_score > best_score_sa:
                        best_score_sa = current_score
                        best_S_sa = S.copy()
                        if best_score_sa > best_score:
                            best_score = best_score_sa
                            best_S = S.copy()
        
        T *= cooling
    
    # Try 2-swap moves: remove two, add two
    for _ in range(100):  # Limited iterations for 2-swap
        if time.time() > deadline - 5.0:
            break
        
        if len(S) >= 2:
            x, y = rng.sample(list(S), 2)
            new_S = S.copy()
            new_S.discard(x)
            new_S.discard(y)
            
            # Find best pair to add
            candidates = []
            for a in all_vertices:
                if a in new_S:
                    continue
                if can_add_to_set(new_S, a):
                    candidates.append(a)
            
            if len(candidates) >= 2:
                # Sort by weight descending
                candidates.sort(key=lambda a: -w[a])
                # Try top few pairs
                for i in range(min(10, len(candidates))):
                    for j in range(i + 1, min(10, len(candidates))):
                        a1 = candidates[i]
                        a2 = candidates[j]
                        if (conflict_mask[a1] >> a2) & 1:
                            continue
                        # Check a1 and a2 don't conflict with new_S
                        if can_add_to_set(new_S, a1) and can_add_to_set(new_S | {a1}, a2):
                            delta = w[a1] + w[a2] - w[x] - w[y]
                            if delta > 0:
                                new_S.add(a1)
                                new_S.add(a2)
                                current_score += delta
                                S = new_S
                                if current_score > best_score_sa:
                                    best_score_sa = current_score
                                    best_S_sa = S.copy()
                                    if best_score_sa > best_score:
                                        best_score = best_score_sa
                                        best_S = S.copy()
                                break
                    else:
                        continue
                    break
    
    return list(best_S)
