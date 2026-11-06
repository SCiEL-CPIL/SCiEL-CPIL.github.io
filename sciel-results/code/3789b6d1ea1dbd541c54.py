import math
import random
import time

def construct():
    deadline = time.time() + 115.0
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
    
    # Weights
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (C - a)
        
    # Precompute conflicts
    conflicts = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if math.gcd(a, b) > 1:
                conflicts[a].add(b)
                conflicts[b].add(a)
                
    # Precompute prime factors for each number
    prime_factors = [set() for _ in range(N + 1)]
    for a in range(2, N + 1):
        for p in primes:
            if p * p > a:
                break
            if a % p == 0:
                prime_factors[a].add(p)
                while a % p == 0:
                    a //= p
        if a > 1:
            prime_factors[a].add(a)
    
    # Helper functions
    def can_add(a, S_set):
        return not (S_set & conflicts[a])
        
    def get_score(S_set):
        return sum(w[a] for a in S_set)
        
    def find_best_add(S_set):
        best_a = None
        best_w = -1
        for a in range(1, N + 1):
            if a in S_set:
                continue
            if can_add(a, S_set):
                if w[a] > best_w:
                    best_w = w[a]
                    best_a = a
        return best_a
    
    def find_k_best_add(S_set, k=3):
        candidates = []
        for a in range(1, N + 1):
            if a in S_set:
                continue
            if can_add(a, S_set):
                candidates.append((w[a], a))
        candidates.sort(reverse=True)
        return [a for _, a in candidates[:k]]
        
    def find_best_2_swap_greedy(S):
        best_delta = 0
        best_move = None
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                x = S_list[i]
                y = S_list[j]
                new_S = S - {x, y}
                u = find_best_add(new_S)
                if u is None:
                    continue
                new_S_u = new_S | {u}
                v = find_best_add(new_S_u)
                if v is None:
                    continue
                delta = w[u] + w[v] - w[x] - w[y]
                if delta > best_delta:
                    best_delta = delta
                    best_move = (x, y, u, v)
        return best_move, best_delta
    
    # Greedy initialization with multiple strategies
    def greedy_init(strategy='weight', rng=None):
        S = set()
        S.add(1)
        
        if strategy == 'weight':
            candidates = list(range(N, 0, -1))
        elif strategy == 'ratio':
            candidates = list(range(N, 0, -1))
            def heuristic(a):
                if a == 1:
                    return 0
                damage = sum(w[b] for b in conflicts[a] if w[b] > 0.1)
                return w[a] / (1.0 + damage)
            candidates.sort(key=heuristic, reverse=True)
        elif strategy == 'random':
            candidates = list(range(N, 0, -1))
            if rng:
                rng.shuffle(candidates)
        elif strategy == 'bipartite':
            # Bipartite matching-inspired greedy:
            # Left: primes < 512, Right: integers 1..511
            # Edge p-a if p|a, weight w(a)
            # Greedy: iteratively select integer with highest weight-to-conflict ratio
            # where conflict = number of shared prime factors with already-selected set
            candidates = list(range(N, 0, -1))
            # Score each candidate by w(a) / (1 + sum of weights of conflicting selected)
            # This mimics the bipartite matching intuition: prefer numbers that don't
            # "compete" for the same prime factors as already-chosen numbers
            def bipartite_heuristic(a):
                if a == 1:
                    return 0.0
                # Count how many of its prime factors are "used" by selected numbers
                used_count = 0
                total_conflict_weight = 0.0
                for p in prime_factors[a]:
                    for b in S:
                        if b != a and p in prime_factors[b]:
                            used_count += 1
                            total_conflict_weight += w[b]
                            break
                return w[a] / (1.0 + 0.5 * total_conflict_weight)
            # Iterative selection: at each step, pick the best remaining candidate
            remaining = list(range(N, 0, -1))
            while remaining:
                best_a = None
                best_score = -1
                for a in remaining:
                    if a in S:
                        continue
                    if not can_add(a, S):
                        continue
                    sc = bipartite_heuristic(a)
                    if sc > best_score:
                        best_score = sc
                        best_a = a
                if best_a is None:
                    break
                S.add(best_a)
                remaining = [a for a in remaining if a != best_a]
            return S
        else:
            candidates = list(range(N, 0, -1))
            
        for a in candidates:
            if a in S:
                continue
            if can_add(a, S):
                S.add(a)
                
        return S
    
    # Local search with multiple move types
    def local_search(S, max_iter=200, rng=None):
        if rng is None:
            rng = random.Random(42)
            
        current_score = get_score(S)
        best_S = S.copy()
        best_score = current_score
        
        improved = True
        iteration = 0
        while improved and iteration < max_iter and time.time() < deadline - 5.0:
            improved = False
            iteration += 1
            S_list = list(S)
            rng.shuffle(S_list)
            
            # Try 1-swap (remove x, add y)
            for x in S_list:
                if time.time() > deadline - 5.0:
                    break
                new_S = S - {x}
                y = find_best_add(new_S)
                if y is not None:
                    delta = w[y] - w[x]
                    if delta > 1e-9:
                        S = new_S | {y}
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_S = S.copy()
                        improved = True
                        break
                        
            if improved:
                continue

            # Try 2-swap greedy (remove x,y add u,v)
            move, delta = find_best_2_swap_greedy(S)
            if move is not None and delta > 1e-9:
                x, y, u, v = move
                S = (S - {x, y}) | {u, v}
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_S = S.copy()
                improved = True
                continue

            # Try 1-remove-2-add
            for x in S_list:
                if time.time() > deadline - 5.0:
                    break
                new_S = S - {x}
                y_candidates = find_k_best_add(new_S, k=5)
                for y in y_candidates:
                    new_S_y = new_S | {y}
                    z = find_best_add(new_S_y)
                    if z is not None:
                        delta = w[y] + w[z] - w[x]
                        if delta > 1e-9:
                            S = new_S_y | {z}
                            current_score += delta
                            if current_score > best_score:
                                best_score = current_score
                                best_S = S.copy()
                            improved = True
                            break
                if improved:
                    break
                    
            # Try 2-remove-1-add
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    if time.time() > deadline - 5.0:
                        break
                    x = S_list[i]
                    y = S_list[j]
                    new_S = S - {x, y}
                    u = find_best_add(new_S)
                    if u is not None:
                        delta = w[u] - w[x] - w[y]
                        if delta > 1e-9:
                            S = new_S | {u}
                            current_score += delta
                            if current_score > best_score:
                                best_score = current_score
                                best_S = S.copy()
                            improved = True
                            break
                if improved:
                    break
                    
        return best_S, best_score
    
    # Try multiple initializations and keep the best
    best_overall_S = set()
    best_overall_score = 0.0
    
    rng = random.Random(42)
    
    # Strategy 1: Pure weight greedy
    S1 = greedy_init('weight', rng)
    S1, score1 = local_search(S1, max_iter=300, rng=rng)
    if score1 > best_overall_score:
        best_overall_score = score1
        best_overall_S = S1.copy()
        
    # Strategy 2: Ratio greedy
    S2 = greedy_init('ratio', rng)
    S2, score2 = local_search(S2, max_iter=300, rng=rng)
    if score2 > best_overall_score:
        best_overall_score = score2
        best_overall_S = S2.copy()
        
    # Strategy 3: Bipartite matching-inspired greedy
    S_bip = greedy_init('bipartite', rng)
    S_bip, score_bip = local_search(S_bip, max_iter=300, rng=rng)
    if score_bip > best_overall_score:
        best_overall_score = score_bip
        best_overall_S = S_bip.copy()
        
    # Strategy 4: Random orderings
    for restart in range(8):
        if time.time() > deadline - 10.0:
            break
        S3 = greedy_init('random', rng)
        S3, score3 = local_search(S3, max_iter=150, rng=rng)
        if score3 > best_overall_score:
            best_overall_score = score3
            best_overall_S = S3.copy()
            
    # Final polish: try to add any remaining coprime numbers
    for a in range(N, 0, -1):
        if a not in best_overall_S and can_add(a, best_overall_S):
            best_overall_S.add(a)
            best_overall_score += w[a]
            
    # One more local search pass on the polished set
    best_overall_S, best_overall_score = local_search(best_overall_S, max_iter=100, rng=rng)
    
    return list(best_overall_S)
