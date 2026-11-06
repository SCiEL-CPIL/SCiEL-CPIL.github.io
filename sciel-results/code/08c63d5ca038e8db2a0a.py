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
    num_primes = len(primes)
    
    # Weights: w[a] = 1/(512 - a)
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (C - a)
    
    # Precompute prime factors for each number
    pf = [[] for _ in range(N + 1)]
    for p in primes:
        for multiple in range(p, N + 1, p):
            pf[multiple].append(p)
    
    # Build conflict masks using bitsets
    conflict_mask = [0] * (N + 1)
    for a in range(2, N + 1):
        if not pf[a]:
            continue
        for b in range(a + 1, N + 1):
            if not pf[b]:
                continue
            common = False
            for p in pf[a]:
                if p in pf[b]:
                    common = True
                    break
            if common:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    # For each prime, precompute which vertices are divisible by it
    prime_to_vertices = {p: [] for p in primes}
    for a in range(2, N + 1):
        for p in pf[a]:
            prime_to_vertices[p].append(a)
    
    def get_score(selected):
        return sum(w[a] for a in selected)
    
    def can_add_to_set(S_set, a):
        for b in S_set:
            if (conflict_mask[b] >> a) & 1:
                return False
        return True
    
    # Strategy 1: Greedy prime-matching heuristic
    def greedy_prime_matching():
        S = set()
        S.add(1)
        
        prime_max_weight = {}
        for p in primes:
            max_w = 0.0
            for multiple in range(p, N + 1, p):
                if w[multiple] > max_w:
                    max_w = w[multiple]
            prime_max_weight[p] = max_w
        
        sorted_primes = sorted(primes, key=lambda p: -prime_max_weight[p])
        
        for p in sorted_primes:
            best_a = None
            best_w = -1
            for a in range(N, p - 1, -1):
                if a % p != 0:
                    continue
                if a in S:
                    continue
                if can_add_to_set(S, a):
                    if w[a] > best_w:
                        best_w = w[a]
                        best_a = a
            if best_a is not None:
                S.add(best_a)
        
        return S
    
    # Strategy 2: Standard greedy by weight
    def greedy_by_weight():
        S = set()
        S.add(1)
        for a in range(N, 1, -1):
            if a in S:
                continue
            if can_add_to_set(S, a):
                S.add(a)
        return S
    
    # Strategy 3: Greedy with conflict-aware ordering
    def greedy_conflict_aware():
        S = set()
        S.add(1)
        remaining = set(range(2, N + 1))
        
        while remaining:
            # For each candidate, compute its weight minus penalty for conflicts with S
            best_a = None
            best_score = -1
            for a in remaining:
                if not can_add_to_set(S, a):
                    continue
                # Score is just w[a] since we can only add if no conflict
                if w[a] > best_score:
                    best_score = w[a]
                    best_a = a
            if best_a is None:
                break
            S.add(best_a)
            remaining.discard(best_a)
            # Remove all that conflict with best_a
            for b in list(remaining):
                if (conflict_mask[best_a] >> b) & 1:
                    remaining.discard(b)
        
        return S
    
    # Get initial solutions from all strategies
    S_pm = greedy_prime_matching()
    S_gw = greedy_by_weight()
    S_ca = greedy_conflict_aware()
    
    candidates = [(get_score(S_pm), S_pm), (get_score(S_gw), S_gw), (get_score(S_ca), S_ca)]
    best_score, best_S = max(candidates, key=lambda x: x[0])
    
    # Branch and bound with tighter upper bound
    def branch_and_bound(initial_set, time_limit):
        best_set = initial_set.copy()
        best_weight = get_score(initial_set)
        
        # Order vertices by weight descending
        vertices_by_weight = sorted(range(1, N + 1), key=lambda a: -w[a])
        
        nodes_explored = 0
        
        # Tighter upper bound: for each prime, at most one vertex divisible by it can be chosen
        # UB = sum over primes p of (max weight of available vertex divisible by p)
        # This is valid because if a vertex is divisible by multiple primes, it's counted multiple times,
        # but since we can only pick one per prime, the sum of max-per-prime is >= true max
        def compute_upper_bound(avail_mask):
            ub = 0.0
            for p in primes:
                max_w_p = 0.0
                for a in prime_to_vertices[p]:
                    if (avail_mask >> a) & 1:
                        if w[a] > max_w_p:
                            max_w_p = w[a]
                ub += max_w_p
            return ub
        
        def bnb(avail_mask, current_set, current_weight, depth):
            nonlocal best_set, best_weight, nodes_explored
            
            if time.time() > deadline:
                return
            
            nodes_explored += 1
            
            if avail_mask == 0:
                if current_weight > best_weight:
                    best_weight = current_weight
                    best_set = current_set.copy()
                return
            
            ub = compute_upper_bound(avail_mask)
            if ub <= best_weight + 1e-12:
                return
            
            # Find highest-weight available vertex
            best_v = -1
            for a in vertices_by_weight:
                if (avail_mask >> a) & 1:
                    best_v = a
                    break
            
            if best_v == -1:
                return
            
            # Option 1: Include best_v
            new_set = current_set.copy()
            new_set.add(best_v)
            new_weight = current_weight + w[best_v]
            
            new_avail = avail_mask & ~(1 << best_v)
            conflicts = conflict_mask[best_v]
            new_avail &= ~conflicts
            
            bnb(new_avail, new_set, new_weight, depth + 1)
            
            # Option 2: Exclude best_v
            new_avail2 = avail_mask & ~(1 << best_v)
            bnb(new_avail2, current_set, current_weight, depth + 1)
        
        # Start with empty set, all vertices available
        initial_avail = 0
        for a in range(1, N + 1):
            initial_avail |= (1 << a)
        
        bnb(initial_avail, set(), 0.0, 0)
        
        return best_set, best_weight
    
    # Run branch and bound
    if time.time() < deadline - 20.0:
        S_bnb, score_bnb = branch_and_bound(best_S, deadline - 20.0)
        if score_bnb > best_score:
            best_score = score_bnb
            best_S = S_bnb
    
    # Multi-swap hill climbing with simulated annealing
    rng = random.Random(42)
    all_vertices = list(range(1, N + 1))
    
    def multi_swap_hill_climb(S_init, time_limit):
        S = S_init.copy()
        current_score = get_score(S)
        best_S_local = S.copy()
        best_score_local = current_score
        
        T = 0.05
        T_min = 1e-9
        cooling = 0.999
        
        while time.time() < time_limit and T > T_min:
            r = rng.random()
            if r < 0.6:
                swap_size = 1
            elif r < 0.9:
                swap_size = 2
            else:
                swap_size = 3
            
            if len(S) < swap_size:
                T *= cooling
                continue
            
            to_remove = rng.sample(list(S), swap_size)
            new_S = S.copy()
            for x in to_remove:
                new_S.discard(x)
            
            candidates = []
            for a in all_vertices:
                if a in new_S:
                    continue
                if can_add_to_set(new_S, a):
                    candidates.append(a)
            
            if len(candidates) < swap_size:
                T *= cooling
                continue
            
            candidates.sort(key=lambda a: -w[a])
            
            best_addition = None
            best_addition_score = -1
            
            top_k = min(15, len(candidates))
            if swap_size == 1:
                if candidates:
                    best_addition = [candidates[0]]
                    best_addition_score = w[candidates[0]]
            elif swap_size == 2:
                for i in range(top_k):
                    a1 = candidates[i]
                    for j in range(i + 1, top_k):
                        a2 = candidates[j]
                        if (conflict_mask[a1] >> a2) & 1:
                            continue
                        if can_add_to_set(new_S, a1) and can_add_to_set(new_S | {a1}, a2):
                            score = w[a1] + w[a2]
                            if score > best_addition_score:
                                best_addition_score = score
                                best_addition = [a1, a2]
            elif swap_size == 3:
                top_k3 = min(8, len(candidates))
                for i in range(top_k3):
                    a1 = candidates[i]
                    for j in range(i + 1, top_k3):
                        a2 = candidates[j]
                        if (conflict_mask[a1] >> a2) & 1:
                            continue
                        for k in range(j + 1, top_k3):
                            a3 = candidates[k]
                            if (conflict_mask[a1] >> a3) & 1 or (conflict_mask[a2] >> a3) & 1:
                                continue
                            if can_add_to_set(new_S, a1) and can_add_to_set(new_S | {a1}, a2) and can_add_to_set(new_S | {a1, a2}, a3):
                                score = w[a1] + w[a2] + w[a3]
                                if score > best_addition_score:
                                    best_addition_score = score
                                    best_addition = [a1, a2, a3]
            
            if best_addition is None:
                T *= cooling
                continue
            
            removed_score = sum(w[x] for x in to_remove)
            added_score = best_addition_score
            delta = added_score - removed_score
            
            if delta > 0 or (T > T_min and rng.random() < math.exp(delta / T)):
                for a in best_addition:
                    new_S.add(a)
                S = new_S
                current_score += delta
                
                if current_score > best_score_local:
                    best_score_local = current_score
                    best_S_local = S.copy()
            
            T *= cooling
        
        return best_S_local, best_score_local
    
    # Run multi-swap hill climbing
    remaining_time = deadline - time.time() - 8.0
    if remaining_time > 5.0:
        S1, score1 = multi_swap_hill_climb(best_S, time.time() + remaining_time)
        if score1 > best_score:
            best_score = score1
            best_S = S1
    
    # Final exact polish: try all single-add improvements until no more can be made
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for a in all_vertices:
            if a in best_S:
                continue
            if can_add_to_set(best_S, a):
                best_S.add(a)
                improved = True
    
    return list(best_S)
