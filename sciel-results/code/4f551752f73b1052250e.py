import math
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140.0  # Safety margin
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
    
    # Precompute prime factors for each number
    pf = [[] for _ in range(N + 1)]
    for p in primes:
        for multiple in range(p, N + 1, p):
            pf[multiple].append(p)
    
    # Build conflict masks using bitsets
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        if a == 1:
            continue
        for b in range(a + 1, N + 1):
            if not pf[a] or not pf[b]:
                continue
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
    
    def can_add_to_set(S_set, a):
        """Check if a can be added to S_set"""
        for b in S_set:
            if (conflict_mask[b] >> a) & 1:
                return False
        return True
    
    # Strategy 1: Greedy prime-matching heuristic
    # Model as: for each prime p, choose at most one number divisible by p
    # Numbers not divisible by any prime (i.e., 1) are always available
    # Greedy: iterate primes in order of max potential weight, pick best compatible number
    
    def greedy_prime_matching():
        """Greedy heuristic based on prime-conflict structure"""
        S = set()
        S.add(1)  # 1 is coprime to everything
        
        # For each prime, find the best number divisible by it that's compatible
        # Sort primes by the max weight of numbers they divide (descending)
        prime_max_weight = {}
        for p in primes:
            max_w = 0.0
            for multiple in range(p, N + 1, p):
                if w[multiple] > max_w:
                    max_w = w[multiple]
            prime_max_weight[p] = max_w
        
        # Process primes in order of max weight descending
        sorted_primes = sorted(primes, key=lambda p: -prime_max_weight[p])
        
        for p in sorted_primes:
            # Find best number divisible by p that's compatible with S
            best_a = None
            best_w = -1
            for a in range(N, p - 1, -1):  # Try high-weight numbers first
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
    
    # Get initial solutions from both strategies
    S_pm = greedy_prime_matching()
    S_gw = greedy_by_weight()
    
    best_score = max(get_score(S_pm), get_score(S_gw))
    best_S = S_pm.copy() if get_score(S_pm) >= get_score(S_gw) else S_gw.copy()
    
    rng = random.Random(42)
    all_vertices = list(range(1, N + 1))
    
    # Multi-swap hill climbing with simulated annealing
    def multi_swap_hill_climb(S_init, time_limit):
        S = S_init.copy()
        current_score = get_score(S)
        best_S_local = S.copy()
        best_score_local = current_score
        
        T = 0.05
        T_min = 1e-9
        cooling = 0.999
        
        step = 0
        while time.time() < time_limit and T > T_min:
            step += 1
            
            # Choose swap type: 1-swap (60%), 2-swap (30%), 3-swap (10%)
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
            
            # Pick vertices to remove
            to_remove = rng.sample(list(S), swap_size)
            new_S = S.copy()
            for x in to_remove:
                new_S.discard(x)
            
            # Find best compatible additions
            # Get all candidates that can be added to new_S
            candidates = []
            for a in all_vertices:
                if a in new_S:
                    continue
                if can_add_to_set(new_S, a):
                    candidates.append(a)
            
            if len(candidates) < swap_size:
                T *= cooling
                continue
            
            # Sort candidates by weight descending
            candidates.sort(key=lambda a: -w[a])
            
            # Try to find best set of swap_size compatible additions
            best_addition = None
            best_addition_score = -1
            
            # For small swap sizes, enumerate combinations of top candidates
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
                # For 3-swap, try top few combinations
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
            
            # Calculate delta
            removed_score = sum(w[x] for x in to_remove)
            added_score = best_addition_score
            delta = added_score - removed_score
            
            # Accept if improvement or with probability exp(delta/T)
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
    
    # Run multi-swap hill climbing from both initial solutions
    remaining_time = deadline - time.time() - 10.0  # Leave 10s safety margin
    if remaining_time > 5.0:
        # Split time between two runs
        time_per_run = remaining_time / 2.0
        
        S1, score1 = multi_swap_hill_climb(S_pm, time.time() + time_per_run)
        if score1 > best_score:
            best_score = score1
            best_S = S1
        
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 5.0:
            S2, score2 = multi_swap_hill_climb(S_gw, time.time() + remaining_time)
            if score2 > best_score:
                best_score = score2
                best_S = S2
    
    # Final polish: try single-add improvements
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for a in all_vertices:
            if a in best_S:
                continue
            if can_add_to_set(best_S, a):
                # Check if adding improves (it always does since w[a] > 0)
                best_S.add(a)
                improved = True
    
    return list(best_S)
