import math
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85.0
    N = 256

    # 1. Preprocessing
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        x = i
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                prime_factors[i].add(p)
                while x % p == 0:
                    x //= p
        if x > 1:
            prime_factors[i].add(x)
    prime_factors[1] = set()
    
    pf_frozen = [frozenset(prime_factors[i]) for i in range(N + 1)]
    
    # 2. Fast greedy initialization (multiple strategies, pick best)
    def greedy_init(strategy):
        best_set = {1}
        used_primes = set()
        if strategy == 0:  # descending
            candidates = list(range(N, 1, -1))
        elif strategy == 1:  # descending by number of prime factors (fewer first)
            candidates = sorted(range(N, 1, -1), key=lambda x: (len(pf_frozen[x]), -x))
        elif strategy == 2:  # descending, but prefer primes first
            primes_list = [p for p in range(N, 1, -1) if is_prime[p]]
            composites = [x for x in range(N, 1, -1) if not is_prime[x]]
            candidates = primes_list + composites
        else:  # random-ish (deterministic shuffle)
            candidates = list(range(N, 1, -1))
            # Simple deterministic "shuffle"
            for i in range(len(candidates) - 1, 0, -1):
                j = (i * 7919) % (i + 1)
                candidates[i], candidates[j] = candidates[j], candidates[i]
        
        for n in candidates:
            if not (pf_frozen[n] & used_primes):
                best_set.add(n)
                used_primes |= pf_frozen[n]
        return best_set
    
    best_set = None
    best_sum = -1
    for strat in range(4):
        s = greedy_init(strat)
        sm = sum(s)
        if sm > best_sum:
            best_sum = sm
            best_set = s
    
    # 3. Aggressive local search
    used_primes = set()
    for s in best_set:
        used_primes |= pf_frozen[s]
    
    def try_add_single(best_set, used_primes):
        for n in range(N, 1, -1):
            if n not in best_set and len(pf_frozen[n] & used_primes) == 0:
                best_set.add(n)
                used_primes = used_primes | pf_frozen[n]
                return n
        return None
    
    def try_swap_1for1(best_set, used_primes):
        best_gain = 0
        best_move = None
        for n_out in list(best_set):
            if n_out == 1:
                continue
            freed = pf_frozen[n_out]
            remaining = used_primes - freed
            for n_in in range(N, 1, -1):
                if n_in in best_set:
                    continue
                if len(pf_frozen[n_in] & remaining) == 0:
                    gain = n_in - n_out
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (n_out, n_in)
                    if n_in <= n_out:
                        break
        if best_move:
            n_out, n_in = best_move
            best_set.remove(n_out)
            best_set.add(n_in)
            used_primes = (used_primes - pf_frozen[n_out]) | pf_frozen[n_in]
            return best_gain
        return 0
    
    def try_swap_2for1(best_set, used_primes):
        best_gain = 0
        best_move = None
        lst = [x for x in best_set if x != 1]
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                freed = pf_frozen[a] | pf_frozen[b]
                remaining = used_primes - freed
                for n_in in range(N, 1, -1):
                    if n_in in best_set:
                        continue
                    if len(pf_frozen[n_in] & remaining) == 0:
                        gain = n_in - a - b
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (a, b, n_in)
                        if n_in <= max(a, b):
                            break
        if best_move:
            a, b, n_in = best_move
            best_set.remove(a)
            best_set.remove(b)
            best_set.add(n_in)
            used_primes = (used_primes - pf_frozen[a] - pf_frozen[b]) | pf_frozen[n_in]
            return best_gain
        return 0
    
    def try_swap_1for2(best_set, used_primes):
        best_gain = 0
        best_move = None
        for n_out in list(best_set):
            if n_out == 1:
                continue
            freed = pf_frozen[n_out]
            remaining = used_primes - freed
            # Try to add two numbers that are coprime to each other and to remaining
            for n_in1 in range(N, 1, -1):
                if n_in1 in best_set:
                    continue
                if len(pf_frozen[n_in1] & remaining) > 0:
                    continue
                rem2 = remaining | pf_frozen[n_in1]
                for n_in2 in range(n_in1 - 1, 1, -1):
                    if n_in2 in best_set:
                        continue
                    if len(pf_frozen[n_in2] & rem2) > 0:
                        continue
                    gain = n_in1 + n_in2 - n_out
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (n_out, n_in1, n_in2)
                    if n_in1 + n_in2 <= n_out:
                        break
        if best_move:
            n_out, n_in1, n_in2 = best_move
            best_set.remove(n_out)
            best_set.add(n_in1)
            best_set.add(n_in2)
            used_primes = (used_primes - pf_frozen[n_out]) | pf_frozen[n_in1] | pf_frozen[n_in2]
            return best_gain
        return 0
    
    def try_swap_2for2(best_set, used_primes):
        best_gain = 0
        best_move = None
        lst = [x for x in best_set if x != 1]
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                freed = pf_frozen[a] | pf_frozen[b]
                remaining = used_primes - freed
                for n_in1 in range(N, 1, -1):
                    if n_in1 in best_set:
                        continue
                    if len(pf_frozen[n_in1] & remaining) > 0:
                        continue
                    rem2 = remaining | pf_frozen[n_in1]
                    for n_in2 in range(n_in1 - 1, 1, -1):
                        if n_in2 in best_set:
                            continue
                        if len(pf_frozen[n_in2] & rem2) > 0:
                            continue
                        gain = n_in1 + n_in2 - a - b
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (a, b, n_in1, n_in2)
                        if n_in1 + n_in2 <= a + b:
                            break
        if best_move:
            a, b, n_in1, n_in2 = best_move
            best_set.remove(a)
            best_set.remove(b)
            best_set.add(n_in1)
            best_set.add(n_in2)
            used_primes = (used_primes - pf_frozen[a] - pf_frozen[b]) | pf_frozen[n_in1] | pf_frozen[n_in2]
            return best_gain
        return 0
    
    unchanged = 0
    max_unchanged = 100
    while unchanged < max_unchanged and time.time() < deadline - 15.0:
        improved = False
        gain = 0
        
        # Try all move types in order of likelihood
        n = try_add_single(best_set, used_primes)
        if n is not None:
            gain += n
            improved = True
        
        if not improved:
            g = try_swap_1for1(best_set, used_primes)
            if g > 0:
                gain += g
                improved = True
        
        if not improved:
            g = try_swap_2for1(best_set, used_primes)
            if g > 0:
                gain += g
                improved = True
        
        if not improved:
            g = try_swap_1for2(best_set, used_primes)
            if g > 0:
                gain += g
                improved = True
        
        if not improved:
            g = try_swap_2for2(best_set, used_primes)
            if g > 0:
                gain += g
                improved = True
        
        if not improved:
            unchanged += 1
        else:
            unchanged = 0
    
    best_sum = sum(best_set)
    best_result = list(best_set)
    
    # 4. Try MILP with remaining time as a final polish
    remaining_time = deadline - time.time() - 3.0
    if remaining_time > 5.0:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            
            indices = list(range(2, N + 1))
            num_vars = len(indices)
            num_primes = len(primes)
            
            A = np.zeros((num_primes, num_vars), dtype=np.int8)
            for idx, p in enumerate(primes):
                for v_idx, n in enumerate(indices):
                    if n % p == 0:
                        A[idx, v_idx] = 1
            
            c = np.zeros(num_vars, dtype=np.float64)
            for v_idx, n in enumerate(indices):
                c[v_idx] = -float(n)
            
            cons = LinearConstraint(A, lb=0, ub=1)
            bounds = Bounds(lb=0, ub=1)
            integrality = np.ones(num_vars, dtype=np.int8)
            
            res = milp(c, constraints=cons, integrality=integrality, bounds=bounds,
                       options={'time_limit': remaining_time})
            
            if res.x is not None:
                sol_set = {1}
                for v_idx, val in enumerate(res.x):
                    if val > 0.5:
                        sol_set.add(indices[v_idx])
                
                # Validate
                lst = list(sol_set)
                valid = True
                for i in range(len(lst)):
                    for j in range(i + 1, len(lst)):
                        if math.gcd(lst[i], lst[j]) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and sum(sol_set) > best_sum:
                    best_result = lst
                    best_sum = sum(sol_set)
        except Exception:
            pass
    
    return [int(x) for x in best_result]
