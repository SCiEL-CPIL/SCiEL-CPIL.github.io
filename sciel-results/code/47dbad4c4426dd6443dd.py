import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 511
    C = 512
    
    # 1. Precompute prime factors for each number
    prime_factors = [set() for _ in range(N + 1)]
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                prime_factors[j].add(i)
    for i in range(int(N**0.5) + 1, N + 1):
        if is_prime[i]:
            prime_factors[i].add(i)
            
    # Ensure 1 has no prime factors (it's coprime to everything)
    prime_factors[1] = set()

    # Weights
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (C - a)
        
    # Sort candidates by weight descending (which is a descending)
    candidates = list(range(N, 0, -1))
    
    # 2. Greedy Initialization
    S = set()
    used_primes = set()
    
    for a in candidates:
        if time.time() > deadline:
            break
        if prime_factors[a].isdisjoint(used_primes):
            S.add(a)
            used_primes.update(prime_factors[a])
            
    current_score = sum(w[a] for a in S)
    best_S = S.copy()
    best_score = current_score
    
    # Helper to update used_primes when removing/adding
    def remove_from_S(a):
        S.discard(a)
        for p in prime_factors[a]:
            # Only remove if no other element in S has p
            # Since S is pairwise coprime, if we remove a, p is definitely free unless another element has it?
            # Wait, if S is pairwise coprime, each prime in used_primes belongs to exactly one element in S.
            # So we can just subtract the factors of a.
            pass 
            
    # Actually, maintaining used_primes efficiently:
    # Since elements in S are pairwise coprime, the set of used primes is the union of prime_factors[a] for a in S.
    # And these sets are disjoint.
    # So used_primes = set().union(*[prime_factors[a] for a in S])
    # When removing a, we can just do used_primes -= prime_factors[a]
    
    # Let's implement the local search loop
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        S_list = list(S)
        
        # Try 1-Swap: Remove x, Add best y
        for x in S_list:
            if time.time() > deadline - 5.0:
                break
            
            # Temporarily remove x
            S.discard(x)
            used_primes -= prime_factors[x]
            
            best_y = None
            best_w_y = -1
            for y in candidates:
                if y in S:
                    continue
                if prime_factors[y].isdisjoint(used_primes):
                    if w[y] > best_w_y:
                        best_w_y = w[y]
                        best_y = y
                        
            if best_y is not None and best_w_y > w[x] + 1e-9:
                S.add(best_y)
                used_primes.update(prime_factors[best_y])
                current_score += best_w_y - w[x]
                if current_score > best_score:
                    best_score = current_score
                    best_S = S.copy()
                improved = True
                break # Restart loop as S changed
            else:
                # Revert
                S.add(x)
                used_primes.update(prime_factors[x])
                
        if improved:
            continue
            
        # Try 1-for-2 Swap: Remove x, Add best y, z
        for x in S_list:
            if time.time() > deadline - 5.0:
                break
            
            # Temporarily remove x
            S.discard(x)
            used_primes -= prime_factors[x]
            
            best_pair = None
            best_pair_w = -1
            
            # We want max w[y] + w[z] such that y, z coprime to S\{x} and y, z coprime to each other.
            # Iterate y
            for y in candidates:
                if y in S:
                    continue
                if not prime_factors[y].isdisjoint(used_primes):
                    continue
                
                # If even the max possible z doesn't beat x, skip
                # Max w[z] is w[511] = 1.0. If w[y] + 1.0 <= w[x], skip.
                if w[y] + w[candidates[0]] <= w[x]:
                    break # candidates is sorted desc, so this is the max possible sum for this y? 
                    # Wait, w is decreasing. candidates[0] is 511.
                    # If w[y] + w[511] <= w[x], then no z can help this y.
                    # Since y is iterated in desc weight, if this fails for y, it might fail for smaller y?
                    # Not necessarily, w[y] is smaller for smaller y. So yes, we can break.
                    
                # Find best z
                # z must be coprime to S\{x} AND coprime to y.
                # So z must be coprime to (S\{x}) U {y}.
                # Used primes for this check: used_primes U prime_factors[y]
                
                temp_used = used_primes | prime_factors[y]
                
                best_z = None
                best_w_z = -1
                for z in candidates:
                    if z == y or z in S:
                        continue
                    if not prime_factors[z].isdisjoint(temp_used):
                        continue
                    if w[z] > best_w_z:
                        best_w_z = w[z]
                        best_z = z
                        
                if best_z is not None:
                    pair_w = w[y] + best_w_z
                    if pair_w > best_pair_w:
                        best_pair_w = pair_w
                        best_pair = (y, best_z)
                        
            if best_pair is not None and best_pair_w > w[x] + 1e-9:
                y, z = best_pair
                S.add(y)
                S.add(z)
                used_primes.update(prime_factors[y])
                used_primes.update(prime_factors[z])
                current_score += best_pair_w - w[x]
                if current_score > best_score:
                    best_score = current_score
                    best_S = S.copy()
                improved = True
                break
            else:
                # Revert
                S.add(x)
                used_primes.update(prime_factors[x])
                
    return list(best_S)
