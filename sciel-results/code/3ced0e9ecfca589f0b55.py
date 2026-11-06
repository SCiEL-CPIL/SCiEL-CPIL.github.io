"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 192
    MAX_K = 3
    
    def fast_greedy(order, deadline):
        """
        Greedy construction with optimized validity checks using auxiliary structures.
        K: Set of all sums of 3 distinct elements in S.
        R: Dict mapping element a to set of residues (u+v) % a for u,v in S\{a}, u!=v.
        """
        S = []
        K = set()  # Sums of 3 distinct elements
        R = {}     # Residues of pair sums mod a
        
        for x in order:
            if time.time() > deadline:
                break
                
            valid = True
            
            # Check 1: x does not divide any sum of 3 distinct elements in S
            # If x divides sum(s), then sum(s) % x == 0
            for s in K:
                if s % x == 0:
                    valid = False
                    break
            
            # Check 2: For each a in S, a does not divide (x + u + v) for any distinct u,v in S\{a}
            # (x + u + v) % a == 0  <=>  (u + v) % a == (-x) % a
            if valid:
                for a in S:
                    target = (-x) % a
                    # R[a] contains residues (u+v)%a for u,v in S\{a}
                    if target in R[a]:
                        valid = False
                        break
            
            if valid:
                old_S = S[:]
                S.append(x)
                
                # Update K: Add new sums involving x
                # New sums are x + u + v where u, v are distinct elements from old_S
                n_old = len(old_S)
                for i in range(n_old):
                    u = old_S[i]
                    for j in range(i + 1, n_old):
                        v = old_S[j]
                        K.add(x + u + v)
                
                # Update R for existing elements in old_S
                for a in old_S:
                    # New pairs involving x: (x, u) where u in old_S, u != a
                    # Residue: (x + u) % a
                    for u in old_S:
                        if u != a:
                            R[a].add((x + u) % a)
                
                # Initialize R for new element x
                # Pairs in S\{x} are pairs in old_S
                R[x] = set()
                for i in range(n_old):
                    u = old_S[i]
                    for j in range(i + 1, n_old):
                        v = old_S[j]
                        R[x].add((u + v) % x)
                
        return S

    def can_add(x, S):
        """Check if x can be added to set S without violating constraints."""
        if len(S) < 3:
            return True
            
        # Check if x divides any sum of 3 in S
        for comb in combinations(S, 3):
            if sum(comb) % x == 0:
                return False
        
        # Check if any a in S divides x + u + v for u,v in S\{a}
        n = len(S)
        for i, a in enumerate(S):
            target = (-x) % a
            found = False
            for u_idx in range(n):
                if u_idx == i:
                    continue
                u = S[u_idx]
                for v_idx in range(u_idx + 1, n):
                    if v_idx == i:
                        continue
                    v = S[v_idx]
                    if (u + v) % a == target:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
        
        return True

    def try_local_improve(S, time_limit):
        """
        Attempts to improve set S by adding elements or swapping.
        Returns improved set if found, else S.
        """
        S = list(S)
        S_set = set(S)
        candidates = [x for x in range(1, 193) if x not in S_set]
        
        # Shuffle candidates for randomization
        random.shuffle(candidates)
        
        # Try direct additions first
        for x in candidates:
            if time.time() > time_limit:
                break
            if can_add(x, S):
                S.append(x)
                S_set.add(x)
                return try_local_improve(S, time_limit)
        
        # Try swaps: remove a from S, add x
        for a in S[:]:
            if time.time() > time_limit:
                break
            S_temp = [s for s in S if s != a]
            S_temp_set = set(S_temp)
            
            for x in candidates:
                if time.time() > time_limit:
                    break
                if x in S_temp_set:
                    continue
                if can_add(x, S_temp):
                    S_temp.append(x)
                    return S_temp
        
        return S

    def generate_order(strategy, rng):
        """Generate a candidate ordering based on strategy."""
        base = list(range(1, 193))
        
        if strategy == 'desc':
            return list(range(192, 0, -1))
        elif strategy == 'asc':
            return list(range(1, 193))
        elif strategy == 'random':
            rng.shuffle(base)
            return base
        elif strategy == 'prime_weighted':
            # Weight primes higher (they're less likely to divide sums)
            weights = []
            for x in range(1, 193):
                # Simple primality test
                is_prime = True
                if x < 2:
                    is_prime = False
                else:
                    for d in range(2, int(x**0.5) + 1):
                        if x % d == 0:
                            is_prime = False
                            break
                weights.append(10 if is_prime else 1)
            
            # Sort by weight descending, with random tie-breaking
            indexed = list(enumerate(base))
            indexed.sort(key=lambda idx: (-weights[idx[0]], rng.random()))
            return [idx[1] for idx in indexed]
        elif strategy == 'large_prime':
            # Prioritize large primes
            primes = []
            for x in range(192, 0, -1):
                is_prime = True
                if x < 2:
                    is_prime = False
                else:
                    for d in range(2, int(x**0.5) + 1):
                        if x % d == 0:
                            is_prime = False
                            break
                if is_prime:
                    primes.append(x)
            
            non_primes = [x for x in range(192, 0, -1) if x not in primes]
            rng.shuffle(non_primes)
            return primes + non_primes
        else:
            rng.shuffle(base)
            return base

    best_set = []
    best_size = 0
    
    rng = random.Random(42)
    
    # Strategy 1: Deterministic orders
    for order in [list(range(192, 0, -1)), list(range(1, 193))]:
        if time.time() > deadline:
            break
        S = fast_greedy(order, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    # Strategy 2: Multiple random restarts with different orderings
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Choose strategy based on iteration
        if i % 4 == 0:
            order = generate_order('prime_weighted', rng)
        elif i % 4 == 1:
            order = generate_order('large_prime', rng)
        elif i % 4 == 2:
            order = generate_order('random', rng)
        else:
            order = generate_order('desc', rng)
        
        S = fast_greedy(order, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
            # If we found a good set, try to improve it
            if best_size < 48:
                local_deadline = min(deadline, time.time() + 1.0)
                improved_S = try_local_improve(best_set, local_deadline)
                if len(improved_S) > best_size:
                    best_size = len(improved_S)
                    best_set = improved_S[:]
    
    # Final local improvement on the best set found
    if best_size < 48 and time.time() < deadline:
        local_deadline = deadline
        improved_S = try_local_improve(best_set, local_deadline)
        if len(improved_S) > best_size:
            best_size = len(improved_S)
            best_set = improved_S[:]
    
    return best_set
# EVOLVE-BLOCK-END
