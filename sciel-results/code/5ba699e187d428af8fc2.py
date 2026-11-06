import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _elapsed():
    return time.time() - _START_TIME

def _time_exhausted():
    return _elapsed() > _TIME_LIMIT

def _precompute_factors(n):
    factors = [set() for _ in range(n + 1)]
    for i in range(2, n + 1):
        if not factors[i]:
            for j in range(i, n + 1, i):
                factors[j].add(i)
    return factors

def _score(chosen):
    return sum(1.0 / (512.0 - a) for a in chosen)

def construct():
    N = 511
    factors = _precompute_factors(N)
    
    # Greedy Initialization
    chosen = []
    chosen_set = set()
    prime_counts = {} # prime -> count of elements in chosen using this prime
    
    for a in range(N, 0, -1):
        if not factors[a]:
            # 1 is coprime to everything, but 1/(512-1) is small.
            # Greedy from top will pick it only if nothing else fits?
            # Actually 1 has no prime factors. It is always coprime.
            # But it has low value.
            pass
        if factors[a]:
            valid = True
            for p in factors[a]:
                if prime_counts.get(p, 0) > 0:
                    valid = False
                    break
            if not valid:
                continue
        
        # Add a
        chosen.append(a)
        chosen_set.add(a)
        for p in factors[a]:
            prime_counts[p] = prime_counts.get(p, 0) + 1
            
    best_score = _score(chosen)
    
    # Local Search
    while not _time_exhausted():
        improved = False
        
        # 1. Try adding elements
        # Check from largest to smallest
        for a in range(N, 0, -1):
            if _time_exhausted():
                break
            if a in chosen_set:
                continue
            
            valid = True
            for p in factors[a]:
                if prime_counts.get(p, 0) > 0:
                    valid = False
                    break
            
            if valid:
                # Add
                chosen.append(a)
                chosen_set.add(a)
                for p in factors[a]:
                    prime_counts[p] = prime_counts.get(p, 0) + 1
                best_score += 1.0 / (512.0 - a)
                improved = True
                break # Restart search with new state
        
        if _time_exhausted():
            break

        # 2. Try swaps
        # For each a in chosen, try replacing with b > a
        for i in range(len(chosen)):
            if _time_exhausted():
                break
            a = chosen[i]
            a_factors = factors[a]
            
            # Calculate which primes become available (count drops to 0)
            # We don't need to explicitly calculate, just check candidate b
            # against the "remaining" used primes.
            
            # To check if b is valid:
            # For each p in factors[b]:
            #   if p in a_factors:
            #       # p is used by a. If we remove a, count becomes prime_counts[p] - 1.
            #       # b can use p if (prime_counts[p] - 1) == 0? 
            #       # No, b can use p if NO OTHER element uses p.
            #       # So we need prime_counts[p] - 1 == 0.
            #       if prime_counts[p] - 1 > 0:
            #           valid = False
            #   else:
            #       # p is not used by a. So it is used by some other element.
            #       # b cannot use p.
            #       if prime_counts.get(p, 0) > 0:
            #           valid = False
            
            # We want b > a to improve score (since 1/(512-b) is increasing with b).
            # Iterate b from N down to a+1.
            for b in range(N, a, -1):
                if b in chosen_set:
                    continue
                
                valid = True
                for p in factors[b]:
                    count = prime_counts.get(p, 0)
                    if p in a_factors:
                        # This prime is shared with a.
                        # After removing a, remaining count is count - 1.
                        if count - 1 > 0:
                            valid = False
                            break
                    else:
                        # This prime is not in a.
                        # So it is used by other elements.
                        if count > 0:
                            valid = False
                            break
                
                if valid:
                    # Perform swap
                    # Remove a
                    chosen.pop(i)
                    chosen_set.remove(a)
                    for p in a_factors:
                        prime_counts[p] -= 1
                        if prime_counts[p] == 0:
                            del prime_counts[p]
                    
                    # Add b
                    chosen.append(b)
                    chosen_set.add(b)
                    for p in factors[b]:
                        if p in prime_counts:
                            prime_counts[p] += 1
                        else:
                            prime_counts[p] = 1
                            
                    # Update score
                    best_score -= 1.0 / (512.0 - a)
                    best_score += 1.0 / (512.0 - b)
                    improved = True
                    break # Restart search
                    
            if improved:
                break # Restart search
                
        if not improved:
            break
            
    return chosen
