import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 115  # Safety margin
    
    N = 512
    # Precompute prime factors for each number 1..511
    factors = [set() for _ in range(N)]
    for i in range(2, N):
        if not factors[i]:
            for j in range(i, N, i):
                factors[j].add(i)
    
    # Weights
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
    
    # Greedy Initialization
    chosen = []
    used_primes = set()
    for a in range(N - 1, 0, -1):
        if not (factors[a] & used_primes):
            chosen.append(a)
            used_primes |= factors[a]
            
    best_score = sum(weights[a] for a in chosen)
    chosen_set = set(chosen)
    
    def is_coprime_to_except(a, except_set, current_used):
        # Check if a is coprime to all in current set except those in except_set
        # We need to check against current_used minus factors of except_set
        # But it's easier to just check against the list of chosen numbers not in except_set
        pass # This helper is complex, let's inline the logic in the loops for speed/clarity
    
    def get_score(chosen):
        return sum(weights[a] for a in chosen)

    # Local Search
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > DEADLINE:
            break
        
        # 1. Try 1-Adds
        # If there's any number coprime to all, add it.
        # This should be rare after greedy, but possible after swaps.
        for a in range(N - 1, 0, -1):
            if a in chosen_set:
                continue
            if not (factors[a] & used_primes):
                chosen.append(a)
                chosen_set.add(a)
                used_primes |= factors[a]
                best_score += weights[a]
                improved = True
                break # Restart search after add
        
        if improved:
            continue
            
        # 2. Try 1-Swaps
        # Remove one, add one
        for i in range(len(chosen)):
            a = chosen[i]
            # Calculate score change for each candidate b
            # b must be coprime to chosen \ {a}
            # used_primes_except_a = used_primes - factors[a]
            # We can't easily subtract sets efficiently in a loop, 
            # so we check: (factors[b] & used_primes) <= factors[a]
            
            # Iterate candidates in descending order of weight
            for b in range(N - 1, 0, -1):
                if b in chosen_set or b == a:
                    continue
                # Check coprimality with others
                # If b shares any prime with used_primes that is not in a, it's invalid.
                if (factors[b] & used_primes) - factors[a]:
                    continue
                
                # Valid swap
                delta = weights[b] - weights[a]
                if delta > 1e-12:
                    # Perform swap
                    chosen[i] = b
                    chosen_set.remove(a)
                    chosen_set.add(b)
                    used_primes = (used_primes - factors[a]) | factors[b]
                    best_score += delta
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # 3. Try 2-Swaps
        # Remove two, add two
        # To limit search, we only consider top candidates for addition
        # Top candidates are those with high weights (large a)
        # Let's consider top 100 candidates not in chosen
        candidates = [a for a in range(N - 1, 0, -1) if a not in chosen_set]
        # Limit to top 100 for speed
        top_candidates = candidates[:100]
        
        n = len(chosen)
        for i in range(n):
            if time.time() - start_time > DEADLINE:
                return chosen
            a1 = chosen[i]
            for j in range(i + 1, n):
                if time.time() - start_time > DEADLINE:
                    return chosen
                a2 = chosen[j]
                
                # Primes used by a1 and a2
                p12 = factors[a1] | factors[a2]
                
                # We want b1, b2 from top_candidates
                # such that (factors[b1] | factors[b2]) is disjoint from (used_primes - p12)
                # and b1, b2 are coprime.
                
                # Pre-filter top_candidates: keep those coprime to used_primes - p12
                # Actually, let's just iterate pairs from top_candidates
                # This is O(100^2 * n^2) = 10000 * 2500 = 25M. Might be tight.
                # Let's optimize:
                # For each b1 in top_candidates:
                #   Check if b1 coprime to used_primes - p12
                #   If yes, for each b2 in top_candidates (b2 != b1):
                #      Check if b2 coprime to used_primes - p12 and b2 coprime to b1
                #      Check score
                
                # To speed up, we can precompute which top_candidates are "valid" relative to the remaining set
                # But p12 changes.
                
                # Let's just do the nested loop with early breaks
                best_delta = 0
                best_pair = None
                
                # Sort top_candidates by weight desc (already is)
                for k1 in range(len(top_candidates)):
                    b1 = top_candidates[k1]
                    # Check b1 against used_primes excluding a1, a2
                    # Intersection with used_primes should be subset of p12
                    if (factors[b1] & used_primes) - p12:
                        continue
                    
                    for k2 in range(k1 + 1, len(top_candidates)):
                        b2 = top_candidates[k2]
                        if (factors[b2] & used_primes) - p12:
                            continue
                        # Check b1, b2 coprime
                        if factors[b1] & factors[b2]:
                            continue
                        
                        delta = weights[b1] + weights[b2] - weights[a1] - weights[a2]
                        if delta > best_delta:
                            best_delta = delta
                            best_pair = (b1, b2)
                
                if best_pair and best_delta > 1e-12:
                    b1, b2 = best_pair
                    # Perform swap
                    chosen[i] = b1
                    chosen[j] = b2
                    chosen_set.remove(a1)
                    chosen_set.remove(a2)
                    chosen_set.add(b1)
                    chosen_set.add(b2)
                    used_primes = (used_primes - p12) | factors[b1] | factors[b2]
                    best_score += best_delta
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # 4. Try 1-for-2 Swaps (Remove 1, Add 2)
        # This allows increasing set size if beneficial
        # Remove a, add b1, b2
        # b1, b2 must be coprime to used_primes - factors[a] and each other.
        
        # Use top candidates again
        top_candidates = [a for a in range(N - 1, 0, -1) if a not in chosen_set]
        top_candidates = top_candidates[:100]
        
        for i in range(len(chosen)):
            if time.time() - start_time > DEADLINE:
                return chosen
            a = chosen[i]
            pa = factors[a]
            
            best_delta = 0
            best_pair = None
            
            for k1 in range(len(top_candidates)):
                b1 = top_candidates[k1]
                if (factors[b1] & used_primes) - pa:
                    continue
                
                for k2 in range(k1 + 1, len(top_candidates)):
                    b2 = top_candidates[k2]
                    if (factors[b2] & used_primes) - pa:
                        continue
                    if factors[b1] & factors[b2]:
                        continue
                    
                    delta = weights[b1] + weights[b2] - weights[a]
                    if delta > best_delta:
                        best_delta = delta
                        best_pair = (b1, b2)
            
            if best_pair and best_delta > 1e-12:
                b1, b2 = best_pair
                # Perform swap
                chosen[i] = b1
                chosen.append(b2)
                chosen_set.remove(a)
                chosen_set.add(b1)
                chosen_set.add(b2)
                used_primes = (used_primes - pa) | factors[b1] | factors[b2]
                best_score += best_delta
                improved = True
                break
        
        if improved:
            continue
            
        # 5. Try 2-for-1 Swaps (Remove 2, Add 1)
        # This reduces set size, but might increase score if the single number is much larger
        # Remove a1, a2, add b
        # b must be coprime to used_primes - (factors[a1] | factors[a2])
        
        top_candidates = [a for a in range(N - 1, 0, -1) if a not in chosen_set]
        top_candidates = top_candidates[:100]
        
        n = len(chosen)
        for i in range(n):
            if time.time() - start_time > DEADLINE:
                return chosen
            a1 = chosen[i]
            for j in range(i + 1, n):
                if time.time() - start_time > DEADLINE:
                    return chosen
                a2 = chosen[j]
                p12 = factors[a1] | factors[a2]
                
                best_delta = 0
                best_b = None
                
                for b in top_candidates:
                    if (factors[b] & used_primes) - p12:
                        continue
                    
                    delta = weights[b] - weights[a1] - weights[a2]
                    if delta > best_delta:
                        best_delta = delta
                        best_b = b
                
                if best_b and best_delta > 1e-12:
                    # Perform swap
                    # Remove a1, a2, add best_b
                    # Update list
                    # Find indices
                    idx1 = chosen.index(a1)
                    idx2 = chosen.index(a2)
                    # Remove larger index first to keep indices valid
                    if idx1 > idx2:
                        chosen.pop(idx1)
                        chosen.pop(idx2)
                    else:
                        chosen.pop(idx2)
                        chosen.pop(idx1)
                    chosen.append(best_b)
                    
                    chosen_set.remove(a1)
                    chosen_set.remove(a2)
                    chosen_set.add(best_b)
                    used_primes = (used_primes - p12) | factors[best_b]
                    best_score += best_delta
                    improved = True
                    break
            if improved:
                break

    return chosen
