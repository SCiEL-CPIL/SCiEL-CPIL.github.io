import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # Every element must share a factor with 255, so must be div by 3, 5, or 17
    # Start with all multiples of 3 (85 elements) - guaranteed valid
    base = set(range(3, 256, 3))
    
    # Try to add elements divisible by 5 or 17 (but not 3)
    # For an element x div by 5 (not 3) to coexist with all multiples of 3,
    # every multiple of 3 in the set must share a factor with x.
    # x is div by 5, not 3. For gcd(x, m) > 1 where m is div by 3:
    #   x and m must share a prime factor. Since m is div by 3 and x is not,
    #   they must share another prime factor.
    
    # Strategy: Build a valid set using greedy approach
    # Candidates: all numbers in [1,255] div by 3, 5, or 17, that include 255
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    # 255 must be included
    # Precompute prime factors for each candidate
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    pf = {c: prime_factors(c) for c in candidates}
    
    # Greedy: start with 255, add elements that are compatible with all current members
    # Compatibility: gcd(a,b) > 1, i.e., pf[a] & pf[b] is non-empty
    
    # Try multiple strategies and keep the best
    best_set = set(base)  # all multiples of 3
    
    # Strategy 1: Try to add non-multiples-of-3 by removing conflicting multiples of 3
    # For each candidate not div by 3, check if we can add it
    # This is complex; let's try a different approach
    
    # Strategy: Use the fact that if we restrict to numbers div by 3 OR (div by 5 AND div by 3) 
    # that's just div by 3. 
    # 
    # Better: consider subsets where all elements are div by at least one of {3,5,17}
    # and for any two, they share a prime.
    #
    # Key observation: If we take all numbers div by 3, that's 85.
    # If we take all numbers div by 5, that's 51.
    # If we take all numbers div by 17, that's 15.
    #
    # Can we combine? Numbers div by 3 AND 5 = div by 15 (17 numbers)
    # Numbers div by 3 AND 17 = div by 51 (5 numbers)  
    # Numbers div by 5 AND 17 = div by 85 (3 numbers)
    #
    # A valid set: all numbers div by 3, plus all numbers div by 5 that are also div by 3 or 17
    # i.e., div by 15 or 85. But div by 15 is already in div by 3. Div by 85: 85, 170, 255.
    # 255 is already in. 85 and 170: are they compatible with all multiples of 3?
    # 85 = 5*17. gcd(85, 3) = 1. No!
    
    # So the simple union doesn't work. Let me try a search approach.
    
    # Approach: Start with multiples of 3. Try adding each non-multiple-of-3 candidate.
    # If adding x breaks compatibility, we need to remove conflicting elements.
    # Net gain = 1 - (number of elements we must remove)
    
    # For a candidate x not div by 3:
    #   x is div by 5 or 17 (or both)
    #   Conflicting multiples of 3: those m where gcd(x,m)=1
    #   m is div by 3. gcd(x,m)=1 means m shares no prime factor with x.
    #   If x div by 5, not 3, not 17: m must not be div by 5. So m div by 3 but not 5.
    #   If x div by 17, not 3, not 5: m must not be div by 17. So m div by 3 but not 17.
    #   If x div by 5 and 17, not 3: m must not be div by 5 AND not div by 17.
    
    current_set = set(base)
    remaining = [c for c in candidates if c not in current_set]
    
    # Sort remaining by potential benefit
    def conflict_count(x, s):
        """Count elements in s that conflict with x"""
        cnt = 0
        xpf = pf[x]
        for m in s:
            if not (xpf & pf[m]):
                cnt += 1
        return cnt
    
    # Greedy: repeatedly add the element with best (1 - conflicts) if positive
    improved = True
    while improved:
        improved = False
        best_gain = 0
        best_elem = None
        for x in remaining:
            if time.time() > deadline:
                break
            conf = conflict_count(x, current_set)
            gain = 1 - conf
            if gain > best_gain:
                best_gain = gain
                best_elem = x
        if best_elem is not None and best_gain > 0:
            # Add best_elem, remove conflicting elements
            xpf = pf[best_elem]
            to_remove = [m for m in current_set if not (xpf & pf[m])]
            for m in to_remove:
                current_set.discard(m)
                if m in remaining:
                    pass  # already not in current_set
            current_set.add(best_elem)
            remaining.remove(best_elem)
            improved = True
        if time.time() > deadline:
            break
    
    # Also try starting from different bases
    # Try: all multiples of 5 (51 elements)
    base5 = set(range(5, 256, 5))
    if len(base5) > len(current_set):
        current_set = base5
        remaining = [c for c in candidates if c not in current_set]
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_elem = None
            for x in remaining:
                if time.time() > deadline:
                    break
                conf = conflict_count(x, current_set)
                gain = 1 - conf
                if gain > best_gain:
                    best_gain = gain
                    best_elem = x
            if best_elem is not None and best_gain > 0:
                xpf = pf[best_elem]
                to_remove = [m for m in current_set if not (xpf & pf[m])]
                for m in to_remove:
                    current_set.discard(m)
                current_set.add(best_elem)
                remaining.remove(best_elem)
                improved = True
            if time.time() > deadline:
                break
    
    # Try: all multiples of 17 (15 elements) - unlikely to beat others
    base17 = set(range(17, 256, 17))
    if len(base17) > len(current_set):
        current_set = base17
        remaining = [c for c in candidates if c not in current_set]
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_elem = None
            for x in remaining:
                if time.time() > deadline:
                    break
                conf = conflict_count(x, current_set)
                gain = 1 - conf
                if gain > best_gain:
                    best_gain = gain
                    best_elem = x
            if best_elem is not None and best_gain > 0:
                xpf = pf[best_elem]
                to_remove = [m for m in current_set if not (xpf & pf[m])]
                for m in to_remove:
                    current_set.discard(m)
                current_set.add(best_elem)
                remaining.remove(best_elem)
                improved = True
            if time.time() > deadline:
                break
    
    # Verify 255 is in the set
    if 255 not in current_set:
        current_set.add(255)
    
    # Final verification
    result = list(current_set)
    result.sort()
    
    # Verify all pairs have gcd > 1
    valid = True
    for i in range(len(result)):
        if time.time() > deadline:
            break
        for j in range(i+1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to safe answer
        return list(range(3, 256, 3))
    
    return result
