"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 256}
_DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _is_prime(a):
    if a < 2:
        return False
    if a < 4:
        return True
    if a % 2 == 0 or a % 3 == 0:
        return False
    i = 5
    while i * i <= a:
        if a % i == 0 or a % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    t0 = time.time()
    n = P['n']
    
    # Strategy: Start with upper half [129, 256] and greedily remove most-conflicted elements.
    # This exploits the antichain property of the upper half (no element divides another),
    # meaning violations only arise from a | bc where a, b, c are distinct.
    
    S = list(range(129, n + 1))
    
    while time.time() - t0 < _DEADLINE:
        counts = {x: 0 for x in S}
        n_elements = len(S)
        if n_elements < 3:
            break
            
        violations_found = False
        
        # Check all distinct triples (i, j, k)
        for i in range(n_elements):
            a = S[i]
            for j in range(i + 1, n_elements):
                b = S[j]
                ab = a * b
                for k in range(j + 1, n_elements):
                    c = S[k]
                    
                    # Check a | b*c
                    if (b * c) % a == 0:
                        counts[a] += 1
                        violations_found = True
                    # Check b | a*c
                    if (a * c) % b == 0:
                        counts[b] += 1
                        violations_found = True
                    # Check c | a*b
                    if ab % c == 0:
                        counts[c] += 1
                        violations_found = True
        
        if not violations_found:
            break
            
        # Find the element with the maximum conflict count
        max_count = max(counts.values())
        candidates = [x for x in S if counts[x] == max_count]
        
        # Remove one candidate. 
        # Heuristic: Remove the smallest one to eliminate frequent divisors.
        remove_elem = min(candidates)
        S.remove(remove_elem)
        
    # Fallback: If the greedy loop didn't finish or resulted in a small set,
    # we can try to verify if the prime set (size 53) is better.
    # But the knowledge says upper half greedy gets 55.
    # Just in case, if S is smaller than 53, use primes.
    primes = [a for a in range(2, n + 1) if _is_prime(a)]
    if len(S) < len(primes):
        S = primes
        
    result = [int(x) for x in S]
    return result
# EVOLVE-BLOCK-END
