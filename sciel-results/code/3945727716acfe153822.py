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

def _count_conflicts(S):
    """Count for each element in S how many distinct triples (a,b,c) with a|bc, b|ac, or c|ab it participates in."""
    n = len(S)
    if n < 3:
        return {x: 0 for x in S}
    counts = {x: 0 for x in S}
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            for k in range(j + 1, n):
                c = S[k]
                if (b * c) % a == 0:
                    counts[a] += 1
                if (a * c) % b == 0:
                    counts[b] += 1
                if (a * b) % c == 0:
                    counts[c] += 1
    return counts

def _has_violations(S):
    """Check if S has any violation. Returns True if at least one exists."""
    n = len(S)
    if n < 3:
        return False
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            for k in range(j + 1, n):
                c = S[k]
                if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                    return True
    return False

def _can_add(S, x):
    """Check if adding x to S preserves the no-violation property."""
    n = len(S)
    if n < 2:
        return True
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            # Triples: (x, a, b), (a, x, b), (a, b, x)
            # x | a*b
            if (a * b) % x == 0:
                return False
            # a | x*b
            if (x * b) % a == 0:
                return False
            # b | a*x
            if (a * x) % b == 0:
                return False
    return True

def construct():
    t0 = time.time()
    n = P['n']
    
    # Strategy: Start with upper half [129, 256] and greedily remove most-conflicted elements.
    S = list(range(129, n + 1))
    
    # Greedy removal phase
    while time.time() - t0 < _DEADLINE:
        counts = _count_conflicts(S)
        n_elements = len(S)
        if n_elements < 3:
            break
        
        violations_found = any(v > 0 for v in counts.values())
        if not violations_found:
            break
        
        # Find the element with the maximum conflict count
        max_count = max(counts.values())
        candidates = [x for x in S if counts[x] == max_count]
        
        # Remove one candidate.
        # Heuristic: Remove the smallest one to eliminate frequent divisors.
        remove_elem = min(candidates)
        S.remove(remove_elem)
    
    # Local search: try to re-add elements that were removed
    # Elements that were removed are those in [129, 256] not in S
    removed = [x for x in range(129, n + 1) if x not in S]
    
    # Try to add back removed elements in order of least conflict potential
    # Sort by how many pairs in S they conflict with
    for x in sorted(removed, key=lambda x: sum(1 for i in range(len(S)) for j in range(i+1, len(S)) if (S[i]*S[j])%x==0 or (x*S[j])%S[i]==0 or (S[i]*x)%S[j]==0)):
        if time.time() - t0 > _DEADLINE:
            break
        if _can_add(S, x):
            S.append(x)
    
    # Try to add elements from lower range too
    lower_candidates = [x for x in range(2, 129) if x not in S]
    # Sort by how many pairs in S they conflict with (ascending)
    def conflict_potential(x):
        cnt = 0
        for i in range(len(S)):
            a = S[i]
            for j in range(i+1, len(S)):
                b = S[j]
                if (a * b) % x == 0 or (x * b) % a == 0 or (a * x) % b == 0:
                    cnt += 1
        return cnt
    
    lower_candidates.sort(key=conflict_potential)
    
    for x in lower_candidates:
        if time.time() - t0 > _DEADLINE:
            break
        if _can_add(S, x):
            S.append(x)
    
    # Fallback: If the result is smaller than primes, use primes.
    primes = [a for a in range(2, n + 1) if _is_prime(a)]
    if len(S) < len(primes):
        S = primes
    
    result = [int(x) for x in S]
    return result
# EVOLVE-BLOCK-END
