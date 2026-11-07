"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _violates(a, b, c):
    """Check if a divides b*c (a, b, c must be distinct values)."""
    return (b * c) % a == 0

def _count_violations(S, x):
    """Count how many pairs (b,c) in S (distinct, not x) cause x to divide b*c."""
    count = 0
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            b, c = S[i], S[j]
            if b == x or c == x:
                continue
            if (b * c) % x == 0:
                count += 1
    return count

def _can_add(S, x):
    """Check if adding x to S maintains validity."""
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            a, b, c = x, S[i], S[j]
            if a == b or a == c or b == c:
                continue
            if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                return False
    return True

def _is_valid(S):
    """Full validation of set S."""
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            for k in range(j+1, len(S)):
                a, b, c = S[i], S[j], S[k]
                if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Start with upper half [129, 256] as initial set
    S = list(range(129, 257))
    
    # Greedy removal: repeatedly remove the element with the most violations
    changed = True
    while changed:
        if time.time() > deadline - 10:
            break
        changed = False
        worst_idx = -1
        worst_count = 0
        for i in range(len(S)):
            x = S[i]
            count = 0
            for j in range(len(S)):
                if i == j:
                    continue
                for k in range(len(S)):
                    if i == k or j == k:
                        continue
                    if (S[j] * S[k]) % x == 0:
                        count += 1
            if count > worst_count:
                worst_count = count
                worst_idx = i
        if worst_idx >= 0 and worst_count > 0:
            S.pop(worst_idx)
            changed = True
    
    # Try to add back elements from [2, 128] in order of least conflicts
    candidates = list(range(2, 129))
    # Sort by how many violations they'd cause with current S
    def conflict_score(x):
        count = 0
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                b, c = S[i], S[j]
                if (b * c) % x == 0 or (x * c) % b == 0 or (x * b) % c == 0:
                    count += 1
        return count
    
    candidates.sort(key=conflict_score)
    
    for x in candidates:
        if time.time() > deadline - 5:
            break
        if x in S:
            continue
        if _can_add(S, x):
            S.append(x)
    
    # Local search: try swapping elements to improve
    for _ in range(20):  # limited iterations
        if time.time() > deadline - 3:
            break
        improved = False
        for i in range(len(S)):
            if time.time() > deadline - 3:
                break
            for x in range(2, 257):
                if x in S:
                    continue
                # Try replacing S[i] with x
                old = S[i]
                S[i] = x
                if _is_valid(S):
                    improved = True
                    break
                S[i] = old
            if improved:
                break
        if not improved:
            break
    
    # Final validation
    if not _is_valid(S):
        # Fallback to primes
        n = 256
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        S = [i for i in range(2, n + 1) if is_prime[i]]
    
    return S

# EVOLVE-BLOCK-END
