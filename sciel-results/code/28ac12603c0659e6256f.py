# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _generate_smooth(max_val):
    """Generate all 2^a * 3^b * 5^c numbers up to max_val, sorted."""
    smooth = set()
    a = 1
    while a <= max_val:
        b = a
        while b <= max_val:
            c = b
            while c <= max_val:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(smooth)

def _exponent_vector(n):
    """Return (a, b, c) for n = 2^a * 3^b * 5^c."""
    a = 0
    while n % 2 == 0:
        a += 1
        n //= 2
    b = 0
    while n % 3 == 0:
        b += 1
        n //= 3
    c = 0
    while n % 5 == 0:
        c += 1
        n //= 5
    return (a, b, c)

def _dominates(e1, e2):
    """Check if e1 dominates e2 (component-wise >=)."""
    return e1[0] >= e2[0] and e1[1] >= e2[1] and e1[2] >= e2[2]

def _is_antichain(terms):
    """Check if no term divides another."""
    n = len(terms)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = terms[i], terms[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                return False
    return True

def _find_antichain_for_target(target, smooth_numbers, deadline):
    """Find the minimum number of smooth terms summing to target with antichain property."""
    # Precompute exponent vectors for smooth numbers <= target
    candidates = [(x, _exponent_vector(x)) for x in smooth_numbers if x <= target]
    
    # Try k = 1, 2, 3, ... up to some limit
    max_k = 8  # We can always represent with many terms if needed, but start small
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        # Try combinations of size k
        # Sort candidates by value for potential pruning
        for combo in combinations(candidates, k):
            terms = [c[0] for c in combo]
            if sum(terms) != target:
                continue
            # Check antichain property
            if not _is_antichain(terms):
                continue
            return terms
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for 90s limit
    
    # Generate smooth numbers up to 128
    smooth_numbers = _generate_smooth(128)
    
    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            # If we're running out of time, use a fallback
            # This shouldn't happen for small n, but just in case
            answers.append(None)
            continue
        
        result = _find_antichain_for_target(n, smooth_numbers, deadline)
        
        if result is None:
            # Fallback: try larger k values
            # For very small targets, we might need to be more creative
            # But for 1..128, k=1 should work for smooth numbers themselves
            # Let's try a more exhaustive search with larger k
            smooth_candidates = [x for x in smooth_numbers if x <= n]
            found = False
            for k in range(1, 15):
                if time.time() > deadline:
                    break
                for combo in combinations(smooth_candidates, k):
                    if sum(combo) == n and _is_antichain(combo):
                        result = list(combo)
                        found = True
                        break
                if found:
                    break
            if result is None:
                # Absolute fallback - shouldn't happen
                result = [n] if n in smooth_numbers else [1] * n
        
        answers.append(result)
    
    return answers
# EVOLVE-BLOCK-END
