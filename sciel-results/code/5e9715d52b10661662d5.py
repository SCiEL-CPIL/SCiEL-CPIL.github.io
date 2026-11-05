"""Improved program: structural change to precomputed lookup tables + exhaustive k-search.
Combines the precomputation idea from inspiration 1 (score 1.6682) with a structural
shift from per-target exhaustive search to table-driven lookup. The key insight is that
for n<=128, the optimal k is small (1-4), and we can precompute all valid antichain
representations once, then look them up. This is a different modeling family: instead of
searching per target, we enumerate all valid (k, terms) tuples and index by sum.
"""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

_DEADLINE = 78.0  # safety margin before 90s kill
_START = time.time()

def _is_smooth(x):
    y = x
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

def _exponent_vector(x):
    """Return (a,b,c) for x = 2^a * 3^b * 5^c."""
    a = b = c = 0
    y = x
    while y % 2 == 0:
        y //= 2; a += 1
    while y % 3 == 0:
        y //= 3; b += 1
    while y % 5 == 0:
        y //= 5; c += 1
    return (a, b, c)

def _dominates(e1, e2):
    """Check if e1 dominates e2 (e1 >= e2 component-wise), meaning x1 divides x2."""
    return e1[0] >= e2[0] and e1[1] >= e2[1] and e1[2] >= e2[2]

def _is_antichain(terms):
    """Check if terms form a divisibility antichain using exponent vectors."""
    n = len(terms)
    exps = [_exponent_vector(t) for t in terms]
    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(exps[i], exps[j]) or _dominates(exps[j], exps[i]):
                return False
    return True

def construct():
    """
    Returns 128 rows for targets 1..128. Each row is a list of 1..30 distinct
    positive 2,3,5-smooth terms summing to the target, forming a divisibility antichain.
    Minimizes total number of summands.
    """
    # Precompute smooth numbers up to 128
    smooth = [x for x in range(1, 129) if _is_smooth(x)]
    smooth_set = set(smooth)
    smooth_sorted = sorted(smooth)  # already sorted
    # Precompute exponent vectors for all smooth numbers
    smooth_exps = {x: _exponent_vector(x) for x in smooth_sorted}
    
    # For each target n, find the minimum-k antichain representation
    # Strategy: precompute all valid antichains of size k=1,2,3,4 and index by sum
    # Then for each target, pick the representation with smallest k
    
    # k=1: single smooth number
    # k=2: pairs of smooth numbers that are antichain
    # k=3: triples
    # k=4: quadruples (unlikely needed for n<=128, but include for completeness)
    
    # Build lookup: sum -> list of (k, terms) sorted by k
    lookup = {}  # n -> list of (k, tuple(terms))
    
    # k=1
    for x in smooth:
        if x <= 128:
            if x not in lookup:
                lookup[x] = []
            lookup[x].append((1, (x,)))
    
    # k=2
    for i in range(len(smooth_sorted)):
        a = smooth_sorted[i]
        ea = smooth_exps[a]
        for j in range(i + 1, len(smooth_sorted)):
            b = smooth_sorted[j]
            s = a + b
            if s > 128:
                break
            # Check antichain using exponent vectors
            eb = smooth_exps[b]
            if _dominates(ea, eb) or _dominates(eb, ea):
                continue
            if s not in lookup:
                lookup[s] = []
            lookup[s].append((2, (a, b)))
    
    # k=3
    for i in range(len(smooth_sorted)):
        a = smooth_sorted[i]
        ea = smooth_exps[a]
        for j in range(i + 1, len(smooth_sorted)):
            b = smooth_sorted[j]
            s2 = a + b
            if s2 > 128:
                break
            eb = smooth_exps[b]
            if _dominates(ea, eb) or _dominates(eb, ea):
                continue
            for k in range(j + 1, len(smooth_sorted)):
                c = smooth_sorted[k]
                s3 = s2 + c
                if s3 > 128:
                    break
                ec = smooth_exps[c]
                if _dominates(ea, ec) or _dominates(ec, ea) or _dominates(eb, ec) or _dominates(ec, eb):
                    continue
                if s3 not in lookup:
                    lookup[s3] = []
                lookup[s3].append((3, (a, b, c)))
    
    # k=4 (only if needed)
    # First check if all targets 1-128 are already covered by k<=3
    all_covered = all(n in lookup for n in range(1, 129))
    if not all_covered:
        for i in range(len(smooth_sorted)):
            a = smooth_sorted[i]
            ea = smooth_exps[a]
            for j in range(i + 1, len(smooth_sorted)):
                b = smooth_sorted[j]
                s2 = a + b
                if s2 > 128:
                    break
                eb = smooth_exps[b]
                if _dominates(ea, eb) or _dominates(eb, ea):
                    continue
                for k in range(j + 1, len(smooth_sorted)):
                    c = smooth_sorted[k]
                    s3 = s2 + c
                    if s3 > 128:
                        break
                    ec = smooth_exps[c]
                    if _dominates(ea, ec) or _dominates(ec, ea) or _dominates(eb, ec) or _dominates(ec, eb):
                        continue
                    for l in range(k + 1, len(smooth_sorted)):
                        d = smooth_sorted[l]
                        s4 = s3 + d
                        if s4 > 128:
                            break
                        ed = smooth_exps[d]
                        if (_dominates(ea, ed) or _dominates(ed, ea) or
                            _dominates(eb, ed) or _dominates(ed, eb) or
                            _dominates(ec, ed) or _dominates(ed, ec)):
                            continue
                        if s4 not in lookup:
                            lookup[s4] = []
                        lookup[s4].append((4, (a, b, c, d)))
    
    # Now construct answers
    answers = []
    for n in range(1, 129):
        if time.time() - _START > _DEADLINE:
            # Fallback: use whatever we have
            if n in lookup:
                best = min(lookup[n], key=lambda x: x[0])
                answers.append(list(best[1]))
            else:
                # Last resort: try a simple greedy
                answers.append([n] if _is_smooth(n) else [1, n-1] if _is_smooth(n-1) else [1, 1, n-2])
            continue
        
        if n in lookup:
            # Pick the representation with smallest k
            best = min(lookup[n], key=lambda x: x[0])
            answers.append(list(best[1]))
        else:
            # Should not happen for n<=128, but fallback
            # Try a simple search
            found = None
            for k in range(1, 8):
                if time.time() - _START > _DEADLINE:
                    break
                for terms in combinations([x for x in smooth if x <= n], k):
                    if sum(terms) == n and _is_antichain(terms):
                        found = list(terms)
                        break
                if found:
                    break
            if found:
                answers.append(found)
            else:
                # Ultimate fallback
                answers.append([n] if _is_smooth(n) else [1, n-1])
    
    return answers
# EVOLVE-BLOCK-END
