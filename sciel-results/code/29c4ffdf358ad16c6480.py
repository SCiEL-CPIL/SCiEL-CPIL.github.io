import time

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_N = len(_DIVISORS)
_MAXV = 1000000
_DEADLINE = time.time() + 80  # Safety margin


def _solve_window(L, U):
    """
    Try to assign distinct multiples of each divisor within [L, U].
    Returns a list of 10 integers if successful, else None.
    """
    n = U - L + 1
    if n < _N:
        return None
        
    # Generate candidates for each divisor
    cands = []
    for d in _DIVISORS:
        # Smallest multiple >= L
        k = (L + d - 1) // d
        c = []
        val = k * d
        while val <= U:
            c.append(val)
            val += d
        cands.append(c)
        
    # Check if any divisor has no candidates
    if any(len(c) == 0 for c in cands):
        return None

    # Order divisors by constraint (fewest candidates first)
    order = sorted(range(_N), key=lambda i: len(cands[i]))
    
    used = [False] * n
    reps = [0] * _N
    
    def bt(pos):
        if pos == _N:
            return True
        i = order[pos]
        for m in cands[i]:
            idx = m - L
            if not used[idx]:
                used[idx] = True
                reps[i] = m
                if bt(pos + 1):
                    return True
                used[idx] = False
        return False

    if bt(0):
        return reps[:]
    return None


def construct():
    start_time = time.time()
    
    # We search for the minimum span S
    # The span must be at least N-1 = 9 because integers are distinct
    # We iterate S from 9 upwards
    
    # To be efficient, we generate candidate starting points L for each S.
    # The minimum element of the solution set must be a multiple of one of the divisors.
    # Let's iterate potential minimums.
    
    # We'll keep a list of candidate Ls to check for each S.
    # Actually, for each S, we can just iterate L from 1 to some bound, but that's slow.
    # Better: For each S, iterate multiples of divisors as potential L_min.
    
    # How high should L_min go?
    # If S is small, the numbers are close together.
    # The "density" of multiples is high.
    # Let's try S from 9 to 50.
    # For each S, we check windows starting at multiples of divisors up to a limit.
    
    # Limit for L_min: 
    # If we haven't found a solution for S <= 50, the span is likely larger.
    # But the optimal span is likely small.
    # Let's set a limit for L_min search to 10000.
    
    L_LIMIT = 10000
    
    # Pre-compute multiples of each divisor up to L_LIMIT + max_span
    # We don't know max_span yet, but we can generate on the fly or use a generous bound.
    # Let's just generate candidates for each S.
    
    for S in range(9, 100):
        if time.time() > _DEADLINE:
            break
            
        # Generate candidate Ls
        # A valid window [L, L+S] must contain a multiple of each divisor.
        # The minimum value in the window, L, doesn't have to be a multiple, 
        # but the *minimum of the selected set* does.
        # Let m_min be the min of the selected set. m_min is a multiple of some d_i.
        # And m_min >= L. Also m_min <= L+S.
        # So L <= m_min <= L+S.
        # This implies L >= m_min - S.
        # So for a specific m_min, L can be in [m_min - S, m_min].
        # To cover all possibilities, we can just check windows [m, m+S] where m is a multiple of some divisor?
        # No, if the min is m, the window is [m, m+S].
        # But what if the min is not the start of the window?
        # The window is defined by L and U. The set is a subset.
        # If we find a set in [L, U] with span S', we are happy.
        # The "span" is max(set) - min(set).
        # If we fix the span S, we are looking for a set with max - min = S.
        # Let min(set) = m. Then max(set) = m + S.
        # So the set is contained in [m, m+S].
        # And m must be a multiple of its divisor.
        # So we just need to check windows [m, m+S] where m is a multiple of one of the divisors.
        
        # So, for this S, we iterate over all multiples m of each divisor d_i, 
        # such that m <= L_LIMIT (and m+S <= _MAXV).
        # For each such m, we check if a valid set exists in [m, m+S].
        
        # To avoid duplicate checks, we can use a set of Ls.
        # But m is the minimum, so L=m.
        
        # Let's collect unique m's.
        candidates = set()
        for d in _DIVISORS:
            # Multiples of d up to L_LIMIT
            # k*d <= L_LIMIT
            max_k = L_LIMIT // d
            for k in range(1, max_k + 1):
                m = k * d
                if m + S <= _MAXV:
                    candidates.add(m)
        
        # Sort candidates to be deterministic (though not strictly necessary)
        for m in sorted(candidates):
            if time.time() > _DEADLINE:
                break
            reps = _solve_window(m, m + S)
            if reps is not None:
                # Found a solution with span S
                # Validate
                if all(1 <= x <= _MAXV for x in reps) and len(set(reps)) == _N:
                    if all(reps[i] % _DIVISORS[i] == 0 for i in range(_N)):
                        # Return as list of ints
                        return [int(x) for x in reps]
        
        # If no solution found for this S with m <= L_LIMIT, 
        # it's possible the optimal m is larger, but for small S, 
        # it's very likely to be small. 
        # If S is small, the "density" argument suggests small m works.
        # If we exhaust S up to 100 without finding one, we might need to expand L_LIMIT 
        # or fall back to a greedy approach.
        
        # For now, if we reach S=100, we might have missed a solution with large m.
        # But given the divisors, a small span solution with small m is highly probable.
        
    # Fallback: Greedy search for a valid set (not necessarily optimal span)
    # This ensures we return *something* valid if the search above fails or times out.
    # We can try to construct a set by picking multiples close to a common center.
    
    # A simple valid set: the divisors themselves.
    # 6, 10, 14, 15, 21, 22, 26, 33, 35, 39
    # Span: 39 - 6 = 33.
    # This is always valid and distinct.
    
    # Let's try to improve on the divisors themselves with a quick greedy.
    # Pick a center C, and for each d, pick multiple closest to C.
    # Resolve collisions.
    
    # But the loop above should find a better one (span < 33) quickly.
    # If it didn't, return the divisors.
    
    fallback = list(_DIVISORS)
    return [int(x) for x in fallback]

# EVOLVE-BLOCK-END
