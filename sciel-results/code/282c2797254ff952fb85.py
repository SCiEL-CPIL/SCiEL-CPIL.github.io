<<<<<<< SEARCH
def _find_first_gap(residues, primes, max_check=2000):
    """Find the smallest x >= 1 such that coverage < 2. Returns x, or max_check+1 if none."""
    for x in range(1, max_check + 1):
        count = 0
        for i, p in enumerate(primes):
            if x % p == residues[i]:
                count += 1
                if count >= 2:
                    break
        if count < 2:
            return x
    return max_check + 1

def _score(residues, primes):
    """Returns the length L. Uses _find_first_gap but allows checking higher if initial check passes."""
    # We can just return (gap - 1)
    # But we want to maximize. If gap is at 2001, score is 2000.
    # We should probably check up to a higher limit if the first gap is far.
    # But for search efficiency, a proxy is fine.
    gap = _find_first_gap(residues, primes, 2000)
    if gap > 2000:
        # It passed 2000, so it's at least 2000.
        # Let's check a bit more to be precise for the final return, 
        # but for search steps, 2000 is a good cap to keep it fast.
        # Actually, let's just return gap - 1.
        # If we want to be more accurate for the final answer, we can re-evaluate.
        pass
    return gap - 1
=======
def _score_vectorized(residues, primes, limit):
    """
    Returns the largest L such that every integer 1..L satisfies at least two congruences.
    Uses vectorized NumPy operations.
    """
    if limit < 1:
        return 0
        
    counts = np.zeros(limit + 1, dtype=np.int16)
    
    for i, p in enumerate(primes):
        r = residues[i]
        
        # Determine start index
        if r == 0:
            start = p
        else:
            start = r
            if start < 1:
                start = 1
                
        if start <= limit:
            counts[start:limit+1:p] += 1
            
    # Find first x >= 1 where counts[x] < 2
    # counts[0] is unused for the logic (we care about 1..limit)
    deficient = counts[1:] < 2
    
    if not np.any(deficient):
        return limit
    else:
        # argmax returns index of first True
        # This index is relative to counts[1:]
        # So if index is k, the x value is k + 1
        # The score L is x - 1 = k
        return np.argmax(deficient)
>>>>>>> REPLACE
