import time
import numpy as np

N = 50

def _get_sidon_set(n):
    """
    Returns a maximal Sidon set in [0, n-1].
    A Sidon set (B_2 sequence) has the property that all pairwise sums 
    (or differences) are unique.
    For this purpose, we want distinct differences to keep side-peaks low.
    """
    # A simple greedy construction for a Sidon set
    # Or use a known small set for N=50
    # Max Sidon set size for N=50 is around 7.
    # Example: {0, 1, 4, 10, 18, 23, 36} (differences are distinct)
    # Let's verify or just use a greedy approach.
    
    S = [0]
    for i in range(1, n):
        # Check if i can be added
        # Differences with existing elements must be unique and not 0
        diffs = set()
        ok = True
        for j in S:
            d = i - j
            if d in diffs:
                ok = False
                break
            diffs.add(d)
        # Also check against existing diffs in S
        existing_diffs = set()
        for a in range(len(S)):
            for b in range(a+1, len(S)):
                existing_diffs.add(S[b] - S[a])
        
        # Actually, simpler check:
        # All differences x_i - x_j (i != j) must be distinct.
        # Let's just try to add i if it doesn't create a duplicate difference.
        
        # Re-evaluate:
        new_diffs = set()
        for j in S:
            d = i - j
            if d in new_diffs:
                ok = False
                break
            new_diffs.add(d)
        
        # Check against existing diffs in S
        if ok:
            for d in new_diffs:
                if d in existing_diffs:
                    ok = False
                    break
        
        if ok:
            S.append(i)
            # Update existing_diffs for next iteration? 
            # For efficiency in this small loop, we can just recompute or maintain.
            # Given N=50, recomputing is fine.
            
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate a Sidon set
    S = _get_sidon_set(N)
    
    # Create the height vector
    h = np.zeros(N)
    for pos in S:
        h[pos] = 1.0
        
    # Normalize to L1 norm = 1
    # This is a safe assumption given the score stability and "raw value" scaling issues.
    l1_norm = np.sum(h)
    if l1_norm > 0:
        h = h / l1_norm
        
    if time.time() > deadline:
        return h.tolist()
        
    return h.tolist()
