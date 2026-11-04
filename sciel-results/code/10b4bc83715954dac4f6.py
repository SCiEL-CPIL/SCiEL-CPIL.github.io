import numpy as np

N = 50

def _get_max_sidon_set(n, max_val):
    """
    Generate a maximal Sidon set of size up to n within [0, max_val].
    A Sidon set has distinct pairwise sums.
    """
    S = [0]
    sums = set([0])
    for i in range(1, max_val + 1):
        if len(S) >= n:
            break
        # Check if i can be added
        valid = True
        new_sums = set()
        for s in S:
            new_sum = i + s
            if new_sum in sums or new_sum in new_sums:
                valid = False
                break
            new_sums.add(new_sum)
        two_i = 2 * i
        if two_i in sums or two_i in new_sums:
            valid = False
        
        if valid:
            S.append(i)
            sums.update(new_sums)
            sums.add(two_i)
    return S

def construct():
    # Try to find a large Sidon set
    # Max size for N=50 is around 7-8
    sidon_indices = _get_max_sidon_set(10, N - 1)
    
    # If the set is too small, try a different starting point or just use what we have
    # A known good set for N=50 is often size 7.
    if len(sidon_indices) < 5:
        # Fallback to a hardcoded set if greedy fails (shouldn't happen)
        sidon_indices = [0, 1, 4, 10, 20, 35, 49]
        
    K = len(sidon_indices)
    h = np.zeros(N)
    for idx in sidon_indices:
        h[idx] = 1.0 / K
        
    return [float(x) for x in h]
