import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Sieve for primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False

    primes = [i for i, v in enumerate(is_prime) if v]
    N = len(primes)

    if N == 0:
        return []

    # Layered DP
    # current_layer: dict {prime_index: min_last_gap}
    # For length 1, there is no last gap. We can treat it as 0.
    # A sequence of length 1 ending at j has "last gap" 0.
    current_layer = {i: 0 for i in range(N)}
    
    best_len = 1
    best_parent = [-1] * N # parent[i] for the sequence of length best_len ending at i?
    # We need to reconstruct the path.
    # In layered DP, we can store parent pointers for each layer.
    # parent[l][i] = j, where j is the predecessor of i in the sequence of length l.
    # But we only keep the min gap. If multiple j lead to same min gap, pick any.
    # Actually, we need the specific j that provided the min gap.
    
    # Let's store:
    # dp_gap[l][i] = min last gap for sequence of length l ending at i
    # dp_parent[l][i] = index of previous prime
    # But l is dynamic.
    
    # We'll keep a list of layers.
    # layers[l] = {i: (gap, parent_i)}
    # layers[1] = {i: (0, -1) for i in range(N)}
    
    layers = []
    current = {i: (0, -1) for i in range(N)}
    layers.append(current)
    
    while True:
        if time.time() > deadline:
            break
            
        next_layer = {}
        # For each i in current layer, try to extend to j > i
        for i, (gap_i, _) in current.items():
            pi = primes[i]
            # We need j > i such that primes[j] - pi >= gap_i
            # Since primes are sorted, we can binary search for the first j
            # such that primes[j] >= pi + gap_i
            min_val = pi + gap_i
            # Find lower bound in primes[i+1:]
            # Using bisect
            import bisect
            start_idx = i + 1
            # We want the first index k >= start_idx such that primes[k] >= min_val
            k = bisect.bisect_left(primes, min_val, start_idx, N)
            
            # For all j from k to N-1, we can extend.
            # The new gap is primes[j] - pi.
            # We want to update next_layer[j] with (new_gap, i)
            # But we want the MIN new_gap for each j.
            # Since primes[j] is increasing, primes[j] - pi is increasing.
            # So the minimum new_gap for a fixed i is at j=k.
            # However, multiple i's can update the same j.
            # We need to take the min over all valid i.
            
            # Optimization:
            # For a fixed j, which i's are valid?
            # i < j and primes[j] - primes[i] >= gap_i.
            # This is hard to invert.
            
            # Let's just iterate j from k to N-1.
            # But this is O(N^2) per layer.
            # N=1229. N^2 = 1.5M.
            # If L=20, 30M operations. Acceptable.
            
            for j in range(k, N):
                new_gap = primes[j] - pi
                # Update next_layer[j]
                if j not in next_layer:
                    next_layer[j] = (new_gap, i)
                else:
                    # Keep min gap
                    if new_gap < next_layer[j][0]:
                        next_layer[j] = (new_gap, i)
                        
        if not next_layer:
            break
            
        layers.append(next_layer)
        current = next_layer
        best_len += 1

    if best_len == 1:
        return [primes[0]] # Or any prime

    # Reconstruct
    # We have layers[1] ... layers[best_len]
    # We need to pick a j in layers[best_len]
    # To maximize length, any j in the last layer is fine.
    # Let's pick the one with the smallest index? Or any.
    # Let's pick the first one.
    
    # Actually, we should pick the one that allows reconstruction.
    # All entries in the last layer are valid endpoints.
    
    curr_idx = list(current.keys())[0]
    curr_len = best_len
    
    path = []
    while curr_len > 0:
        path.append(primes[curr_idx])
        if curr_len == 1:
            break
        # Get parent from layers[curr_len]
        # layers[curr_len] is the layer for length curr_len
        # The entry is (gap, parent)
        _, parent_idx = layers[curr_len][curr_idx]
        curr_idx = parent_idx
        curr_len -= 1
        
    path.reverse()
    return path
