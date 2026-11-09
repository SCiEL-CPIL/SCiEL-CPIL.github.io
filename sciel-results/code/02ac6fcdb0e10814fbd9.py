import array
import bisect

P = {'maximum': 10000000, 'target': [3, 2]}

def construct():
    N = P['maximum']
    
    # Use array for memory efficiency
    # sigma[n] can be up to ~10^8, fits in signed 32-bit? 2^31 is 2.1e9. Yes.
    # But to be safe, let's check max possible sigma.
    # sigma(n) < n * (1 + 1/2 + ... + 1/n) approx n log log n.
    # For 10^7, log log 10^7 is small. sigma(10^7) is around 10^8.
    # So 'i' (signed int, 4 bytes) is safe.
    sigma = array.array('i', [0]) * (N + 1)
    p_power = array.array('i', [0]) * (N + 1)
    sigma_p_power = array.array('i', [0]) * (N + 1)
    is_composite = bytearray(N + 1)
    primes = []
    
    sigma[1] = 1
    p_power[1] = 1
    sigma_p_power[1] = 1
    
    for i in range(2, N + 1):
        if not is_composite[i]:
            primes.append(i)
            sigma[i] = 1 + i
            p_power[i] = i
            sigma_p_power[i] = 1 + i
            
        for p in primes:
            if i * p > N:
                break
            is_composite[i * p] = 1
            
            if p == primes[0] or (i % p == 0): # p is smallest prime factor of i
                # Actually, in linear sieve, the loop breaks when p divides i.
                # So if we reach this point, p is the smallest prime factor of i*p?
                # No, p is a prime factor. If i % p == 0, then p is the smallest prime factor of i (since primes are iterated in increasing order).
                p_power[i * p] = p_power[i] * p
                sigma_p_power[i * p] = sigma_p_power[i] + p_power[i * p]
                # sigma(i*p) = sigma(i) / sigma(p^k) * sigma(p^(k+1))
                # sigma(i) = sigma(p^k) * sigma(m)
                # sigma(i*p) = sigma(p^(k+1)) * sigma(m)
                # sigma(i*p) = sigma(i) * (sigma(p^(k+1)) / sigma(p^k))
                sigma[i * p] = (sigma[i] // sigma_p_power[i]) * sigma_p_power[i * p]
                break
            else:
                p_power[i * p] = p
                sigma_p_power[i * p] = 1 + p
                sigma[i * p] = sigma[i] * (1 + p)
                
    # Now we have sigma[1..N]
    # Create list of indices 1..N
    indices = list(range(1, N + 1))
    
    # Sort indices by sigma value
    # Using lambda might be slow. 
    # indices.sort(key=lambda x: sigma[x])
    # Let's try to optimize this.
    # We can create a list of tuples (sigma[i], i) and sort that?
    # That uses more memory.
    # Let's stick to lambda.
    indices.sort(key=lambda x: sigma[x])
    
    best_a = -1
    best_b = -1
    best_err_num = 10**18 # Numerator of error |2a-3b|
    best_err_den = 1
    
    # Iterate through sorted indices to find groups with same sigma
    i = 0
    while i < N:
        s = sigma[indices[i]]
        j = i
        while j < N and sigma[indices[j]] == s:
            j += 1
            
        # Group is indices[i:j]
        # Extract group to a list for binary search
        # This copy takes O(K)
        group = indices[i:j]
        
        if len(group) >= 2:
            # For each b in group, find best a
            # group is sorted
            for b in group:
                target = 1.5 * b
                # Find position of target in group
                pos = bisect.bisect_left(group, target)
                
                # Check pos and pos-1
                candidates = []
                if pos < len(group):
                    candidates.append(group[pos])
                if pos > 0:
                    candidates.append(group[pos-1])
                
                for a in candidates:
                    if a == b:
                        continue
                    # Error is |a/b - 1.5| = |2a - 3b| / 2b
                    num = abs(2*a - 3*b)
                    if num == 0:
                        # Invalid, skip
                        continue
                    den = 2*b
                    
                    # Compare num/den with best_err_num/best_err_den
                    # num/den < best_num/best_den
                    # num * best_den < best_num * den
                    if num * best_err_den < best_err_num * den:
                        best_err_num = num
                        best_err_den = den
                        best_a = a
                        best_b = b
                        
        i = j
        
    if best_a == -1:
        # Fallback, should not happen
        return [1, 1]
        
    return [best_a, best_b]
