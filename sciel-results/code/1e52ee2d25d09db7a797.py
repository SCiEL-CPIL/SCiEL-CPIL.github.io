# EVOLVE-BLOCK-START
P = {'n': 40, 'maximum_size': 18}

def _is_valid_subset(set_vals):
    """Check if all subset products are distinct."""
    k = len(set_vals)
    if k > 20: # 2^20 is ~1M, might be slow if not optimized, but our k is 16. 2^16 = 65536. Fast.
        return False
    
    products = set()
    # Use bitmasks to enumerate subsets
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= set_vals[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def construct():
    # Direct construction based on prime power analysis
    # 2: {2, 4, 16}
    # 3: {3, 9}
    # 5: {5, 25}
    # Others: {p}
    candidate = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify validity to be safe
    if _is_valid_subset(candidate):
        return candidate
    
    # Fallback to primes if candidate is invalid (should not happen)
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
# EVOLVE-BLOCK-END
