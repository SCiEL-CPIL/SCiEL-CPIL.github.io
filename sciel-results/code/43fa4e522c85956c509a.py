def _get_products(seq):
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods

def _check_insert(seq, pos, val, existing_products):
    """Check if inserting val at pos in seq creates duplicate block products
    AND maintains strictly increasing order."""
    # Check strictly increasing order
    if pos > 0 and seq[pos - 1] >= val:
        return False
    if pos < len(seq) and val >= seq[pos]:
        return False
    
    n = len(seq)
    # New sequence would be seq[:pos] + [val] + seq[pos:]
    # Blocks containing val: start i in [0, pos], end j in [pos, pos + n - 1]
    # Product of block [i, j] in new seq = product of seq[i:pos] * val * product of seq[pos:j+1]
    
    # Left: product of seq[i:pos] for i in 0..pos
    left_prods = [1] * (pos + 1)
    prod = 1
    for i in range(pos - 1, -1, -1):
        prod *= seq[i]
        left_prods[i] = prod
    # left_prods[i] = product of seq[i:pos]
    
    # Right: product of seq[pos:j+1] for j in pos..pos+n-1
    right_prods = [1] * (n - pos + 1)
    prod = 1
    for j in range(n - pos - 1, -1, -1):
        prod *= seq[pos + j]
        right_prods[j] = prod
    # right_prods[j] = product of seq[pos:pos+j+1]
    
    # Now check all blocks containing val
    new_prods = set()
    for i in range(pos + 1):
        for j in range(n - pos + 1):
            p = left_prods[i] * val * right_prods[j]
            # Check against existing products (blocks not containing val)
            if p in existing_products:
                return False
            # Check against other new products
            if p in new_prods:
                return False
            new_prods.add(p)
    return True

def _try_insert(seq, pos, val, existing_products):
    """Try inserting val at position pos in seq. Return new seq if valid, else None."""
    if not _check_insert(seq, pos, val, existing_products):
        return None
    return seq[:pos] + [val] + seq[pos:]
