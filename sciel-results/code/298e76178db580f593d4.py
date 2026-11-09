def _is_valid_addition(triple, triple_set):
    """
    Check if adding triple (a,b,c) creates a K(2,2,2).
    A K(2,2,2) is formed if there exist a', b', c' such that all 8 transversals
    of pairs (a,a'), (b,b'), (c,c') are in the set.
    Since (a,b,c) is the one being added, we need the other 7 to be in triple_set.
    """
    a, b, c = triple
    # Iterate over all possible a', b', c'
    # a' != a, b, c
    # b' != a, b, c, a'
    # c' != a, b, c, a', b'
    for a_prime in range(16):
        if a_prime in (a, b, c):
            continue
        for b_prime in range(16):
            if b_prime in (a, b, c, a_prime):
                continue
            for c_prime in range(16):
                if c_prime in (a, b, c, a_prime, b_prime):
                    continue
                
                # Check the other 7 transversals
                # (a', b, c)
                if (a_prime, b, c) not in triple_set: continue
                # (a, b', c)
                if (a, b_prime, c) not in triple_set: continue
                # (a, b, c')
                if (a, b, c_prime) not in triple_set: continue
                # (a', b', c)
                if (a_prime, b_prime, c) not in triple_set: continue
                # (a', b, c')
                if (a_prime, b, c_prime) not in triple_set: continue
                # (a, b', c')
                if (a, b_prime, c_prime) not in triple_set: continue
                # (a', b', c')
                if (a_prime, b_prime, c_prime) not in triple_set: continue
                
                # All 8 are present, so this addition is invalid
                return False
    return True
