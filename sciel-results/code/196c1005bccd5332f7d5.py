<<<<<<< SEARCH
def greedy_construct(order, deadline_margin=5.0):
    """Greedily build a valid set by adding elements in the given order using bitsets."""
    s = []
    bitsets = [1]  # bitsets[0] = 1 (empty sum)
    n = 0
    for val in order:
        if elapsed() > DEADLINE - deadline_margin:
            break
        # Try adding val
        n_new = n + 1
        # Build new bitsets
        new_bitsets = [0] * (n_new + 1)
        new_bitsets[0] = bitsets[0]
        for k in range(n_new, 0, -1):
            if k - 1 <= n:
                new_bitsets[k] |= (bitsets[k - 1] << val)
            if k <= n:
                new_bitsets[k] |= bitsets[k]
        
        # Check validity
        valid = True
        for i in range(n_new + 1):
            for j in range(i + 1, n_new + 1):
                if new_bitsets[i] & new_bitsets[j]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            s.append(val)
            bitsets = new_bitsets
            n = n_new
    return s
=======
def greedy_construct(order, deadline_margin=5.0):
    """Greedily build a valid set by adding elements in the given order using bitsets."""
    s = []
    bitsets = [1]  # bitsets[0] = 1 (empty sum)
    n = 0
    for val in order:
        if elapsed() > DEADLINE - deadline_margin:
            break
        # Try adding val
        n_new = n + 1
        # Build new bitsets
        new_bitsets = [0] * (n_new + 1)
        new_bitsets[0] = bitsets[0]
        for k in range(1, n_new + 1):
            if k <= n:
                new_bitsets[k] |= bitsets[k]
            if k - 1 <= n:
                new_bitsets[k] |= (bitsets[k - 1] << val)
        
        # Check validity
        valid = True
        for i in range(n_new + 1):
            for j in range(i + 1, n_new + 1):
                if new_bitsets[i] & new_bitsets[j]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            s.append(val)
            bitsets = new_bitsets
            n = n_new
    return s
>>>>>>> REPLACE

