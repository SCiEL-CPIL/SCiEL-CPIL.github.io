def get_classic_order(n):
    if n == 1:
        return [(1,)]
    order_prev = get_classic_order(n - 1)
    # The order in W_n is derived from the windows of W_{n-1}
    # But we can just generate the words recursively to find the order?
    # That's expensive.
    # Actually, the order is just the list of permutations in the order they appear in the superpermutation of n-1.
    # We can simulate the construction to get the order.
    
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
        
    # Now extract the order for n
    perms, seen = [], set()
    for i in range(len(word) - n + 2):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    return perms
