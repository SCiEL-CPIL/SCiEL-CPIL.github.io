def priority(vector, n):
    # Count distinct symbols in the vector
    distinct = len(set(vector))
    # Count occurrences of each symbol (0, 1, 2)
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    # L1 deviation from uniform distribution (n/3 of each symbol)
    target = n / 3.0
    l1_dev = abs(c0 - target) + abs(c1 - target) + abs(c2 - target)
    # Priority: distinct count dominates, balance breaks ties
    # Higher distinct count = more generic = less likely to form lines
    # Lower L1 deviation = more balanced = less structured
    return distinct * 1000.0 - l1_dev
