def crossover(a, b):
    """Order-preserving crossover on comparator sequences."""
    if len(a) < len(b):
        a, b = b, a
    cut = random.randint(1, len(b) - 1)
    child = list(b[:cut])
    seen = set(child)
    for c in a:
        if c not in seen:
            child.append(c)
            seen.add(c)
    for c in b[cut:]:
        if c not in seen:
            child.append(c)
            seen.add(c)
    return child
