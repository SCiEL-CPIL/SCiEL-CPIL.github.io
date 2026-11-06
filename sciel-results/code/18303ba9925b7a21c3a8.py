def _can_append(values, products, x):
    # New blocks are: [x], [values[-1], x], ...
    p = x
    if p in products: return False
    for a in reversed(values):
        p *= a
        if p in products: return False
    return True
