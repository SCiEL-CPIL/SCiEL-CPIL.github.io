def get_forbidden_points(a, b):
    # ... existing code ...
    # Filter out a and b
    if a in forbidden: forbidden.remove(a)
    if b in forbidden: forbidden.remove(b)
    return forbidden
