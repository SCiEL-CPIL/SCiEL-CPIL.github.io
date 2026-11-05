    if k == 5 and g == 9:
        # Try smaller d first
        for d_try in [4, 5, 6]:
            n, edges = _luw_construct(k, g, d_try)
            girth = _check_girth(n, edges, g)
            if girth >= g:
                return n, edges
        # Fallback to original logic if none worked (shouldn't happen if d=5 works)
        return _luw_construct(k, g)
