    if k == 5 and g == 9:
        # d=4 gives size 2*5^4 = 1250
        # We want the smallest n, so try d=4 first
        n, edges = _construct_luw(k, 4)
        # Check if girth is sufficient
        if _girth_at_least(n, edges, g):
            return n, edges
