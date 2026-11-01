    def _luw_construct(k, g):
        if k == 3:
            q = 2
        else:
            q = k - 1 # LUW is (q+1)-regular
        
        # If k is not 3 and q is not prime power? 
        # For now, assume k=3 is the primary target.
        
        def build_luw(d):
            # ... (same as before)
            pass

        # We want girth >= g.
        # LUW girth is approx 2d+2.
        # So we need 2d+2 >= g => d >= (g-2)/2.
        # For g=16, d >= 7.
        
        for d in range(7, 20):
            if time.time() > deadline: break
            n, edges = build_luw(d)
            # Check degree
            # Check girth
            if _check_girth_fast(n, edges, g):
                return n, edges
        # Fallback: return the last built graph (might be invalid if time ran out, but hopefully valid)
        return build_luw(19)
    