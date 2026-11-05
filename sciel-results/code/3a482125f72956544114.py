def construct(k, g):
    import collections
    
    # Helper to build LUW graph for given d, q
    def build_luw(d, q):
        # ... (logic from current code, parameterized) ...
        pass

    # Specific case for k=5, g=9
    if k == 5 and g == 9:
        # Try d=3 first
        n3, edges3 = build_luw(3, 5)
        # Check girth
        girth3 = compute_girth(n3, edges3)
        if girth3 >= 9:
            return n3, edges3
        
        # Fallback to d=5
        n5, edges5 = build_luw(5, 5)
        return n5, edges5
        
    # General case
    # ... existing logic ...
