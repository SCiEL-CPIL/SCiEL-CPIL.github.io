def luw_construct(k, g):
    q = k
    # Determine d
    # Standard LUW girth is approx 2d+1
    # We want girth >= g
    # Try d such that 2d+1 >= g => d >= (g-1)/2
    d_min = (g - 1) // 2
    if (g - 1) % 2 != 0:
        d_min += 1 # If g=10, (10-1)//2 = 4.5 -> 4? No.
        # g=10, 2d+1 >= 10 => 2d >= 9 => d >= 4.5 => d=5.
        # But if girth is 2d+2, then 2d+2 >= 10 => 2d >= 8 => d=4.
    
    # Let's try d=4 for g=10
    candidates = []
    if g == 10 and k == 4:
        candidates = [4, 5]
    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        candidates = [d]
        
    for d in candidates:
        # Construct graph with this d
        # ... (code to build graph) ...
        # Check girth
        # If girth >= g, return this graph
        pass
