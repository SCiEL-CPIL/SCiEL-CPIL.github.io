    while time.time() < deadline:
        # 1. Try to grow by scanning P
        # This is O(|P| * |S|)
        # If |S| ~ 500, |P| ~ 500, ops ~ 250,000.
        # 120s / 250k = 480 iterations.
        # This is very slow for local search.
        
        # Better: Random sampling for growth.
        # If we find a valid element, add it.
        # This is O(1) per attempt.
