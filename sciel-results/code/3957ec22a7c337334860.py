def construct():
    start = time.time()
    deadline = 55.0  # safety margin before 60s kill

    best_m = 0
    best_L = 0

    # We know f(12, m) <= 144.
    # If can_match(m, 143) is False, then f(12, m) >= 144.
    # Since f(12, m) <= 144, f(12, m) == 144.
    # This is the maximum possible value.
    
    for m in range(PERIOD):
        if time.time() - start > deadline:
            break
        
        # Check if L=143 is sufficient
        if can_match(m, 143):
            # L <= 143
            # Check if L=142 is sufficient to see if L=143 is the exact min
            if not can_match(m, 142):
                if best_L < 143:
                    best_L = 143
                    best_m = m
            # If can_match(m, 142) is True, L < 143, ignore for now
        else:
            # L > 143, so L >= 144
            # Since max is 144, L == 144
            return int(m)

    return int(best_m)
