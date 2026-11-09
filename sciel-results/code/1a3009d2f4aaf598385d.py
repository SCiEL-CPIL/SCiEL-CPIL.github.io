import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    colors = [-1] * n
    
    # Precompute for each position, which APs it participates in as the last element
    # And which APs it affects
    aps_ending_at = [[] for _ in range(n)]
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            e = a + 3 * d
            aps_ending_at[e].append((a, a + d, a + 2 * d))
    
    # Greedy: for each position, try colors 0,1,2 and pick the first that works
    for pos in range(n):
        for color in range(3):
            colors[pos] = color
            valid = True
            for (a, b, c) in aps_ending_at[pos]:
                if len({colors[a], colors[b], colors[c], color}) < 3:
                    valid = False
                    break
            if valid:
                break
        else:
            # Should not happen with 3 colors, but fallback
            colors[pos] = 0
    
    # Verify and return
    for pos in range(n):
        for (a, b, c) in aps_ending_at[pos]:
            assert len({colors[a], colors[b], colors[c], colors[pos]}) >= 3, f"Failed at pos {pos}"
    
    return colors
