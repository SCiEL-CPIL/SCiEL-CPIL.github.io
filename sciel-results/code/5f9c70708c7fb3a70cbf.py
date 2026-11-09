import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Try to find a 6-term solution
    # ... (search logic) ...
    
    # Fallback to DP for 7+ terms
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if time.time() > deadline:
            break
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
            
    raise RuntimeError("no baseline representation")
