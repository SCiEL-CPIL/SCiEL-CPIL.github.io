import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    selected = []
    reachable = {0}
    
    # Iterate from smallest to largest to maximize cardinality
    # Smaller numbers allow more elements to be included
    for x in range(1, 101):
        # Check if adding x creates a sum of 200
        # We only care about sums <= 200
        new_sums = set()
        for s in reachable:
            ns = s + x
            if ns <= 200:
                new_sums.add(ns)
        
        if 200 not in new_sums:
            selected.append(x)
            reachable.update(new_sums)
            
        # Check time (though this loop is extremely fast)
        if time.time() > deadline:
            break
            
    return selected
