<<<<<<< SEARCH
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        n, p, seed = spec
        paths = solve_instance(n, p, seed)
        results.append(paths)
        
        if time.time() > deadline:
            # If we ran out of time, fill remaining with trivial solution?
            # But we should have finished.
            break
            
    # If we didn't finish all specs, ensure we return a valid structure
    while len(results) < 3:
         # This should not happen if time management is correct
         results.append([])

    return results
=======
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    # Allocate time: ~45s per instance
    time_per_instance = 45.0
    instance_deadlines = []
    for i in range(3):
        instance_deadlines.append(start_time + (i + 1) * time_per_instance)
    
    for idx, spec in enumerate(specs):
        n, p, seed = spec
        # Temporarily adjust deadline for this instance
        global deadline
        old_deadline = deadline
        deadline = instance_deadlines[idx]
        
        paths = solve_instance(n, p, seed)
        results.append(paths)
        
        deadline = old_deadline
        
        if time.time() > deadline:
            break
            
    # If we didn't finish all specs, ensure we return a valid structure
    while len(results) < 3:
         results.append([])

    return results
>>>>>>> REPLACE

