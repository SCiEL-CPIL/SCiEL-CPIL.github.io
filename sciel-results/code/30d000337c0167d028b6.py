    A = np.array(_get_petersen_walks(g-1), dtype=np.int32)
    # A is (N_walks, 15)
    
    for m in range(52, 100):
        # ...
        for attempt in range(10000): # Increase attempts
             # ...
             voltages = rng.randint(1, m, size=15)
             sums = (A @ voltages) % m
             if not np.any(sums == 0):
                 # Found
