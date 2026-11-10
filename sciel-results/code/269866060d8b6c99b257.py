<<<<<<< SEARCH
    # Phase 4: More random sampling
    while time_left() > 5:
        s = random.randint(1, 10**12 - 200)
        k = find_run(s)
        if k > best_k:
            best_k = k
            best_s = s
        attempts += 1
=======
    # Phase 4: More random sampling
    # Use a tighter time limit to ensure Phase 5 runs
    while time_left() > 10:
        s = random.randint(1, 10**12 - 200)
        k = find_run(s)
        if k > best_k:
            best_k = k
            best_s = s
        attempts += 1
>>>>>>> REPLACE
