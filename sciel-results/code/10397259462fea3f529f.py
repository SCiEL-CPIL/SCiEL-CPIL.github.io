<<<<<<< SEARCH
    # WalkSAT loop
    p_flip_unsat = 0.5  # probability to flip a variable in an unsatisfied clause
    max_steps = 200000

    for step in range(max_steps):
        if time.time() > deadline - 5.0:
            break
=======
    # WalkSAT loop
    p_flip_unsat = 0.5  # probability to flip a variable in an unsatisfied clause
    # Use remaining time for WalkSAT
    remaining_time = deadline - 15.0 - time.time()
    if remaining_time < 5.0:
        # Not enough time for WalkSAT, skip to final reduction
        pass
    else:
        # Estimate steps based on time
        max_steps = int(remaining_time * 10000)  # ~10k steps per second estimate
        max_steps = max(10000, min(max_steps, 500000))

        for step in range(max_steps):
            if time.time() > deadline - 15.0:
                break
>>>>>>> REPLACE

