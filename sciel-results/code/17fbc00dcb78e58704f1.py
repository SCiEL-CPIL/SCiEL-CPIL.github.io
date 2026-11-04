    # If the linear algebra approach didn't work, try numerical optimization
    if best_rank == n*m*p and time.time() < deadline:
