    for seed in range(40):
        if time.time() > deadline:
            break
        result = run_pass(seed)
        if result is not None and len(result) < best_score:
            best_score = len(result)
            best_blocks = result
