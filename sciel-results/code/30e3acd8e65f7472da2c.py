def design(problem):
    t0 = time.time()
    deadline = t0 + 170.0
    N = problem.get("pixels", 32)
    rng = random.Random(12345)
    best_score = -1e9
    best_params = (0.2, 0.0, 0.0, 0)  # seed-like
    # Phase 1: random
    n_rand = 400
    for it in range(n_rand):
        if time.time() > deadline - 30: break
        r0 = rng.uniform(0.0, 0.45)
        r1 = rng.uniform(0.05, 0.55)
        w1 = rng.uniform(0.0, 0.25)
        invert = rng.choice([0,1])
        s = _surrogate(r0, r1, w1, invert)
        if s > best_score:
            best_score = s; best_params = (r0,r1,w1,invert)
    # Phase 2: local search from best
    ...
    cell = _build_cell(*best_params)
    return cell.tolist()
