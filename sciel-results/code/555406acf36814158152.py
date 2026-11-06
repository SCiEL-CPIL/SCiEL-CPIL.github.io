<<<<<<< SEARCH
    # --- Initialization: try several seeds, keep best ---
    rng = random.Random(12345)
    best_assign = None
    best_unsat = m + 1

    # Generate candidate initial assignments
    candidates = []
    for _ in range(12):
        a = [rng.random() < 0.5 for _ in range(n)]
        candidates.append(a)
    candidates.append([True] * n)
    candidates.append([False] * n)
    candidates.append([i % 2 == 0 for i in range(n)])
    candidates.append([i % 3 == 0 for i in range(n)])
=======
    # --- Initialization: try several seeds, keep best ---
    rng = random.Random(12345)
    best_assign = None
    best_unsat = m + 1

    # Try spectral relaxation first
    spec_assign = spectral_init()
    if spec_assign is not None:
        if is_satisfied(spec_assign):
            return spec_assign
        u_spec = unsat_count(spec_assign)
        if u_spec < best_unsat:
            best_unsat = u_spec
            best_assign = list(spec_assign)

    # Generate candidate initial assignments
    candidates = []
    for _ in range(12):
        a = [rng.random() < 0.5 for _ in range(n)]
        candidates.append(a)
    candidates.append([True] * n)
    candidates.append([False] * n)
    candidates.append([i % 2 == 0 for i in range(n)])
    candidates.append([i % 3 == 0 for i in range(n)])
    if spec_assign is not None:
        candidates.append(list(spec_assign))
>>>>>>> REPLACE

