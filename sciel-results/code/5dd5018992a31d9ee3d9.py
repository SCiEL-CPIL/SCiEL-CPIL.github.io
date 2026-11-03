<<<<<<< SEARCH
    # Run multiple strategies
    best_selected = None
    best_len = float('inf')

    configs = [
        ('primal_dual', 42),
        ('primal_dual', 123),
        ('inverse', 42),
        ('random', 123),
        ('random', 777),
        ('exp', 555),
        ('random', 111),
        ('inverse', 999),
        ('primal_dual', 555),
        ('random', 2024),
        ('exp', 314),
    ]

    for i, (mode, seed) in enumerate(configs):
        if time_left() < 20:
            break
        if mode == 'primal_dual':
            sel = run_primal_dual(seed=seed)
        else:
            sel = run_greedy(mode, seed)
        sel = local_search(sel, 0.12)
        if len(sel) < best_len:
            best_len = len(sel)
            best_selected = sel[:]

    # Final polish
    if best_selected and time_left() > 15:
        best_selected = local_search(best_selected, 0.25)
=======
    # Run multiple strategies
    best_selected = None
    best_len = float('inf')

    configs = [
        ('primal_dual', 42),
        ('primal_dual', 123),
        ('inverse', 42),
        ('random', 123),
        ('random', 777),
        ('exp', 555),
        ('random', 111),
        ('inverse', 999),
        ('primal_dual', 555),
        ('random', 2024),
        ('exp', 314),
        ('random', 3141),
        ('inverse', 2718),
    ]

    for i, (mode, seed) in enumerate(configs):
        if time_left() < 25:
            break
        if mode == 'primal_dual':
            sel = run_primal_dual(seed=seed)
        else:
            sel = run_greedy(mode, seed)
        sel = local_search(sel, 0.12)
        if len(sel) < best_len:
            best_len = len(sel)
            best_selected = sel[:]

    # Final polish with more time
    if best_selected and time_left() > 20:
        best_selected = local_search(best_selected, 0.25)
>>>>>>> REPLACE
