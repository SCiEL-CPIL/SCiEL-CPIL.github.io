<<<<<<< SEARCH
    # Phase 3: Branch and bound with greedy coloring
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        bb_classes = _branch_and_bound_solve(time.time() + remaining_time - 2.0)
        if _is_valid(bb_classes) and _score(bb_classes) > _score(classes):
            classes = bb_classes
=======
    # Phase 3: Branch and bound with greedy coloring
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        bb_classes = _branch_and_bound_solve(time.time() + remaining_time - 3.0)
        if _is_valid(bb_classes) and _score(bb_classes) > _score(classes):
            classes = bb_classes
    elif remaining_time > 3.0:
        # Quick local improvement pass
        sa_classes = _simulated_annealing(classes, time_budget=remaining_time - 1.0)
        if _is_valid(sa_classes) and _score(sa_classes) > _score(classes):
            classes = sa_classes
>>>>>>> REPLACE
