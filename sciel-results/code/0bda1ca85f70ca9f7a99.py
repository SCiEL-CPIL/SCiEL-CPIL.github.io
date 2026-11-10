<<<<<<< SEARCH
def construct():
    """Main entry point: find maximum valid set."""
    deadline = _DEADLINE
    random.seed(42)  # Deterministic
    
    # Phase 1: Band optimization (model-and-optimize)
    best = _band_optimization()
    
    # Phase 2: Greedy with various orders
    if _time_left() > 30:
        orders = [
            range(_N, 0, -1),
            range(1, _N + 1),
            sorted(range(1, _N + 1), key=lambda x: -x),
            list(range(50, _N + 1)) + list(range(1, 50)),
            list(range(80, _N + 1)) + list(range(1, 80)),
            list(range(100, _N + 1)) + list(range(1, 100)),
            list(range(120, _N + 1)) + list(range(1, 120)),
            list(range(130, _N + 1)) + list(range(1, 130)),
            list(range(140, _N + 1)) + list(range(1, 140)),
        ]
        for order in orders:
            if _time_left() < 30:
                break
            S = _greedy_optimized(order)
            if len(S) > len(best):
                best = set(S)
    
    # Phase 3: Multi-start randomized greedy
    if _time_left() > 25:
        S = _multi_start_greedy(num_starts=30)
        if len(S) > len(best):
            best = set(S)
    
    # Phase 4: Local search from best
    if _time_left() > 20:
        budget = min(18.0, _time_left() - 10)
        S = _local_search_model(best, time_budget=budget)
        if len(S) > len(best):
            best = set(S)
    
    # Phase 5: Refinement pass (high-precision polish)
    if _time_left() > 12:
        budget = min(10.0, _time_left() - 6)
        S = _refinement_pass(best, time_budget=budget)
        if len(S) > len(best):
            best = set(S)
    
    # Phase 6: Final local search if time permits
    if _time_left() > 8:
        budget = min(6.0, _time_left() - 4)
        S = _local_search_model(best, time_budget=budget)
        if len(S) > len(best):
            best = set(S)
    
    return list(best)
# EVOLVE-BLOCK-END
=======
def construct():
    """Main entry point: find maximum valid set."""
    deadline = _DEADLINE
    random.seed(42)  # Deterministic
    
    # Phase 1: Band optimization (model-and-optimize)
    best = _band_optimization()
    
    # Phase 2: Greedy with various orders
    if _time_left() > 30:
        orders = [
            range(_N, 0, -1),
            range(1, _N + 1),
            list(range(50, _N + 1)) + list(range(1, 50)),
            list(range(80, _N + 1)) + list(range(1, 80)),
            list(range(100, _N + 1)) + list(range(1, 100)),
            list(range(120, _N + 1)) + list(range(1, 120)),
            list(range(130, _N + 1)) + list(range(1, 130)),
            list(range(140, _N + 1)) + list(range(1, 140)),
        ]
        for order in orders:
            if _time_left() < 30:
                break
            S = _greedy_optimized(order)
            if len(S) > len(best):
                best = set(S)
    
    # Phase 3: Multi-start randomized greedy
    if _time_left() > 25:
        S = _multi_start_greedy(num_starts=30)
        if len(S) > len(best):
            best = set(S)
    
    # Phase 4: Backtracking to try to reach 48
    if _time_left() > 20:
        target = _MAX
        budget = min(15.0, _time_left() - 10)
        S = _backtrack_search(best, target, time_budget=budget)
        if len(S) > len(best):
            best = set(S)
    
    # Phase 5: Local search from best
    if _time_left() > 15:
        budget = min(12.0, _time_left() - 8)
        S = _local_search_model(best, time_budget=budget)
        if len(S) > len(best):
            best = set(S)
    
    # Phase 6: Refinement pass (high-precision polish)
    if _time_left() > 10:
        budget = min(8.0, _time_left() - 5)
        S = _refinement_pass(best, time_budget=budget)
        if len(S) > len(best):
            best = set(S)
    
    # Phase 7: Final local search if time permits
    if _time_left() > 6:
        budget = min(4.0, _time_left() - 3)
        S = _local_search_model(best, time_budget=budget)
        if len(S) > len(best):
            best = set(S)
    
    return list(best)
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

