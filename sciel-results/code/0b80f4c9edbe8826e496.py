<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    best_set = []
    best_size = 0

    # Strategy 1: Ascending greedy
    set1 = _greedy_asc()
    if len(set1) > best_size:
        best_size = len(set1)
        best_set = set1

    if time.time() > deadline - 100:
        return _polish(best_set, deadline)

    # Strategy 2: Descending greedy
    set2 = _greedy_desc()
    if len(set2) > best_size:
        best_size = len(set2)
        best_set = set2

    if time.time() > deadline - 90:
        return _polish(best_set, deadline)

    # Strategy 3: Multiple random-order greedies
    for trial in range(10):
        if time.time() > deadline - 80:
            break
        s = _greedy_random(seed=trial * 7 + 1)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    if time.time() > deadline - 70:
        return _polish(best_set, deadline)

    # Strategy 4: Density-aware greedy
    for trial in range(5):
        if time.time() > deadline - 60:
            break
        s = _greedy_density_aware(seed=trial * 13 + 3)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    if time.time() > deadline - 50:
        return _polish(best_set, deadline)

    # Strategy 5: Saturating greedy
    for trial in range(3):
        if time.time() > deadline - 40:
            break
        s = _saturating_greedy(seed=trial * 17 + 5)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    if time.time() > deadline - 30:
        return _polish(best_set, deadline)

    # Strategy 6: Conflict graph greedy
    for trial in range(3):
        if time.time() > deadline - 20:
            break
        s = _conflict_graph_greedy(seed=trial * 23 + 7)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Final polish on the best set with insert-based local search
    if time.time() < deadline - 5:
        polished = _polish(best_set, deadline)
        if len(polished) > len(best_set):
            best_set = polished

    return best_set
=======
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    best_set = []
    best_size = 0

    # Strategy 1: Ascending greedy
    set1 = _greedy_asc()
    if len(set1) > best_size:
        best_size = len(set1)
        best_set = set1

    if time.time() > deadline - 100:
        return _polish(best_set, deadline)

    # Strategy 2: Descending greedy
    set2 = _greedy_desc()
    if len(set2) > best_size:
        best_size = len(set2)
        best_set = set2

    if time.time() > deadline - 90:
        return _polish(best_set, deadline)

    # Strategy 3: Multiple random-order greedies
    for trial in range(10):
        if time.time() > deadline - 80:
            break
        s = _greedy_random(seed=trial * 7 + 1)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    if time.time() > deadline - 70:
        return _polish(best_set, deadline)

    # Strategy 4: Density-aware greedy
    for trial in range(5):
        if time.time() > deadline - 60:
            break
        s = _greedy_density_aware(seed=trial * 13 + 3)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    if time.time() > deadline - 50:
        return _polish(best_set, deadline)

    # Strategy 5: Saturating greedy
    for trial in range(3):
        if time.time() > deadline - 40:
            break
        s = _saturating_greedy(seed=trial * 17 + 5)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    if time.time() > deadline - 30:
        return _polish(best_set, deadline)

    # Strategy 6: Conflict graph greedy
    for trial in range(3):
        if time.time() > deadline - 20:
            break
        s = _conflict_graph_greedy(seed=trial * 23 + 7)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Strategy 7: Simulated annealing from best greedy result
    if time.time() < deadline - 35:
        sa_deadline = deadline - 30
        sa_result = _simulated_annealing(best_set, deadline=sa_deadline, max_steps=30000)
        if len(sa_result) > best_size:
            best_size = len(sa_result)
            best_set = sa_result

    # Final polish on the best set with insert-based local search
    if time.time() < deadline - 5:
        polished = _polish(best_set, deadline)
        if len(polished) > len(best_set):
            best_set = polished

    return best_set
>>>>>>> REPLACE
