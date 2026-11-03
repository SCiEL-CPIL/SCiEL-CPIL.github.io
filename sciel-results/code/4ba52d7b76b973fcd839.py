<<<<<<< SEARCH
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # Theoretical lower bound: need n_rows * k*(k+1)/2 distinct positive differences
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1  # differences are 1..M, so max entry >= num_diffs - 1

    k_plus_1 = k + 1

    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. Pull-to-smallest compaction (structural: rebuild each row greedily)
    best_solution = _pull_compact(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 3. Kick-and-pull: escape local minima by perturbing a row then re-pulling
    best_solution = _kick_and_pull(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 4. Randomized backtracking for tighter target scopes
    attempt = 0
    while time.time() < deadline - 5.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        found = False
        for _ in range(60):
            if time.time() > deadline - 5.0:
                break

            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # Compact the new solution
                    improved = _pull_compact(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    kicked = _kick_and_pull(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in kicked)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = kicked
                    if best_scope <= lower_bound:
                        return best_solution
                    break
            attempt += 1

        if not found and target_scope <= lower_bound + 3:
            break

    # Final compaction
    best_solution = _pull_compact(best_solution, n_rows, k, deadline)
    best_solution = _kick_and_pull(best_solution, n_rows, k, deadline)

    return best_solution
=======
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # Theoretical lower bound: need n_rows * k*(k+1)/2 distinct positive differences
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1  # differences are 1..M, so max entry >= num_diffs - 1

    k_plus_1 = k + 1
    rng = random.Random(12345)

    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. Multi-row iterative compaction (new structural approach)
    best_solution = _pull_all_rows(best_solution, n_rows, k, deadline, rng)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 3. Kick-and-pull with 2-element swaps and multi-row re-pull
    best_solution = _kick_and_pull(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 4. Randomized backtracking for tighter target scopes
    attempt = 0
    while time.time() < deadline - 5.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break

        found = False
        for _ in range(40):
            if time.time() > deadline - 5.0:
                break

            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # Multi-row compaction
                    improved = _pull_all_rows(best_solution, n_rows, k, deadline, rng)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    # Kick-and-pull
                    kicked = _kick_and_pull(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in kicked)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = kicked
                    if best_scope <= lower_bound:
                        return best_solution
                    break
            attempt += 1

        if not found and target_scope <= lower_bound + 3:
            break

    # Final multi-row compaction
    best_solution = _pull_all_rows(best_solution, n_rows, k, deadline, rng)
    best_solution = _kick_and_pull(best_solution, n_rows, k, deadline)

    return best_solution
>>>>>>> REPLACE
