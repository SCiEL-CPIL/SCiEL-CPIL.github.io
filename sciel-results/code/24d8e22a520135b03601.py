<<<<<<< SEARCH
    # Main search loop
    best_result = None
    best_score = float('inf')
    rng = random.Random(42)

    # Deterministic greedy first
    det_rows = greedy_initial(n_rows, k, rng, perturb=0)
    if det_rows:
        valid, max_entry = validate(det_rows)
        if valid:
            best_result = det_rows
            best_score = max_entry
            improved, improved_score = simulated_annealing(det_rows, n_rows, k)
            if improved_score < best_score:
                best_score = improved_score
                best_result = improved

    # Randomized restarts with SA
    max_restarts = 80
    for restart in range(max_restarts):
        if time.time() > deadline - 15:
            break

        perturb = rng.randint(0, 6)
        rows = greedy_initial(n_rows, k, rng, perturb=perturb)
        if not rows:
            continue

        valid, max_entry = validate(rows)
        if not valid:
            continue

        if max_entry < best_score:
            best_score = max_entry
            best_result = rows
            improved, improved_score = simulated_annealing(rows, n_rows, k)
            if improved_score < best_score:
                best_score = improved_score
                best_result = improved
        else:
            # Even if not better than best, SA might find a better path
            # Only do SA if we have time
            if time.time() < deadline - 30 and max_entry <= best_score + 5:
                improved, improved_score = simulated_annealing(rows, n_rows, k, max_iters=15000)
                if improved_score < best_score:
                    best_score = improved_score
                    best_result = improved

    if best_result is None:
        # Fallback: simple greedy
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    return best_result
=======
    # Main search loop
    best_result = None
    best_score = float('inf')
    rng = random.Random(42)

    # Deterministic greedy first
    det_rows = greedy_initial(n_rows, k, rng, perturb=0)
    if det_rows:
        valid, max_entry = validate(det_rows)
        if valid:
            best_result = det_rows
            best_score = max_entry
            improved, improved_score = simulated_annealing(det_rows, n_rows, k)
            if improved_score < best_score:
                best_score = improved_score
                best_result = improved

    # Randomized restarts with SA
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break

        perturb = rng.randint(0, 8)
        rows = greedy_initial(n_rows, k, rng, perturb=perturb)
        if not rows:
            continue

        valid, max_entry = validate(rows)
        if not valid:
            continue

        if max_entry < best_score:
            best_score = max_entry
            best_result = rows
            # Use more iterations for promising solutions
            iters = 40000 if max_entry < best_score - 2 else 30000
            improved, improved_score = simulated_annealing(rows, n_rows, k, max_iters=iters)
            if improved_score < best_score:
                best_score = improved_score
                best_result = improved
        else:
            # Even if not better than best, SA might find a better path
            # Only do SA if we have time and the solution is close
            if time.time() < deadline - 35 and max_entry <= best_score + 8:
                improved, improved_score = simulated_annealing(rows, n_rows, k, max_iters=20000)
                if improved_score < best_score:
                    best_score = improved_score
                    best_result = improved

    # Final polish: if we have time left, do one more SA pass on the best result
    if best_result is not None and time.time() < deadline - 25:
        final_improved, final_score = simulated_annealing(best_result, n_rows, k, max_iters=50000)
        if final_score < best_score:
            best_score = final_score
            best_result = final_improved

    if best_result is None:
        # Fallback: simple greedy
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    return best_result
>>>>>>> REPLACE
