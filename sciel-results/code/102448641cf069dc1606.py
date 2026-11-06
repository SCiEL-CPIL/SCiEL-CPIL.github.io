<<<<<<< SEARCH
    # ---- Main optimization ----

    best_solution = []

    # 1. Single-prime constructions
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        if time.time() > deadline:
            break
        sol = construct_multiples_of(p)
        if len(sol) > len(best_solution):
            best_solution = sol

    # 2. Greedy with various orderings
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    orders = [
        list(range(2, 129)),
        list(range(128, 1, -1)),
        sorted(moduli, key=lambda x: -num_divisors(x)),
        sorted(moduli, key=lambda x: x.bit_count(), reverse=True),
        sorted(moduli, key=lambda x: -x.bit_count()),
    ]

    for order in orders:
        if time.time() > deadline:
            break
        sol = construct_greedy_order(order, deterministic=True)
        if len(sol) > len(best_solution):
            best_solution = sol

    # 3. Randomized restarts with greedy
    for _ in range(200):
        if time.time() > deadline:
            break
        random_order = moduli[:]
        random.shuffle(random_order)
        sol = construct_greedy_order(random_order, deterministic=False)
        if len(sol) > len(best_solution):
            best_solution = sol

    # 4. Local search from best solution
    best_solution = local_search(best_solution, max_rounds=100)

    # 5. More randomized restarts with local search
    for _ in range(100):
        if time.time() > deadline:
            break
        random_order = moduli[:]
        random.shuffle(random_order)
        sol = construct_greedy_order(random_order, deterministic=False)
        sol = local_search(sol, max_rounds=30)
        if len(sol) > len(best_solution):
            best_solution = sol

    # 6. Try to further improve with more aggressive local search
    if time.time() < deadline - 5.0:
        best_solution = local_search(best_solution, max_rounds=200)

    # Final safety: ensure non-empty
    if not best_solution:
        best_solution = [[0, 2]]

    return [[r, m] for r, m in best_solution]
=======
    # ---- Main optimization ----

    best_solution = []

    # 1. Single-prime constructions
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        if time.time() > deadline:
            break
        sol = construct_multiples_of(p)
        if len(sol) > len(best_solution):
            best_solution = sol

    # 2. Greedy with various orderings
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    orders = [
        list(range(2, 129)),
        list(range(128, 1, -1)),
        sorted(moduli, key=lambda x: -num_divisors(x)),
        sorted(moduli, key=lambda x: x.bit_count(), reverse=True),
        sorted(moduli, key=lambda x: -x.bit_count()),
    ]

    for order in orders:
        if time.time() > deadline:
            break
        sol = construct_greedy_order(order, deterministic=True)
        if len(sol) > len(best_solution):
            best_solution = sol

    # 3. CSP-based approach: select moduli and use backtracking with AC-3
    # Strategy: try to include all moduli, use CSP to find valid assignments
    # If that fails, try subsets
    
    # First, try the full set with CSP
    if time.time() < deadline - 10.0:
        # Use greedy to get an initial selection, then CSP to optimize
        for order in orders[:3]:
            if time.time() > deadline - 10.0:
                break
            sol = construct_greedy_order(order, deterministic=True)
            if len(sol) > len(best_solution):
                best_solution = sol
        
        # CSP on remaining moduli
        if best_solution and time.time() < deadline - 10.0:
            selected_set = set(m for _, m in best_solution)
            remaining = [m for m in moduli if m not in selected_set]
            if remaining:
                # Try to add more using CSP
                domains = ac3_prune(best_solution)
                if domains:
                    # Filter remaining to those with non-empty domains
                    valid_remaining = [m for m in remaining if m in domains and domains[m]]
                    if valid_remaining:
                        # Use backtracking to find best assignment
                        csp_result = csp_backtrack(best_solution, domains, valid_remaining)
                        if len(csp_result) > len(best_solution):
                            best_solution = csp_result

    # 4. Local search from best solution
    best_solution = local_search(best_solution, max_rounds=100)

    # 5. More randomized restarts with local search
    for _ in range(100):
        if time.time() > deadline:
            break
        random_order = moduli[:]
        random.shuffle(random_order)
        sol = construct_greedy_order(random_order, deterministic=False)
        sol = local_search(sol, max_rounds=30)
        if len(sol) > len(best_solution):
            best_solution = sol

    # 6. Try to further improve with more aggressive local search
    if time.time() < deadline - 5.0:
        best_solution = local_search(best_solution, max_rounds=200)

    # Final safety: ensure non-empty
    if not best_solution:
        best_solution = [[0, 2]]

    return [[r, m] for r, m in best_solution]
>>>>>>> REPLACE
