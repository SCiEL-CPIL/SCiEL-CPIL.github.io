<<<<<<< SEARCH
    # Main loop: try increasing n with hybrid search
    for target_n in range(10, 35):
        if time.time() > deadline - 10:
            break

        structs = build_structures(target_n)
        num_triples = structs['num_triples']

        found = False
        result_colors = None

        # Strategy 1: Hybrid constraint-guided search
        result_colors = hybrid_search(target_n, structs, deadline)
        if result_colors is not None:
            rk4, bk5, _, _ = compute_violations(result_colors, structs)
            if rk4 == 0 and bk5 == 0:
                found = True

        # Strategy 2: If hybrid failed, try pure constraint backtracking
        if not found and time.time() < deadline - 5:
            bt_result = constraint_backtrack(target_n, structs, deadline, max_nodes=30000)
            if bt_result is not None:
                result_colors = bt_result
                found = True

        # Strategy 3: If still failed, try more local search restarts
        if not found:
            max_restarts = 20
            for restart in range(max_restarts):
                if time.time() > deadline - 3:
                    break

                color_arr = generate_initial_coloring(target_n, restart, num_triples, structs['triple_list'])
                color_arr, rk4, bk5 = local_search(color_arr, structs, max_iters=50000, deadline=deadline)

                if rk4 == 0 and bk5 == 0:
                    result_colors = color_arr
                    found = True
                    break

        if found:
            colors_list = result_colors.tolist()
            if is_valid(target_n, colors_list):
                best_n = target_n
                best_colors = colors_list
            else:
                break
        else:
            if target_n >= 25:
                break

    return {'n': best_n, 'colors': best_colors}
=======
    # Main loop: try increasing n with hybrid search
    for target_n in range(10, 35):
        if time.time() > deadline - 10:
            break

        structs = build_structures(target_n)
        num_triples = structs['num_triples']

        found = False
        result_colors = None

        # Strategy 1: Hybrid constraint-guided search
        result_colors = hybrid_search(target_n, structs, deadline)
        if result_colors is not None:
            rk4, bk5, _, _ = compute_violations(result_colors, structs)
            if rk4 == 0 and bk5 == 0:
                found = True

        # Strategy 2: If hybrid failed, try pure constraint backtracking
        if not found and time.time() < deadline - 5:
            bt_result = constraint_backtrack(target_n, structs, deadline, max_nodes=30000)
            if bt_result is not None:
                result_colors = bt_result
                found = True

        # Strategy 3: If still failed, try more local search restarts
        if not found:
            max_restarts = 30
            for restart in range(max_restarts):
                if time.time() > deadline - 3:
                    break

                color_arr = generate_initial_coloring(target_n, restart, num_triples, structs['triple_list'])
                color_arr, rk4, bk5 = local_search(color_arr, structs, max_iters=80000, deadline=deadline)

                if rk4 == 0 and bk5 == 0:
                    result_colors = color_arr
                    found = True
                    break

        if found:
            colors_list = result_colors.tolist()
            if is_valid(target_n, colors_list):
                best_n = target_n
                best_colors = colors_list
            else:
                break
        else:
            if target_n >= 25:
                break

    return {'n': best_n, 'colors': best_colors}
>>>>>>> REPLACE
