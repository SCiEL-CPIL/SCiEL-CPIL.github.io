<<<<<<< SEARCH
    # Initial polish
    if time_left() > 15:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Iterated Local Search
    while time_left() > 30:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

        # 2. Kick
        if time_left() < 25:
            break

        kicked_rows = kick(best_rows)

        # 3. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows)

        # 4. Update best
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Final high-precision polish
    if time_left() > 10:
        polished = final_high_precision_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
=======
    # Initial polish
    if time_left() > 15:
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Iterated Local Search
    while time_left() > 30:
        # 1. Local Search
        polished = local_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

        # 2. Kick
        if time_left() < 25:
            break

        kicked_rows = kick(best_rows)

        # 3. Local Search on kicked rows
        polished_kicked = local_polish(kicked_rows)

        # 4. Update best
        if scope(polished_kicked) < best_scope:
            best_rows = polished_kicked
            best_scope = scope(polished_kicked)

    # Final high-precision polish
    if time_left() > 10:
        polished = final_high_precision_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
>>>>>>> REPLACE
