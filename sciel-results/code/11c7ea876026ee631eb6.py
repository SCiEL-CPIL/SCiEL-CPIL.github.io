<<<<<<< SEARCH
    # Main Loop
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False

        # Phase 1: Peeling (aggressive removal)
        while peel_lemma():
            improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if time.time() > deadline - 10:
            break

        # Phase 2: Swaps
        shuffled_code = list(code)
        random.shuffle(shuffled_code)
        for c in shuffled_code:
            if time.time() > deadline - 10:
                break
            if c not in code_set:
                continue
            if try_swap(c):
                improved = True

        # Phase 3: More peeling after swaps
        while peel_lemma():
            improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if time.time() > deadline - 10:
            break

        # Phase 4: Kicks
        for k in [1, 2, 3]:
            kicks_tried = 0
            max_kicks = 20
            while kicks_tried < max_kicks and time.time() < deadline - 10:
                if kick(k) or kick_2swap(k):
                    improved = True
                    kicks_tried = 0
                else:
                    kicks_tried += 1

            # Peel after kicks
            while peel_lemma():
                improved = True

            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

    # Restore best
    if len(best_code) < len(code):
        code = best_code
        code_set = set(best_code)
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors_list[idx]] += 1

    result = [idx_to_tuple(cw) for cw in code]
    return result
=======
    # Main Loop
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False

        # Phase 1: Peeling (aggressive removal)
        while peel_lemma():
            improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if time.time() > deadline - 15:
            break

        # Phase 2: Single swaps
        shuffled_code = list(code)
        random.shuffle(shuffled_code)
        for c in shuffled_code:
            if time.time() > deadline - 15:
                break
            if c not in code_set:
                continue
            if try_swap(c):
                improved = True

        # Phase 3: 2-swaps (replace 2 codewords with 1 or 2)
        if time.time() < deadline - 15:
            shuffled_code = list(code)
            random.shuffle(shuffled_code)
            for i in range(len(shuffled_code) - 1):
                if time.time() > deadline - 15:
                    break
                c1 = shuffled_code[i]
                c2 = shuffled_code[i + 1]
                if c1 not in code_set or c2 not in code_set:
                    continue
                if c1 == c2:
                    continue
                if try_2swap(c1, c2):
                    improved = True
                    break  # Restart shuffle after a successful 2-swap

        # Phase 4: More peeling after swaps
        while peel_lemma():
            improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if time.time() > deadline - 15:
            break

        # Phase 5: Kicks
        for k in [1, 2, 3]:
            kicks_tried = 0
            max_kicks = 15
            while kicks_tried < max_kicks and time.time() < deadline - 15:
                if kick(k) or kick_2swap(k):
                    improved = True
                    kicks_tried = 0
                else:
                    kicks_tried += 1

            # Peel after kicks
            while peel_lemma():
                improved = True

            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

    # Final polish: repeated single-swap and peel until no improvement
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            progress = False
            while peel_lemma():
                progress = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
            shuffled_code = list(code)
            random.shuffle(shuffled_code)
            for c in shuffled_code:
                if time.time() > deadline - 5:
                    break
                if c not in code_set:
                    continue
                if try_swap(c):
                    progress = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
            if not progress:
                break

    # Restore best
    if len(best_code) < len(code):
        code = best_code
        code_set = set(best_code)
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors_list[idx]] += 1

    result = [idx_to_tuple(cw) for cw in code]
    return result
>>>>>>> REPLACE

