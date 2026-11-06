def construct():
    deadline = time.time() + 85
    rng = random.Random(42)
    best_coeffs = None
    best_nz = 100

    def try_candidate(c):
        nonlocal best_coeffs, best_nz
        c_trimmed = _trim(c)
        if not _is_valid_candidate(c_trimmed):
            return
        nz = _score_candidate(c_trimmed)
        if nz < best_nz:
            best_nz = nz
            best_coeffs = c_trimmed[:]

    # Random Search
    for i in range(100000):
        if time.time() > deadline:
            break
        degrees = sorted(rng.sample(range(41), 12))
        coeffs = [0] * 41
        for d in degrees:
            coeffs[d] = rng.choice([-2, -1, 1, 2])
        try_candidate(coeffs)

    if best_coeffs is None:
        best_coeffs = [1]*12
        best_nz = _score_candidate(best_coeffs)

    def local_search(c, steps=1000):
        nonlocal best_coeffs, best_nz
        # c is a list of length 41
        current = c[:]
        
        # Get initial score
        current_trimmed = _trim(current)
        if not _is_valid_candidate(current_trimmed):
            return
        current_nz = _score_candidate(current_trimmed)
        
        for s in range(steps):
            if time.time() > deadline:
                return
                
            non_zero_indices = [i for i, val in enumerate(current) if val != 0]
            if len(non_zero_indices) != 12:
                # Should not happen if we start valid and only swap/move within valid bounds
                # but reversion might break it? No, we revert.
                pass

            # Move 1: Change coefficient
            idx = rng.choice(non_zero_indices)
            old_val = current[idx]
            new_val = rng.choice([-2, -1, 1, 2])
            if new_val != old_val:
                current[idx] = new_val
                t = _trim(current)
                if _is_valid_candidate(t):
                    new_nz = _score_candidate(t)
                    if new_nz < current_nz:
                        current_nz = new_nz
                        if new_nz < best_nz:
                            best_nz = new_nz
                            best_coeffs = t[:]
                    else:
                        current[idx] = old_val
                else:
                    current[idx] = old_val

            # Move 2: Move term
            idx = rng.choice(non_zero_indices)
            old_deg = idx
            new_deg = rng.randint(0, 40)
            if new_deg != old_deg:
                if current[new_deg] != 0:
                    continue
                val = current[old_deg]
                current[old_deg] = 0
                current[new_deg] = val
                t = _trim(current)
                if _is_valid_candidate(t):
                    new_nz = _score_candidate(t)
                    if new_nz < current_nz:
                        current_nz = new_nz
                        if new_nz < best_nz:
                            best_nz = new_nz
                            best_coeffs = t[:]
                    else:
                        current[new_deg] = 0
                        current[old_deg] = val
                else:
                    current[new_deg] = 0
                    current[old_deg] = val

            # Move 3: Swap
            if len(non_zero_indices) >= 2:
                idx1, idx2 = rng.sample(non_zero_indices, 2)
                v1, v2 = current[idx1], current[idx2]
                current[idx1], current[idx2] = v2, v1
                t = _trim(current)
                if _is_valid_candidate(t):
                    new_nz = _score_candidate(t)
                    if new_nz < current_nz:
                        current_nz = new_nz
                        if new_nz < best_nz:
                            best_nz = new_nz
                            best_coeffs = t[:]
                    else:
                        current[idx1], current[idx2] = v1, v2
                else:
                    current[idx1], current[idx2] = v1, v2

    # Local search loop
    for i in range(50):
        if time.time() > deadline:
            break
        if best_coeffs:
            # Pad best_coeffs to length 41 for local search
            padded = best_coeffs + [0] * (41 - len(best_coeffs))
            local_search(padded, steps=500)
        
        if time.time() > deadline:
            break
        
        # New random start
        degrees = sorted(rng.sample(range(41), 12))
        coeffs = [0] * 41
        for d in degrees:
            coeffs[d] = rng.choice([-2, -1, 1, 2])
        local_search(coeffs, steps=500)

    return best_coeffs
