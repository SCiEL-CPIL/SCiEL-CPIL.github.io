<<<<<<< SEARCH
    # =====================================================================
    # Phase 4: Hill climb local search
    # =====================================================================
    def hill_climb(exps, coefs, time_budget=2.0):
        nonlocal best_count, best_coeffs
        t0 = time.time()
        exps = list(exps)
        coefs = list(coefs)
        count = _square_nonzero_count_sparse(exps, coefs)
        try_update(count, exps, coefs)
        no_improve_count = 0
        vals = [-4, -3, -2, -1, 1, 2, 3, 4, -8, 8, -16, 16]

        while time.time() - t0 < time_budget:
            improved = False
            for _ in range(5):
                idx = random.randrange(12)
                e = exps[idx]
                c = coefs[idx]
                if random.random() < 0.7:
                    new_c = random.choice(vals)
                    if new_c != c and abs(new_c) <= 64:
                        coefs2 = list(coefs)
                        coefs2[idx] = new_c
                        if _count_nonzero(coefs2) != 12:
                            continue
                        c2 = _square_nonzero_count_sparse(exps, coefs2)
                        if c2 < count:
                            coefs = coefs2
                            count = c2
                            try_update(count, exps, coefs)
                            improved = True
                            no_improve_count = 0
                            break
                else:
                    shift = random.choice([-2, -1, 1, 2])
                    new_e = e + shift
                    if 0 <= new_e <= 40:
                        exps2 = list(exps)
                        exps2.remove(e)
                        if new_e not in exps2:
                            exps2.append(new_e)
                            exps2.sort()
                            c2 = _square_nonzero_count_sparse(exps2, coefs)
                            if c2 < count:
                                exps = exps2
                                count = c2
                                try_update(count, exps, coefs)
                                improved = True
                                no_improve_count = 0
                                break
            if not improved:
                no_improve_count += 1
                if no_improve_count > 30:
                    for _ in range(2):
                        idx = random.randrange(12)
                        new_c = random.choice([-8, -4, -2, -1, 1, 2, 4, 8])
                        if abs(new_c) <= 64:
                            coefs[idx] = new_c
                    no_improve_count = 0
                    count = _square_nonzero_count_sparse(exps, coefs)
                    try_update(count, exps, coefs)
            if best_count == 12:
                break
        return exps, coefs, count

    # Start hill climb from best found so far
    if best_coeffs is not None:
        exps, coefs = to_sparse_list(best_coeffs)
        if len(exps) == 12:
            hill_climb(exps, coefs, time_budget=5.0)
            if best_count == 12:
                return best_coeffs

    # Random starts
    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        if random.random() < 0.5:
            exps = list(range(12))
        else:
            exps = sorted(random.sample(range(40), 12))
        coefs = [random.choice(vals) for _ in range(12)]
        for i in range(12):
            if coefs[i] == 0:
                coefs[i] = 1
        hill_climb(exps, coefs, time_budget=1.0)
        if best_count == 12:
            return best_coeffs
        if attempts % 10 == 0:
            base_exp = random.randint(1, 3)
            exps = [i * base_exp for i in range(12)]
            if exps[-1] > 40:
                continue
            coefs = [random.choice(vals) for _ in range(12)]
            hill_climb(exps, coefs, time_budget=0.5)
            if best_count == 12:
                return best_coeffs
=======
    # =====================================================================
    # Phase 4: Hill climb local search (optimized)
    # =====================================================================
    def hill_climb(exps, coefs, time_budget=2.0):
        nonlocal best_count, best_coeffs
        t0 = time.time()
        exps = list(exps)
        coefs = list(coefs)
        count = _square_nonzero_count_sparse(exps, coefs)
        try_update(count, exps, coefs)
        no_improve_count = 0
        vals = [-4, -3, -2, -1, 1, 2, 3, 4, -8, 8, -16, 16]

        while time.time() - t0 < time_budget:
            improved = False
            for _ in range(8):
                idx = random.randrange(12)
                e = exps[idx]
                c = coefs[idx]
                if random.random() < 0.7:
                    new_c = random.choice(vals)
                    if new_c != c and abs(new_c) <= 64:
                        coefs2 = list(coefs)
                        coefs2[idx] = new_c
                        if _count_nonzero(coefs2) != 12:
                            continue
                        c2 = _square_nonzero_count_sparse(exps, coefs2)
                        if c2 < count:
                            coefs = coefs2
                            count = c2
                            try_update(count, exps, coefs)
                            improved = True
                            no_improve_count = 0
                            break
                else:
                    shift = random.choice([-2, -1, 1, 2])
                    new_e = e + shift
                    if 0 <= new_e <= 40:
                        exps2 = list(exps)
                        exps2.remove(e)
                        if new_e not in exps2:
                            exps2.append(new_e)
                            exps2.sort()
                            c2 = _square_nonzero_count_sparse(exps2, coefs)
                            if c2 < count:
                                exps = exps2
                                count = c2
                                try_update(count, exps, coefs)
                                improved = True
                                no_improve_count = 0
                                break
            if not improved:
                no_improve_count += 1
                if no_improve_count > 30:
                    for _ in range(2):
                        idx = random.randrange(12)
                        new_c = random.choice([-8, -4, -2, -1, 1, 2, 4, 8])
                        if abs(new_c) <= 64:
                            coefs[idx] = new_c
                    no_improve_count = 0
                    count = _square_nonzero_count_sparse(exps, coefs)
                    try_update(count, exps, coefs)
            if best_count == 12:
                break
        return exps, coefs, count

    # Start hill climb from best found so far
    if best_coeffs is not None:
        exps, coefs = to_sparse_list(best_coeffs)
        if len(exps) == 12:
            hill_climb(exps, coefs, time_budget=5.0)
            if best_count == 12:
                return best_coeffs

    # Random starts
    vals = [-4, -3, -2, -1, 1, 2, 3, 4]
    attempts = 0
    while time.time() < deadline - 15:
        attempts += 1
        if random.random() < 0.5:
            exps = list(range(12))
        else:
            exps = sorted(random.sample(range(40), 12))
        coefs = [random.choice(vals) for _ in range(12)]
        for i in range(12):
            if coefs[i] == 0:
                coefs[i] = 1
        hill_climb(exps, coefs, time_budget=1.0)
        if best_count == 12:
            return best_coeffs
        if attempts % 10 == 0:
            base_exp = random.randint(1, 3)
            exps = [i * base_exp for i in range(12)]
            if exps[-1] > 40:
                continue
            coefs = [random.choice(vals) for _ in range(12)]
            hill_climb(exps, coefs, time_budget=0.5)
            if best_count == 12:
                return best_coeffs
>>>>>>> REPLACE

