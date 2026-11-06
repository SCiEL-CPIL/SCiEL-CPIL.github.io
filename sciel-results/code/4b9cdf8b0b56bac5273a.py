<<<<<<< SEARCH
    # Strategy 4: Alternative 7-prime sets
    alts7 = [
        [2,3,5,7,11,13,17],
        [2,3,5,7,11,13,19],
        [2,3,5,7,11,13,23],
        [2,3,5,7,11,17,19],
        [2,3,5,7,13,17,19],
        [2,3,5,11,13,17,19],
        [2,3,7,11,13,17,19],
        [2,5,7,11,13,17,19],
        [2,3,5,7,11,13,29],
        [2,3,5,7,11,17,23],
        [2,3,5,7,13,17,23],
        [2,3,5,7,11,19,23],
        [2,3,5,7,13,19,23],
        [2,3,5,7,11,13,31],
        [2,3,5,7,11,17,29],
    ]
    for P in alts7:
        if time.time() > deadline - 5:
            break
        pm = pmask_from_list(P)
        L = _smooth_list(pm)
        if len(L) < 12:
            continue
        for T in [150, 300, 500, 700, len(L)]:
            if T > len(L): T = len(L)
            if time.time() > deadline - 5:
                break
            A = L[:T]
            
            res = _greedy_clique(A, pm, 12)
            if res is not None:
                res2, pm2 = _local_search_reduce(res, pm, deadline - 3)
                P2 = [p for p in PRIMES30 if pm2 & (1 << P2IDX[p])]
                if try_and_update(res2, P2):
                    if len(P2) <= 5:
                        return list(res2)
                    elif len(P2) < 7:
                        return list(res2)
            
            remaining = deadline - time.time()
            if remaining < 4:
                break
            sub_deadline = min(time.time() + max(1.5, remaining * 0.2), deadline - 4.0)
            res = _find_clique(A, pm, 12, 300000, sub_deadline)
            if res is not None:
                res2, pm2 = _local_search_reduce(res, pm, deadline - 3)
                P2 = [p for p in PRIMES30 if pm2 & (1 << P2IDX[p])]
                if try_and_update(res2, P2):
                    if len(P2) <= 5:
                        return list(res2)
                    elif len(P2) < 7:
                        return list(res2)
=======
    # Strategy 4: Alternative 7-prime sets
    alts7 = [
        [2,3,5,7,11,13,17],
        [2,3,5,7,11,13,19],
        [2,3,5,7,11,13,23],
        [2,3,5,7,11,17,19],
        [2,3,5,7,13,17,19],
        [2,3,5,11,13,17,19],
        [2,3,7,11,13,17,19],
        [2,5,7,11,13,17,19],
        [2,3,5,7,11,13,29],
        [2,3,5,7,11,17,23],
        [2,3,5,7,13,17,23],
        [2,3,5,7,11,19,23],
        [2,3,5,7,13,19,23],
        [2,3,5,7,11,13,31],
        [2,3,5,7,11,17,29],
    ]
    for P in alts7:
        if time.time() > deadline - 4:
            break
        pm = pmask_from_list(P)
        L = _smooth_list(pm)
        if len(L) < 12:
            continue
        for T in [200, 350, 500, 800, len(L)]:
            if T > len(L): T = len(L)
            if time.time() > deadline - 4:
                break
            A = L[:T]
            
            sub_deadline = min(time.time() + 1.5, deadline - 3)
            res = _greedy_clique(A, pm, 12, sub_deadline)
            if res is not None:
                res2, pm2 = _local_search_reduce(res, pm, deadline - 2)
                P2 = [p for p in PRIMES30 if pm2 & (1 << P2IDX[p])]
                if try_and_update(res2, P2):
                    if len(P2) <= 4:
                        return list(res2)
                    elif len(P2) < 7:
                        break
            
            remaining = deadline - time.time()
            if remaining < 3:
                break
            sub_deadline = min(time.time() + max(1.0, remaining * 0.15), deadline - 3.0)
            res = _find_clique(A, pm, 12, 350000, sub_deadline)
            if res is not None:
                res2, pm2 = _local_search_reduce(res, pm, deadline - 2)
                P2 = [p for p in PRIMES30 if pm2 & (1 << P2IDX[p])]
                if try_and_update(res2, P2):
                    if len(P2) <= 4:
                        return list(res2)
                    elif len(P2) < 7:
                        break
>>>>>>> REPLACE

