<<<<<<< SEARCH
    def local_search(S, rng, rounds):
        S = list(S)
        Sset = set(S)
        improved = True
        while improved and rounds > 0:
            improved = False
            rounds -= 1
            # try additions
            for x in all_candidates:
                if x in Sset:
                    continue
                if not can_add(x, S):
                    continue
                S.append(x)
                Sset.add(x)
                improved = True
                break
            if improved:
                continue
            # try swaps: remove a, add x
            # pick a random element to remove
            for _ in range(6):
                a = rng.choice(S)
                S2 = [v for v in S if v != a]
                S2set = set(S2)
                # find a good x to add
                cands = [x for x in all_candidates if x not in S2set]
                rng.shuffle(cands)
                found = False
                for x in cands:
                    if not can_add(x, S2):
                        continue
                    # only accept if it doesn't reduce size (it won't, we removed 1 added 1)
                    # accept swap to diversify
                    S2.append(x)
                    S2set.add(x)
                    S = S2
                    Sset = S2set
                    improved = True
                    found = True
                    break
                if found:
                    break
        return S
=======
    def local_search(S, rng, rounds):
        S = list(S)
        Sset = set(S)
        improved = True
        while improved and rounds > 0:
            improved = False
            rounds -= 1
            # try additions
            for x in all_candidates:
                if x in Sset:
                    continue
                if not can_add(x, S):
                    continue
                S.append(x)
                Sset.add(x)
                improved = True
                break
            if improved:
                continue
            # try 1-swap: remove a, add x
            for _ in range(8):
                a = rng.choice(S)
                S2 = [v for v in S if v != a]
                S2set = set(S2)
                cands = [x for x in all_candidates if x not in S2set]
                rng.shuffle(cands)
                found = False
                for x in cands:
                    if not can_add(x, S2):
                        continue
                    S2.append(x)
                    S2set.add(x)
                    S = S2
                    Sset = S2set
                    improved = True
                    found = True
                    break
                if found:
                    break
            if improved:
                continue
            # try 2-swap: remove a,b, add x,y
            if len(S) >= 2:
                for _ in range(4):
                    a, b = rng.sample(S, 2)
                    S2 = [v for v in S if v != a and v != b]
                    S2set = set(S2)
                    cands = [x for x in all_candidates if x not in S2set]
                    rng.shuffle(cands)
                    # try to find two elements to add
                    added = []
                    for x in cands:
                        if len(added) == 2:
                            break
                        if not can_add(x, S2 + added):
                            continue
                        added.append(x)
                    if len(added) == 2:
                        S = S2 + added
                        Sset = set(S)
                        improved = True
                        break
        return S
>>>>>>> REPLACE

