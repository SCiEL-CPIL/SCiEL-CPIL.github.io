import time
import random
import math

def construct():
    n = 12
    m = 40
    N = 1 << n
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute comparability: comp[a][b] = 1 if a⊆b or b⊆a
    comp = [[0] * N for _ in range(N)]
    for a in range(N):
        comp[a][a] = 1
        for b in range(a + 1, N):
            if (a & b) == a or (a & b) == b:
                comp[a][b] = 1
                comp[b][a] = 1

    def count_comparable(fam):
        c = 0
        for i in range(m):
            fi = fam[i]
            for j in range(i + 1, m):
                if comp[fi][fam[j]]:
                    c += 1
        return c

    def contrib(fam, idx):
        """Number of comparable pairs involving fam[idx] with other family members."""
        f = fam[idx]
        c = 0
        for j in range(m):
            if j != idx and comp[f][fam[j]]:
                c += 1
        return c

    def greedy_chain():
        """Build a long chain greedily: always add a set comparable to all existing."""
        family = [0]
        used = {0}
        # Try to extend chain by adding supersets/subsets
        while len(family) < m:
            best = None
            best_comp = -1
            for cand in range(N):
                if cand in used:
                    continue
                # Check comparable to all existing
                ok = all(comp[cand][f] for f in family)
                if ok:
                    # Prefer sets that allow further extension (more "room")
                    # Score by popcount distance from middle
                    pc = bin(cand).count('1')
                    score = -abs(pc - 6)
                    if score > best_comp:
                        best_comp = score
                        best = cand
            if best is None:
                break
            family.append(best)
            used.add(best)
        # Fill remaining with random comparable sets
        if len(family) < m:
            random.shuffle(list(range(N)))
            for cand in range(N):
                if len(family) >= m:
                    break
                if cand not in used:
                    family.append(cand)
                    used.add(cand)
        return family[:m]

    def local_search(family, max_iter, allow_2swap=True):
        """Local search with 1-swap and 2-swap moves, full candidate scan for worst element."""
        family = list(family)
        used = set(family)
        best_score = count_comparable(family)
        contribs = [contrib(family, i) for i in range(m)]

        for iteration in range(max_iter):
            if time.time() > deadline:
                break

            improved = False

            # Find the element with minimum contribution
            min_idx = min(range(m), key=lambda i: contribs[i])
            min_val = contribs[min_idx]

            # Try to replace min_idx with a better unused set (full scan)
            best_new = -1
            best_delta = 0
            old = family[min_idx]
            old_c = contribs[min_idx]

            for cand in range(N):
                if cand in used:
                    continue
                # Compute new contribution
                new_c = 0
                for j in range(m):
                    if j != min_idx and comp[cand][family[j]]:
                        new_c += 1
                delta = new_c - old_c
                if delta > best_delta:
                    best_delta = delta
                    best_new = cand

            if best_new >= 0 and best_delta > 0:
                used.remove(old)
                used.add(best_new)
                family[min_idx] = best_new
                # Update contributions
                old_c = contribs[min_idx]
                contribs[min_idx] = contribs[min_idx] + best_delta
                for j in range(m):
                    if j != min_idx:
                        if comp[best_new][family[j]] and not comp[old][family[j]]:
                            contribs[j] += 1
                        elif comp[old][family[j]] and not comp[best_new][family[j]]:
                            contribs[j] -= 1
                best_score += best_delta
                improved = True

            # 2-swap move: replace two elements simultaneously
            if allow_2swap and not improved and random.random() < 0.3:
                # Pick two elements with low contribution
                sorted_idx = sorted(range(m), key=lambda i: contribs[i])
                i1, i2 = sorted_idx[0], sorted_idx[1]
                old1, old2 = family[i1], family[i2]
                c1, c2 = contribs[i1], contribs[i2]
                # Contribution of pair (i1,i2) to each other
                pair_comp = comp[old1][old2]

                best_pair = None
                best_pair_delta = 0
                # Sample candidates for speed
                candidates = random.sample(range(N), min(200, N))
                for cand1 in candidates:
                    if cand1 in used or cand1 == old1:
                        continue
                    for cand2 in candidates:
                        if cand2 in used or cand2 == old2 or cand2 == cand1:
                            continue
                        # Compute delta
                        # Remove old1, old2 contributions (excluding each other)
                        # Add cand1, cand2 contributions
                        delta = 0
                        for j in range(m):
                            if j == i1 or j == i2:
                                continue
                            fj = family[j]
                            delta -= comp[old1][fj]
                            delta -= comp[old2][fj]
                            delta += comp[cand1][fj]
                            delta += comp[cand2][fj]
                        # Pair contribution change
                        delta -= pair_comp
                        delta += comp[cand1][cand2]

                        if delta > best_pair_delta:
                            best_pair_delta = delta
                            best_pair = (cand1, cand2)

                if best_pair and best_pair_delta > 0:
                    cand1, cand2 = best_pair
                    used.discard(old1)
                    used.discard(old2)
                    used.add(cand1)
                    used.add(cand2)
                    family[i1] = cand1
                    family[i2] = cand2
                    # Recompute contributions for affected
                    contribs[i1] = contrib(family, i1)
                    contribs[i2] = contrib(family, i2)
                    for j in range(m):
                        if j != i1 and j != i2:
                            # Recompute if needed
                            new_c = 0
                            for k in range(m):
                                if k != j and comp[family[j]][family[k]]:
                                    new_c += 1
                            contribs[j] = new_c
                    best_score += best_pair_delta
                    improved = True

            if not improved:
                # Perturb: randomly replace 1-2 elements
                for _ in range(random.randint(1, 2)):
                    idx = random.randint(0, m - 1)
                    old = family[idx]
                    new = random.randint(0, N - 1)
                    if new not in used:
                        used.discard(old)
                        used.add(new)
                        family[idx] = new
                        contribs[idx] = contrib(family, idx)
                        for j in range(m):
                            if j != idx:
                                new_c = 0
                                for k in range(m):
                                    if k != j and comp[family[j]][family[k]]:
                                        new_c += 1
                                contribs[j] = new_c
                        best_score = count_comparable(family)
                contribs = [contrib(family, i) for i in range(m)]
                best_score = count_comparable(family)

        return family, best_score

    # Multiple restarts with different strategies
    best_family = None
    best_score = -1

    for restart in range(20):
        if time.time() > deadline - 5:
            break

        random.seed(restart * 7919 + 1)

        if restart == 0:
            family = greedy_chain()
        elif restart < 5:
            # Greedy with different biases
            family = greedy_chain()
        else:
            # Random initialization
            family = random.sample(range(N), m)

        # Local search with budget based on remaining time
        remaining = deadline - time.time()
        max_iter = int(remaining * 50)
        max_iter = max(100, min(max_iter, 5000))

        family, score = local_search(family, max_iter)

        if score > best_score:
            best_score = score
            best_family = list(family)

    # Final exhaustive polish: full 4096 scan, more iterations
    if best_family is not None and time.time() < deadline - 3:
        remaining = deadline - time.time() - 3
        max_iter = int(remaining * 100)
        max_iter = max(200, min(max_iter, 20000))
        best_family, best_score = local_search(best_family, max_iter, allow_2swap=True)

    # Convert to list of lists
    result = [[b for b in range(n) if (mask >> b) & 1] for mask in best_family]

    # Verify
    assert len(result) == m
    assert len(set(tuple(sorted(r)) for r in result)) == m
    for r in result:
        assert len(r) == len(set(r))
        assert all(isinstance(x, int) and 0 <= x <= n - 1 for x in r)

    return result
