import time
import random

def construct():
    start = time.time()
    deadline = start + 55.0
    n = 12
    m = 40
    total = 1 << n

    # Precompute comparability: comp[i] is a list of masks j where i and j are comparable
    comp = [None] * total
    for i in range(total):
        ci = []
        for j in range(total):
            if i != j and ((i & j) == i or (i & j) == j):
                ci.append(j)
        comp[i] = ci

    # Build a strong initial solution
    initial = []
    initial.append(0)  # empty
    for i in range(n):
        initial.append(1 << i)  # 12 singletons
    for i in range(1, n):
        initial.append((1 << 0) | (1 << i))  # 11 pairs with 0
    for i in range(2, n):
        initial.append((1 << 0) | (1 << 1) | (1 << i))  # 10 triples with {0,1}
    # That's 1+12+11+10 = 34
    initial.append((1 << 0) | (1 << 1) | (1 << 2) | (1 << 3))
    initial.append((1 << 0) | (1 << 1) | (1 << 2) | (1 << 4))
    initial.append((1 << 0) | (1 << 1) | (1 << 3) | (1 << 4))
    initial.append((1 << 0) | (1 << 2) | (1 << 3) | (1 << 4))
    initial.append((1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4))
    initial.append((1 << n) - 1)  # full set

    seen = set()
    selected = []
    for mask in initial:
        if mask not in seen:
            seen.add(mask)
            selected.append(mask)
    if len(selected) < m:
        for i in range(1, n):
            for j in range(i + 1, n):
                if (1 << i) | (1 << j) not in seen:
                    seen.add((1 << i) | (1 << j))
                    selected.append((1 << i) | (1 << j))
                    if len(selected) == m:
                        break
            if len(selected) == m:
                break
    selected = selected[:m]

    sel_set = set(selected)

    def score_of(sel):
        s = 0
        for x in sel:
            for y in comp[x]:
                if y in sel:
                    s += 1
        return s // 2

    current = score_of(selected)
    best = current
    best_sel = list(selected)

    # Maintain gain[m] = number of comparable elements of m currently in sel_set
    gain = [0] * total
    for mask in selected:
        for y in comp[mask]:
            if y in sel_set:
                gain[y] += 1

    def update_gain_add(mask):
        for y in comp[mask]:
            gain[y] += 1

    def update_gain_remove(mask):
        for y in comp[mask]:
            gain[y] -= 1

    # Local search with restarts
    max_restarts = 8
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
        improved = True
        no_improve_passes = 0
        while improved and time.time() < deadline:
            improved = False
            # 1-swap
            for idx in range(m):
                if time.time() >= deadline:
                    break
                old = selected[idx]
                loss = gain[old]  # number of comparable elements in sel_set
                # Find best replacement
                best_gain = -1
                best_new = -1
                for new in range(total):
                    if new in sel_set:
                        continue
                    # gain[new] counts comparables in sel_set including old
                    # We need gain excluding old
                    g = gain[new]
                    if old in comp[new]:
                        g -= 1
                    if g > best_gain:
                        best_gain = g
                        best_new = new
                if best_new >= 0 and best_gain > loss:
                    # Remove old, add best_new
                    update_gain_remove(old)
                    sel_set.remove(old)
                    selected[idx] = best_new
                    sel_set.add(best_new)
                    update_gain_add(best_new)
                    current += best_gain - loss
                    if current > best:
                        best = current
                        best_sel = list(selected)
                    improved = True

            # 2-swap: try swapping two elements
            if time.time() < deadline:
                # Limit 2-swap to reduce runtime
                num_2swap_pairs = min(200, m * (m - 1) // 2)
                pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
                random.shuffle(pairs)
                pairs = pairs[:num_2swap_pairs]
                for i1, i2 in pairs:
                    if time.time() >= deadline:
                        break
                    old1, old2 = selected[i1], selected[i2]
                    # Compute loss of removing both
                    loss = 0
                    for y in comp[old1]:
                        if y in sel_set and y != old1 and y != old2:
                            loss += 1
                    for y in comp[old2]:
                        if y in sel_set and y != old1 and y != old2:
                            loss += 1
                    if (old1 & old2) == old1 or (old1 & old2) == old2:
                        loss += 1

                    best_gain = -1
                    best_new1, best_new2 = -1, -1
                    # Get top candidates
                    candidates = []
                    for new in range(total):
                        if new in sel_set:
                            continue
                        g = gain[new]
                        # Adjust for old1 and old2 being removed
                        if old1 in comp[new]:
                            g -= 1
                        if old2 in comp[new]:
                            g -= 1
                        candidates.append((g, new))
                    candidates.sort(reverse=True)
                    candidates = candidates[:30]

                    for c1 in candidates:
                        if time.time() >= deadline:
                            break
                        g1, n1 = c1
                        for c2 in candidates:
                            if c2[1] == c1[1]:
                                continue
                            g2, n2 = c2
                            extra = 0
                            if (n1 & n2) == n1 or (n1 & n2) == n2:
                                extra = 1
                            total_gain = g1 + g2 + extra
                            if total_gain > best_gain:
                                best_gain = total_gain
                                best_new1, best_new2 = n1, n2
                    if best_new1 >= 0 and best_gain > loss:
                        update_gain_remove(old1)
                        update_gain_remove(old2)
                        sel_set.remove(old1)
                        sel_set.remove(old2)
                        selected[i1] = best_new1
                        selected[i2] = best_new2
                        sel_set.add(best_new1)
                        sel_set.add(best_new2)
                        update_gain_add(best_new1)
                        update_gain_add(best_new2)
                        current += best_gain - loss
                        if current > best:
                            best = current
                            best_sel = list(selected)
                        improved = True

            if not improved:
                no_improve_passes += 1
                if no_improve_passes >= 2:
                    break

        # If we didn't reach deadline, perturb for next restart
        if time.time() >= deadline:
            break
        # Perturbation: randomly swap 3-5 elements
        num_perturb = random.randint(3, 5)
        for _ in range(num_perturb):
            idx = random.randint(0, m - 1)
            old = selected[idx]
            # Find a random replacement not in sel_set
            new = random.randint(0, total - 1)
            attempts = 0
            while new in sel_set and attempts < 100:
                new = random.randint(0, total - 1)
                attempts += 1
            if new not in sel_set:
                update_gain_remove(old)
                sel_set.remove(old)
                selected[idx] = new
                sel_set.add(new)
                update_gain_add(new)
                current += gain[new] - (gain[old] + 1)  # approximate, recompute below
        # Recompute current score accurately after perturbation
        current = score_of(selected)

    # Convert to lists
    result = []
    for mask in best_sel:
        result.append([int(b) for b in range(n) if (mask >> b) & 1])
    return result
