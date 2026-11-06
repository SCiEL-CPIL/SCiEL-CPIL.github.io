import time

def construct():
    t0 = time.time()
    DEADLINE = 80.0

    # Generate smooth numbers
    smooth_set = set()
    for a in range(0, 8):
        for b in range(0, 6):
            for c in range(0, 5):
                v = (2 ** a) * (3 ** b) * (5 ** c)
                if 1 <= v <= 128:
                    smooth_set.add(v)
    smooth = sorted(smooth_set)
    n = len(smooth)
    val_to_idx = {v: i for i, v in enumerate(smooth)}

    # Divisibility conflict
    div_conflict = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if smooth[j] % smooth[i] == 0:
                div_conflict[i][j] = True
                div_conflict[j][i] = True

    # Precompute compatible successors: compat[i] = sorted list of j > i with no conflict
    compat = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not div_conflict[i][j]:
                compat[i].append(j)

    # best_for_sum[s] = (count, tuple_of_indices)
    best_for_sum = [None] * 129

    # Size 1
    for i in range(n):
        s = smooth[i]
        if s <= 128:
            best_for_sum[s] = (1, (i,))

    # Size 2
    for i in range(n):
        si = smooth[i]
        if si >= 128:
            break
        for j in compat[i]:
            sj = smooth[j]
            s = si + sj
            if s > 128:
                break
            if best_for_sum[s] is None or best_for_sum[s][0] > 2:
                best_for_sum[s] = (2, (i, j))

    # Size 3
    for i in range(n):
        si = smooth[i]
        if si >= 128:
            break
        for j in compat[i]:
            sj = smooth[j]
            s2 = si + sj
            if s2 >= 128:
                break
            for k in compat[j]:
                sk = smooth[k]
                s3 = s2 + sk
                if s3 > 128:
                    break
                if div_conflict[i][k]:
                    continue
                if best_for_sum[s3] is None or best_for_sum[s3][0] > 3:
                    best_for_sum[s3] = (3, (i, j, k))

    # Size 4
    for i in range(n):
        si = smooth[i]
        if si >= 128:
            break
        for j in compat[i]:
            sj = smooth[j]
            s2 = si + sj
            if s2 >= 128:
                break
            for k in compat[j]:
                sk = smooth[k]
                s3 = s2 + sk
                if s3 >= 128:
                    break
                if div_conflict[i][k]:
                    continue
                for l in compat[k]:
                    sl = smooth[l]
                    s4 = s3 + sl
                    if s4 > 128:
                        break
                    if div_conflict[i][l] or div_conflict[j][l]:
                        continue
                    if best_for_sum[s4] is None or best_for_sum[s4][0] > 4:
                        best_for_sum[s4] = (4, (i, j, k, l))

    # Size 5
    for i in range(n):
        si = smooth[i]
        if si >= 128:
            break
        for j in compat[i]:
            sj = smooth[j]
            s2 = si + sj
            if s2 >= 128:
                break
            for k in compat[j]:
                sk = smooth[k]
                s3 = s2 + sk
                if s3 >= 128:
                    break
                if div_conflict[i][k]:
                    continue
                for l in compat[k]:
                    sl = smooth[l]
                    s4 = s3 + sl
                    if s4 >= 128:
                        break
                    if div_conflict[i][l] or div_conflict[j][l]:
                        continue
                    for m in compat[l]:
                        sm = smooth[m]
                        s5 = s4 + sm
                        if s5 > 128:
                            break
                        if div_conflict[i][m] or div_conflict[j][m] or div_conflict[k][m]:
                            continue
                        if best_for_sum[s5] is None or best_for_sum[s5][0] > 5:
                            best_for_sum[s5] = (5, (i, j, k, l, m))

    # Check uncovered
    uncovered = [t for t in range(1, 129) if best_for_sum[t] is None]

    if uncovered:
        for target in uncovered:
            if time.time() - t0 > DEADLINE:
                break
            found = False
            # Size 6
            for i in range(n):
                si = smooth[i]
                if si >= target:
                    break
                for j in compat[i]:
                    sj = smooth[j]
                    s2 = si + sj
                    if s2 >= target:
                        break
                    for k in compat[j]:
                        sk = smooth[k]
                        s3 = s2 + sk
                        if s3 >= target:
                            break
                        if div_conflict[i][k]:
                            continue
                        for l in compat[k]:
                            sl = smooth[l]
                            s4 = s3 + sl
                            if s4 >= target:
                                break
                            if div_conflict[i][l] or div_conflict[j][l]:
                                continue
                            for m in compat[l]:
                                sm = smooth[m]
                                s5 = s4 + sm
                                if s5 >= target:
                                    break
                                if div_conflict[i][m] or div_conflict[j][m] or div_conflict[k][m]:
                                    continue
                                rem = target - s5
                                if rem <= sm:
                                    break
                                if rem in smooth_set:
                                    idx = val_to_idx[rem]
                                    if idx > m and not (div_conflict[i][idx] or div_conflict[j][idx] or div_conflict[k][idx] or div_conflict[l][idx] or div_conflict[m][idx]):
                                        best_for_sum[target] = (6, (i, j, k, l, m, idx))
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                continue
            # Size 7
            for i in range(n):
                si = smooth[i]
                if si >= target:
                    break
                for j in compat[i]:
                    sj = smooth[j]
                    s2 = si + sj
                    if s2 >= target:
                        break
                    for k in compat[j]:
                        sk = smooth[k]
                        s3 = s2 + sk
                        if s3 >= target:
                            break
                        if div_conflict[i][k]:
                            continue
                        for l in compat[k]:
                            sl = smooth[l]
                            s4 = s3 + sl
                            if s4 >= target:
                                break
                            if div_conflict[i][l] or div_conflict[j][l]:
                                continue
                            for m in compat[l]:
                                sm = smooth[m]
                                s5 = s4 + sm
                                if s5 >= target:
                                    break
                                if div_conflict[i][m] or div_conflict[j][m] or div_conflict[k][m]:
                                    continue
                                for p in compat[m]:
                                    sp = smooth[p]
                                    s6 = s5 + sp
                                    if s6 >= target:
                                        break
                                    if div_conflict[i][p] or div_conflict[j][p] or div_conflict[k][p] or div_conflict[l][p]:
                                        continue
                                    rem = target - s6
                                    if rem <= sp:
                                        break
                                    if rem in smooth_set:
                                        idx = val_to_idx[rem]
                                        if idx > p and not (div_conflict[i][idx] or div_conflict[j][idx] or div_conflict[k][idx] or div_conflict[l][idx] or div_conflict[m][idx] or div_conflict[p][idx]):
                                            best_for_sum[target] = (7, (i, j, k, l, m, p, idx))
                                            found = True
                                            break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break

    # Build answers
    answers = []
    for t in range(1, 129):
        if best_for_sum[t] is not None:
            count, indices = best_for_sum[t]
            row = sorted(smooth[i] for i in indices)
            answers.append(row)
        else:
            if t in smooth_set:
                answers.append([t])
            else:
                answers.append([1, t - 1])

    return answers
