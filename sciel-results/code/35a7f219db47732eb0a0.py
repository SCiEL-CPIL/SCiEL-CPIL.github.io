def _nj(D, labels):
    n = len(labels)
    if n == 2:
        return "(%s,%s);" % (labels[0], labels[1])
    if n == 3:
        return "((%s,%s),%s);" % (labels[0], labels[1], labels[2])

    D = D.astype(np.float64, copy=True)
    trees = {i: labels[i] for i in range(n)}
    alive = list(range(n))

    while len(alive) > 3:
        if not _time_ok():
            break
        m = len(alive)
        r = np.array(alive, dtype=np.int64)
        sub = D[np.ix_(r, r)]
        rowsum = sub.sum(axis=1)

        # Standard NJ Q criterion
        Q = (m - 2) * sub - rowsum[:, None] - rowsum[None, :]
        np.fill_diagonal(Q, np.inf)

        idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = int(idx[0]), int(idx[1])
        if i == j:
            break
        a, b = r[i], r[j]

        # Branch lengths (optional for topology, but good for distance updates)
        d_ab = D[a, b]
        da = rowsum[i]
        db = rowsum[j]
        la = 0.5 * d_ab + (da - db) / (2.0 * (m - 2))
        lb = 0.5 * d_ab - (da - db) / (2.0 * (m - 2))
        la = max(la, 0.0)
        lb = max(lb, 0.0)

        new_idx = len(alive)
        joined = "(%s,%s);" % (trees[a], trees[b])
        trees[new_idx] = joined

        rest = [x for x in r if x != a and x != b]
        D[new_idx, new_idx] = 0.0
        for k, x in enumerate(rest):
            val = 0.5 * (D[a, x] + D[b, x] - d_ab)
            D[new_idx, x] = val
            D[x, new_idx] = val

        alive = rest + [new_idx]

    if len(alive) == 3:
        return "((%s,%s),%s);" % (trees[alive[0]], trees[alive[1]], trees[alive[2]])
    return "(" + ",".join(trees[x] for x in alive) + ");"
