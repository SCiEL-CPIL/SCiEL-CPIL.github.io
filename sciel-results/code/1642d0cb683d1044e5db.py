"""Minimum maximal no-isosceles set (N=16)."""
import time

N = 16
NPTS = N * N
DEADLINE = 322.0  # wall-clock safety margin under the 330s kill


def _idx(p):
    return p[0] * N + p[1]


def _precompute():
    """Return conflict bitsets: conf[i] has bit j set iff (i,j) pair blocks something.

    Specifically, for a chosen pair (i, j), the set of grid points that form an
    isosceles triangle with i and j (so cannot be added) is stored.
    """
    coords = [(x, y) for x in range(N) for y in range(N)]
    # For each ordered pair (i, j), compute the set of points k such that
    # triangle (i, j, k) is isosceles.  We store as a Python int bitmask.
    conf = [0] * (NPTS * NPTS)
    for i in range(NPTS):
        xi, yi = coords[i]
        for j in range(NPTS):
            if i == j:
                continue
            xj, yj = coords[j]
            d2_ij = (xi - xj) ** 2 + (yi - yj) ** 2
            mask = 0
            for k in range(NPTS):
                if k == i or k == j:
                    continue
                xk, yk = coords[k]
                d2_ik = (xi - xk) ** 2 + (yi - yk) ** 2
                d2_jk = (xj - xk) ** 2 + (yj - yk) ** 2
                if d2_ik == d2_ij or d2_jk == d2_ij or d2_ik == d2_jk:
                    mask |= (1 << k)
            conf[i * NPTS + j] = mask
    return coords, conf


def _build_maximal(coords, conf, order):
    """Greedy build in given order; returns (chosen_indices, blocked_mask)."""
    chosen = []
    blocked = 0
    for p in order:
        if (1 << p) & blocked:
            continue
        # Adding p: must not form isosceles with any pair in chosen.
        ok = True
        for a in chosen:
            # check p with a and every other chosen b
            for b in chosen:
                if a == b:
                    continue
                # triangle (p, a, b) isosceles?
                # equivalently p is in conf[a*NPTS+b]
                if (conf[a * NPTS + b] >> p) & 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        # add p and update blocked
        for a in chosen:
            blocked |= conf[p * NPTS + a]
            blocked |= conf[a * NPTS + p]
        chosen.append(p)
    return chosen, blocked


def _is_maximal(chosen, blocked):
    """Check every unchosen point is blocked."""
    full = (1 << NPTS) - 1
    chosen_mask = 0
    for c in chosen:
        chosen_mask |= (1 << c)
    unchosen = full & ~chosen_mask
    return (uncchosen & ~blocked) == 0 if False else ((uncchosen & ~blocked) == 0)


def construct():
    t0 = time.time()
    coords, conf = _precompute()
    full = (1 << NPTS) - 1

    # Initial greedy in row-major order
    order = list(range(NPTS))
    chosen, blocked = _build_maximal(coords, conf, order)

    best = list(chosen)
    best_blocked = blocked
    best_size = len(best)

    # Removal pass: try to remove points while staying maximal and no-isosceles.
    def try_remove(chosen, blocked):
        """Attempt to remove each point; return new (chosen, blocked) if improved."""
        n = len(chosen)
        for ri in range(n):
            if time.time() > DEADLINE:
                break
            p = chosen[ri]
            # After removing p, newly unblocked points = points that were blocked
            # only by pairs involving p.  Compute points that become unblocked.
            # A point q (not in chosen) is blocked if any chosen pair (a,b) has q in conf.
            # After removing p, q is still blocked if some pair not involving p blocks it.
            # Simpler: recompute blocked without p, then check maximality.
            new_chosen = [c for c in chosen if c != p]
            # Recompute blocked from scratch (expensive but exact)
            new_blocked = 0
            for i in range(len(new_chosen)):
                a = new_chosen[i]
                for j in range(i + 1, len(new_chosen)):
                    b = new_chosen[j]
                    new_blocked |= conf[a * NPTS + b]
                    new_blocked |= conf[b * NPTS + a]
            # maximality check
            new_chosen_mask = 0
            for c in new_chosen:
                new_chosen_mask |= (1 << c)
            unchosen = full & ~new_chosen_mask
            if (uncchosen & ~new_blocked) == 0:
                return new_chosen, new_blocked
        return chosen, blocked

    # Iteratively remove
    progress = True
    while progress and time.time() < DEADLINE:
        progress = False
        new_chosen, new_blocked = try_remove(chosen, blocked)
        if len(new_chosen) < len(chosen):
            chosen, blocked = new_chosen, new_blocked
            progress = True
            if len(chosen) < best_size:
                best = list(chosen)
                best_blocked = blocked
                best_size = len(best)

    # Return as list of (x, y)
    result = [coords[c] for c in best]
    return result
