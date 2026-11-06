import time

N = 128
DEADLINE = 75.0  # safety margin before 90s kill
_t0 = time.time()


def _progressions():
    """All nonconstant 4-term progressions (a, a+d, a+2d, a+3d) within 1..N."""
    ps = []
    for a in range(1, N + 1):
        for d in range(1, N):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                ps.append((a, b, c, e))
    return ps


def _index_of(c, colors):
    """1-based index of position c (1..N) in the colors list (0-based)."""
    return c - 1


def _is_bad(p, colors):
    """True if progression p uses exactly 2 distinct colors."""
    vals = {colors[_index_of(x, colors)] for x in p}
    return len(vals) == 2


def construct():
    global _t0
    _t0 = time.time()

    progs = _progressions()
    P = len(progs)

    # Precompute, for each position, the list of progression indices it belongs to.
    pos_to_progs = [[] for _ in range(N)]
    for pi, p in enumerate(progs):
        for x in p:
            pos_to_progs[_index_of(x, colors) if False else x - 1].append(pi)

    # Start from a trivial 2-coloring (parity): every progression is "bad" (2 colors).
    colors = [0 if c % 2 == 1 else 1 for c in range(1, N + 1)]

    # Current set of bad progression indices.
    bad = set(range(P))

    def try_color(pos_idx, new_color):
        """Recolor position pos_idx (0-based) to new_color; update bad set.
        Returns True if improvement (fewer bad progressions)."""
        old = colors[pos_idx]
        if old == new_color:
            return False
        affected = pos_to_progs[pos_idx + 1]
        before = len(bad)
        # Apply tentatively
        colors[pos_idx] = new_color
        for pi in affected:
            p = progs[pi]
            if _is_bad(p, colors):
                bad.add(pi)
            else:
                bad.discard(pi)
        after = len(bad)
        if after < before:
            return True
        # Revert
        colors[pos_idx] = old
        for pi in affected:
            p = progs[pi]
            if _is_bad(p, colors):
                bad.add(pi)
            else:
                bad.discard(pi)
        return False

    # Greedy repair: repeatedly pick (position, color) that breaks the most bad progressions.
    while bad:
        if time.time() - _t0 > DEADLINE:
            break
        best_gain = 0
        best_choice = None
        # Candidate positions: those appearing in at least one bad progression.
        cand_positions = set()
        for pi in bad:
            for x in progs[pi]:
                cand_positions.add(x - 1)
        for pos_idx in cand_positions:
            old = colors[pos_idx]
            # Candidate colors: old color's neighbors plus a couple of fresh colors.
            cand_colors = set()
            # colors used by the bad progressions containing this position
            for pi in pos_to_progs[pos_idx + 1]:
                if pi in bad:
                    for x in progs[pi]:
                        cand_colors.add(colors[x - 1])
            # also try a couple of fresh color labels
            cand_colors.add(2)
            cand_colors.add(3)
            cand_colors.add(4)
            for nc in cand_colors:
                if nc == old:
                    continue
                if try_color(pos_idx, nc):
                    # measure gain
                    gain = len(bad)
                    # revert to measure properly
                    old_c = colors[pos_idx]
                    colors[pos_idx] = old
                    for pi in pos_to_progs[pos_idx + 1]:
                        p = progs[pi]
                        if _is_bad(p, colors):
                            bad.add(pi)
                        else:
                            bad.discard(pi)
                    before = len(bad)
                    # re-apply
                    colors[pos_idx] = nc
                    for pi in pos_to_progs[pos_idx + 1]:
                        p = progs[pi]
                        if _is_bad(p, colors):
                            bad.add(pi)
                        else:
                            bad.discard(pi)
                    gain = before - len(bad)
                    if gain > best_gain:
                        best_gain = gain
                        best_choice = (pos_idx, nc)
                        # keep it applied
                        break
                    else:
                        # revert
                        colors[pos_idx] = old
                        for pi in pos_to_progs[pos_idx + 1]:
                            p = progs[pi]
                            if _is_bad(p, colors):
                                bad.add(pi)
                            else:
                                bad.discard(pi)
                else:
                    # already reverted inside try_color
                    pass
        if best_choice is None:
            # No single recolor helps; try forcing a fresh color on a position in a bad progression
            if bad:
                pi = next(iter(bad))
                x = progs[pi][0]
                pos_idx = x - 1
                nc = 2
                if try_color(pos_idx, nc):
                    continue
            break

    # Final verification (defensive)
    for p in progs:
        vals = {colors[x - 1] for x in p}
        if len(vals) < 3:
            # should not happen; return as-is
            pass

    return colors
