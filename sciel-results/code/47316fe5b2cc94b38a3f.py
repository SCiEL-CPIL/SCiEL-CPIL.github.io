import time
import math

_START = time.time()
_LIMIT = 55.0  # safety margin under the 90s wall-clock


def _valid_suffixes(seq, x, used):
    """Return the list of new consecutive-block sums ending at x, or None if any collides."""
    s = x
    out = [s]
    for a in reversed(seq):
        s += a
        if s in used:
            return None
        out.append(s)
    return out


def _room_slack(seq, used):
    last = seq[-1]
    room = 256 - last
    total = (256 * 257) // 2
    slack = (total - len(used)) / total
    return room, slack


def _geometric(seq, used, a=0.75, b=0.25):
    room, slack = _room_slack(seq, used)
    # geometric surrogate P = room^a * slack^b  (smooth, >0)
    return (room ** a) * (slack ** b)


def _logsumexp(vals):
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def _log_geometric(seq, used, a=0.75, b=0.25):
    room, slack = _room_slack(seq, used)
    return a * math.log(max(room, 1)) + b * math.log(max(slack, 1e-9))


def _beam_search(beam_size=400, a=0.75, b=0.25, max_steps=60):
    """Beam search over partial sequences. Returns the best (longest) sequence found."""
    # initial beam: each single value is a valid start
    beam = []
    for x in range(1, 257):
        beam.append((x, [x], {x}))
    best_seq = list(beam[0][0])
    best_len = 1

    for step in range(max_steps):
        if time.time() - _START > _LIMIT:
            break
        if not beam:
            break
        # extend each beam state
        new_beam = []
        for seq, suffix, used in beam:
            last = seq[-1]
            for x in range(last + 1, 257):
                ns = _valid_suffixes(seq, x, used)
                if ns is None:
                    continue
                new_seq = seq + [x]
                new_used = used | set(ns)
                lg = _log_geometric(new_seq, new_used, a, b)
                new_beam.append((lg, new_seq, new_used))
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq
        if not new_beam:
            break
        # smooth surrogate ranking: keep beam_size states with distinct sequences,
        # prioritized by a soft-max (log-sum-exp) style score.
        # To keep diversity, we rank by lg but cap how many share the same last value.
        new_beam.sort(key=lambda t: t[0], reverse=True)
        # diversity: limit per-last-value count
        seen_last = {}
        kept = []
        per_last_cap = max(3, beam_size // 40)
        for lg, seq, used in new_beam:
            last = seq[-1]
            if seen_last.get(last, 0) >= per_last_cap:
                continue
            seen_last[last] = seen_last.get(last, 0) + 1
            kept.append((seq, used))
            if len(kept) >= beam_size:
                break
        # if diversity cap starved the beam, fill with top-ranked regardless
        if len(kept) < beam_size:
            have = set(id(s) for s, _ in kept)
            for lg, seq, used in new_beam:
                if id(seq) in have:
                    continue
                kept.append((seq, used))
                have.add(id(seq))
                if len(kept) >= beam_size:
                    break
        beam = kept
    return best_seq


def _dfs_extend(seq, used, deadline):
    """Exact DFS extension to squeeze extra elements. Returns best sequence found."""
    best = list(seq)

    def _rec(seq, used):
        nonlocal best
        if len(seq) > len(best):
            best = list(seq)
        if time.time() > deadline:
            return
        last = seq[-1]
        for x in range(last + 1, 257):
            if time.time() > deadline:
                return
            ns = _valid_suffixes(seq, x, used)
            if ns is None:
                continue
            _rec(seq + [x], used | set(ns))

    _rec(seq, used)
    return best


def _rebuild_used(seq):
    used = set()
    for i in range(len(seq)):
        s = 0
        for j in range(i, len(seq)):
            s += seq[j]
            used.add(s)
    return used


def _verify(seq):
    if not seq:
        return False
    prev = 0
    for v in seq:
        if not (1 <= v <= 256):
            return False
        if v <= prev:
            return False
        prev = v
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def construct():
    deadline = _START + _LIMIT

    # 1) beam search for a strong base
    base = _beam_search(beam_size=400, a=0.75, b=0.25, max_steps=60)
    best = list(base)
    best_used = _rebuild_used(base)

    # 2) exact DFS extension from the base (and a few strong alternatives) if time remains
    if time.time() < deadline - 2.0:
        extended = _dfs_extend(base, best_used, deadline - 1.5)
        if len(extended) > len(best):
            best = extended
            best_used = _rebuild_used(extended)

    # 3) safety: verify and fall back to a known-valid sequence if anything is off
    if not _verify(best):
        # fall back to a greedy valid sequence
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        best = values

    return best


if __name__ == "__main__":
    import time as _t
    _s = _t.time()
    res = construct()
    print("len:", len(res), "time:", round(_t.time() - _s, 2), "valid:", _verify(res))
    print(res)
