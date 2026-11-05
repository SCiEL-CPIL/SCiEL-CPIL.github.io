# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

_HARD_LIMIT = 120.0
_DEADLINE = _HARD_LIMIT - 2.0   # safety margin
_T0 = time.monotonic()


def _remaining():
    return _DEADLINE - (time.monotonic() - _T0)


def construct(n):
    global _T0, _DEADLINE
    _T0 = time.monotonic()
    _DEADLINE = _HARD_LIMIT - 2.0

    if n <= 0:
        return ""
    if n == 1:
        return "1"

    rng = random.Random(12345)

    # ---------- Greedy construction ----------
    # state: list of used perms (each a tuple of ints 0..n-1)
    usage = [0] * (n ** n)

    def idx_of(p):
        v = 0
        for x in p:
            v = v * n + x
        return v

    def hash_of(p):
        v = 0
        for x in p:
            v = (v * 131 + x) & 0xFFFFFFFF
        return v

    state = []
    suffix = []
    total = 0
    for _ in range(n ** n):
        if len(suffix) < n - 1:
            # not enough context; pick the least-used symbol
            best = min(range(n), key=lambda x: (usage[x], hash_of((x,))))
            suffix.append(best)
            usage[best] += 1
            total += 1
            continue
        last = tuple(suffix[-(n - 1):])
        # find least-used symbol completing last into a new perm
        best_sym = None
        best_key = None
        for s in range(n):
            cand = last + (s,)
            ci = idx_of(cand)
            if usage[ci] > 0:
                continue
            key = (usage[s], hash_of(cand))
            if best_key is None or key < best_key:
                best_key = key
                best_sym = s
        if best_sym is None:
            # fallback (should not happen): pick any unused
            for s in range(n):
                ci = idx_of(last + (s,))
                if usage[ci] == 0:
                    best_sym = s
                    break
        suffix.append(best_sym)
        total += 1
        state.append(last + (best_sym,))
        usage[idx_of(last + (best_sym,))] += 1

    # ---------- Overlap precomputation (Z-function) ----------
    # suffix_len[i][j] = longest suffix of state[i] that is a prefix of state[j]
    m = n ** n
    # build byte strings
    bytes_list = [bytes(p) for p in state]
    sep = b'\xff'

    def z_overlap(a_bytes, b_bytes):
        # return longest suffix of a that is a prefix of b
        s = a_bytes + sep + b_bytes
        L = len(s)
        z = [0] * L
        l = r = 0
        for i in range(1, L):
            if i < r:
                z[i] = min(r - i, z[i - l])
            while i + z[i] < L and s[z[i]] == s[i + z[i]]:
                z[i] += 1
            if i + z[i] > r:
                l, r = i, i + z[i]
        # overlap = longest k such that a[-k:] == b[:k]
        # In s, position of b start = len(a)+1. z at that position gives
        # longest prefix of b matching substring starting there, but we need
        # suffix of a. Instead compute on reversed strings.
        ra = a_bytes[::-1]
        rb = b_bytes[::-1]
        t = rb + sep + ra
        Lt = len(t)
        zt = [0] * Lt
        l = r = 0
        for i in range(1, Lt):
            if i < r:
                zt[i] = min(r - i, zt[i - l])
            while i + zt[i] < Lt and t[zt[i]] == t[i + zt[i]]:
                zt[i] += 1
            if i + zt[i] > r:
                l, r = i, i + zt[i]
        # at i = len(rb)+1, zt[i] = longest suffix of ra (= prefix of a) that is prefix of rb (= suffix of b)
        # that's not what we want. Let's do it directly with the first method.
        # Actually suffix of a == prefix of b  <=>  reverse: suffix of b == prefix of a.
        # Let's just brute with the z on (b + sep + a): at position len(b)+1,
        # z gives longest prefix of a matching substring at that pos in (b+sep+a),
        # i.e., longest prefix of a that appears starting at end of b... not suffix.
        # Simplest correct: use z on (a + sep + b) is for prefix of a. We need suffix.
        # Use reversed approach correctly:
        # suffix of a of length k == prefix of b of length k
        # <=> a[-k:] == b[:k]
        # <=> reverse(a[-k:]) == reverse(b[:k])
        # <=> prefix of reverse(a) of length k == prefix of reverse(b) of length k
        # So it's just the longest common PREFIX of reverse(a) and reverse(b)!
        pass
        # compute LCP of reverse(a) and reverse(b)
        ra = a_bytes[::-1]
        rb = b_bytes[::-1]
        k = 0
        lim = min(len(ra), len(rb))
        while k < lim and ra[k] == rb[k]:
            k += 1
        return k

    # Precompute overlap matrix
    suffix_len = [[0] * m for _ in range(m)]
    for i in range(m):
        ai = bytes_list[i]
        for j in range(m):
            if i == j:
                suffix_len[i][j] = n  # full
            else:
                suffix_len[i][j] = z_overlap(ai, bytes_list[j])

    # ---------- Length computation ----------
    # total length = n + sum over i of (n - suffix_len[state[i]][state[i+1]])
    # Build an array of overlaps between consecutive elements of the current order.
    order = list(range(m))

    def compute_overlaps(order):
        ov = [0] * (m - 1)
        for i in range(m - 1):
            ov[i] = suffix_len[order[i]][order[i + 1]]
        return ov

    ov = compute_overlaps(order)

    def total_length(ov):
        return n + (m - 1) * n - sum(ov)

    best_len = total_length(ov)

    # ---------- Local search: Or-opt (move a block of size 1..3) ----------
    # Reuse difference: when we move a contiguous block [s..e] to position t,
    # only overlaps at s-1, e, e+1, t-1, t change (with edge cases).
    # We'll just recompute the affected overlaps and update sum.

    def do_move(order, ov, s, e, t):
        # remove block s..e (inclusive), then insert at t (t is position in the
        # list AFTER removal). Returns new ov and length delta, or None if no change.
        # We'll just recompute ov for affected indices to be safe but fast.
        # Simpler: rebuild ov fully is O(m) = 5039, too slow for many moves.
        # Do targeted recompute.
        # Identify affected indices in the ORIGINAL ov array (before move):
        # We'll construct new order and recompute only affected overlaps.
        block = order[s:e + 1]
        # order after removal
        new_order = order[:s] + order[e + 1:]
        # insert at t (0 <= t <= len(new_order))
        new_order = new_order[:t] + block + new_order[t:]
        # affected overlap indices in new_order: around t-1, t, t+len(block), and
        # the gap where block was removed: indices s-1 and (e in new order)
        # To be safe, recompute overlaps at indices: t-1, t..t+len(block)-1, t+len(block),
        # and s-1, s (where s is the index in new_order corresponding to the removal gap)
        # The removal gap in new_order is at index s (since we removed s..e, the element
        # that was at e+1 is now at index s).
        aff = set()
        aff.add(t - 1)
        for k in range(len(block)):
            aff.add(t + k)
        aff.add(t + len(block))
        aff.add(s - 1)
        aff.add(s)
        old_sum = sum(ov)
        # recompute affected
        new_ov = ov[:]
        changed = False
        for idx in aff:
            if 0 <= idx < m - 1:
                val = suffix_len[new_order[idx]][new_order[idx + 1]]
                if new_ov[idx] != val:
                    new_ov[idx] = val
                    changed = True
        if not changed:
            return None
        return new_order, new_ov

    def try_move(order, ov, s, e, t):
        res = do_move(order, ov, s, e, t)
        if res is None:
            return None
        new_order, new_ov = res
        new_len = total_length(new_ov)
        return new_order, new_ov, new_len

    # Hill-climbing with random restarts of temperature (accept worsening early)
    cur_order = order
    cur_ov = ov
    cur_len = best_len

    block_sizes = [1, 2, 3]
    no_improve = 0
    max_no_improve = 4000
    start_time = time.monotonic()

    while _remaining() > 3.0:
        # pick a move
        bs = rng.choice(block_sizes)
        s = rng.randrange(0, m - bs)
        e = s + bs - 1
        # choose target t, avoid same location
        t = rng.randrange(0, m - bs)
        if t == s:
            continue
        # ensure t is not within [s, e] (after removal it shifts)
        if t > e:
            t = t - bs
        if t == s:
            continue
        res = try_move(cur_order, cur_ov, s, e, t)
        if res is None:
            continue
        new_order, new_ov, new_len = res
        if new_len < cur_len:
            cur_order, cur_ov, cur_len = new_order, new_ov, new_len
            if cur_len < best_len:
                best_len = cur_len
            no_improve = 0
        else:
            no_improve += 1
            # occasionally accept a small worsening to escape (simulated-annealing-like)
            if no_improve % 200 == 0 and new_len - cur_len <= 2 and rng.random() < 0.05:
                cur_order, cur_ov, cur_len = new_order, new_ov, new_len
        if no_improve > max_no_improve:
            # restart: reshuffle order (still valid), recompute
            cur_order = list(range(m))
            rng.shuffle(cur_order)
            cur_ov = compute_overlaps(cur_order)
            cur_len = total_length(cur_ov)
            no_improve = 0
            max_no_improve = int(max_no_improve * 1.3)
        # periodic time check
        if time.monotonic() - start_time > 5.0:
            start_time = time.monotonic()

    # ---------- Build final word ----------
    final_order = cur_order
    # Build word: start with first perm, then append (n - overlap) chars for each next.
    word = list(state[final_order[0]])
    for i in range(1, m):
        prev = state[final_order[i - 1]]
        cur = state[final_order[i]]
        ov_len = suffix_len[final_order[i - 1]][final_order[i]]
        word.extend(cur[ov_len:])

    return "".join(str(x + 1) for x in word)
# EVOLVE-BLOCK-END
