"""Combined approach: prefix/diff-set representation + beam search over
multi-start, mixed (greedy-smallest / randomized-window) candidate policies."""
import time
import random

# We want a strictly increasing sequence in [1,256] where all nonempty
# consecutive-block sums are distinct. Maximize length.

# Representation:
#   prefixes: [0, a0, a0+a1, ...]  (length n+1)
#   diffs:    set of all (prefixes[j] - prefixes[i]) for i<j  == all block sums
# Adding a new element a (with new prefix P = prefixes[-1] + a) is valid iff
#   P - p  not in diffs   for every p in prefixes
# (these are exactly the new block sums ending at the new element).
# Then we insert {P - p for p in prefixes} into diffs.

def _all_block_sums(seq):
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _check_valid(seq):
    if not seq:
        return True
    for i in range(1, len(seq)):
        if seq[i] <= seq[i - 1]:
            return False
    if any(x < 1 or x > 256 for x in seq):
        return False
    n = len(seq)
    if len(_all_block_sums(seq)) != n * (n + 1) // 2:
        return False
    return True

def _try_extend(prefixes, diffs, last_a, policy, rng, window):
    """Try to extend one partial sequence by one element.
    policy: 'smallest' -> smallest valid a; 'random' -> random valid a in window.
    Returns (new_prefixes, new_diffs, new_last) or None.
    """
    base = prefixes[-1]
    if policy == 'smallest':
        # scan ascending, take first valid
        for a in range(last_a + 1, 257):
            P = base + a
            ok = True
            for p in prefixes:
                if (P - p) in diffs:
                    ok = False
                    break
            if ok:
                ndiffs = set(diffs)
                for p in prefixes:
                    ndiffs.add(P - p)
                return (prefixes + [P], ndiffs, a)
        return None
    else:
        # random from a window [last_a+1, last_a+window], shuffled
        upper = min(256, last_a + window)
        if upper <= last_a:
            return None
        cands = list(range(last_a + 1, upper + 1))
        rng.shuffle(cands)
        for a in cands:
            P = base + a
            ok = True
            for p in prefixes:
                if (P - p) in diffs:
                    ok = False
                    break
            if ok:
                ndiffs = set(diffs)
                for p in prefixes:
                    ndiffs.add(P - p)
                return (prefixes + [P], ndiffs, a)
        return None

def _beam_search(start_value, beam_width, max_time, start_time, rng):
    """Beam search starting from a fixed first element.
    Each beam member is (prefixes, diffs, last_a, policy, window).
    """
    # seed with first element
    prefixes = [0, start_value]
    diffs = {start_value}
    last_a = start_value

    beam = [(prefixes, diffs, last_a, 'smallest', 0)]

    best_prefixes = prefixes[:]
    best_len = 1

    while True:
        if time.time() - start_time > max_time:
            break
        next_beam = []
        for (pref, dset, la, pol, win) in beam:
            # try several extensions per beam member to add diversity
            # use the member's own policy; occasionally vary window
            for _ in range(beam_width):
                if time.time() - start_time > max_time:
                    break
                # vary the window for random policy
                if pol == 'random':
                    w = win if win > 0 else rng.choice([10, 20, 40, 80, 160, 250])
                else:
                    w = 0
                res = _try_extend(pref, dset, la, pol, rng, w)
                if res is None:
                    break
                npref, ndset, nla = res
                next_beam.append((npref, ndset, nla, pol, w))
                if len(npref) - 1 > best_len:
                    best_len = len(npref) - 1
                    best_prefixes = npref[:]

        if not next_beam:
            break

        # prune beam to keep the best (longest, then smallest last) members
        next_beam.sort(key=lambda t: (-(len(t[0]) - 1), t[2]))
        # dedupe by (prefixes tuple) to avoid identical lines
        seen = set()
        deduped = []
        for t in next_beam:
            key = (tuple(t[0]), t[3])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(t)
            if len(deduped) >= beam_width:
                break
        beam = deduped

    return best_prefixes

def construct():
    total_start = time.time()
    deadline = total_start + 85.0  # safety margin before 90s kill
    rng = random.Random(12345)

    best_seq = []
    best_len = 0

    def consider(prefixes):
        nonlocal best_seq, best_len
        seq = prefixes[1:]
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

    # Phase 1: beam search from several small starts (dense, deep lines)
    for sv in range(1, 13):
        if time.time() > deadline - 5.0:
            break
        bw = 6
        consider(_beam_search(sv, bw, deadline - 2.0, total_start, rng))

    # Phase 2: randomized restarts with varied windows to open sum-space
    # (structural diversity: different first elements, different window sizes)
    while time.time() < deadline - 2.0:
        sv = rng.randint(1, 40)
        bw = rng.choice([4, 6, 8])
        consider(_beam_search(sv, bw, deadline - 2.0, total_start, rng))

    # Also try a couple of larger starts (they leave more room in [1,256]
    # for the dense tail, which can sometimes yield longer sequences).
    for sv in [64, 128, 192]:
        if time.time() > deadline - 2.0:
            break
        consider(_beam_search(sv, 5, deadline - 2.0, total_start, rng))

    # Fallback
    if not best_seq:
        best_seq = [1, 2, 5]

    # Final validity guard
    if not _check_valid(best_seq):
        # rebuild a guaranteed-valid small sequence
        seq = [1]
        prefixes = [0, 1]
        diffs = {1}
        last_a = 1
        while True:
            res = _try_extend(prefixes, diffs, last_a, 'smallest', rng, 0)
            if res is None:
                break
            prefixes, diffs, last_a = res
        if len(prefixes) - 1 > len(seq):
            seq = prefixes[1:]
        best_seq = seq

    return best_seq
