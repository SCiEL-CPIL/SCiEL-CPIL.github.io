"""Bitmask iterative-deepening DFS for increasing sequences with unique consecutive-block sums."""
import time

# ---- Global search state (avoid passing many args through recursion) ----
_MAX = 256
_deadline = 0.0
_best_seq = None
_best_len = 0
_nodes = 0
_stop = False


def _update_best(seq, length):
    global _best_seq, _best_len
    if length > _best_len:
        _best_len = length
        _best_seq = tuple(seq)


def _dfs(seq, tails, forbidden, last, length):
    """Iterative-deepening DFS. `tails` is a bitmask of active block-sums ending
    at the current last element; `forbidden` is the bitmask of all used sums."""
    global _nodes, _stop
    _nodes += 1
    if _stop or time.time() > _deadline:
        _stop = True
        return

    # Upper bound on remaining elements given we must stay strictly increasing
    # within [last+1, 256].
    max_more = (256 - last) // 2
    if length + max_more < _best_len:
        return  # cannot beat best; prune

    # Try each candidate next value (ascending for speed; first hit updates best)
    start = last + 1
    end = 256
    for x in range(start, end + 1):
        new_tails = tails | (tails << x)
        if (new_tails & forbidden) == 0:
            # valid
            seq.append(x)
            new_forbidden = forbidden | new_tails
            _dfs(seq, new_tails, new_forbidden, x, length + 1)
            seq.pop()
            if _stop or time.time() > _deadline:
                _stop = True
                return
            _update_best(seq, length + 1)


def _greedy():
    """Fast greedy seed: iterate 1..256, take x if it doesn't collide."""
    values = []
    forbidden = 0
    for x in range(1, 257):
        # tails of the current sequence: all block-sums ending at values[-1]
        # but we need to compute tails for the candidate x appended.
        # Compute tails for the existing sequence ending at each position.
        # Simpler: maintain tails for the full sequence.
        # tails_i = sum of values[i:] (suffix sums)
        # When we append x, new tails = old suffix sums + x, plus x itself.
        # Actually: new_tails = {x} ∪ {x + s for s in old_tails}
        # where old_tails are the suffix sums of the current sequence.
        # Let's maintain `tails` as a set of suffix sums.
        pass
    # Redo cleanly
    values = []
    tails = set()  # suffix sums of current sequence
    forbidden = 0
    for x in range(1, 257):
        new_tails = {x} | {x + s for s in tails}
        # check collision
        collides = False
        for s in new_tails:
            if (forbidden >> s) & 1:
                collides = True
                break
        if not collides:
            values.append(x)
            tails = new_tails
            for s in new_tails:
                forbidden |= (1 << s)
    return values


def construct():
    global _deadline, _best_seq, _best_len, _nodes, _stop

    t0 = time.time()
    _deadline = t0 + 80.0  # 10s safety margin before 90s kill
    _best_seq = None
    _best_len = 0
    _nodes = 0
    _stop = False

    # Phase 1: greedy seed
    seq = _greedy()
    _update_best(seq, len(seq))

    # Phase 2: iterative deepening DFS from small depths upward
    # Start from the greedy length and try to extend.
    # We'll do IDDFS: for target in range(best_len, 257):
    #   try to find a sequence of length >= target
    # But that's expensive. Instead, do a single DFS that explores deeply.

    # Better: do a DFS that tries to build the longest sequence, using the
    # upper-bound prune. Start from empty sequence.
    # But that's too slow for large depths. Use the greedy as a starting point
    # and try to extend it via DFS.

    # Strategy: 
    # 1. Greedy gives a baseline.
    # 2. Try to extend the greedy sequence by DFS from its end.
    # 3. Also try different starting points.

    # For simplicity and speed, do a single DFS from scratch with the
    # upper-bound prune. The prune makes it fast for proving optimality.

    # Actually, let's do IDDFS properly:
    # For target_length in range(1, 257):
    #   if target_length <= best_len: continue
    #   do DFS to find sequence of length target_length
    #   if found: best_len = target_length

    # But DFS from scratch for each target is wasteful. Instead:
    # Do a single DFS that goes as deep as possible, updating best as we go.

    # Let's just do a DFS from the empty sequence, with the prune.
    # The prune (length + (256-last)//2 < best_len) will cut off branches
    # that can't beat the current best.

    _dfs([], 0, 0, 0, 0)

    if _best_seq is None:
        return list(seq)
    return list(_best_seq)
