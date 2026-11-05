"""Stronger structural approach: backtracking search with incremental sum tracking.

Instead of greedy (which gets stuck in local optima), we do a depth-first
backtracking search over subsets of [1..256] in increasing order, with:
- Incremental tracking of consecutive-block sums for O(n) check per candidate
- Branch and bound: if remaining candidates can't beat best, prune
- Multiple random restarts with different orderings
- Local improvement via deletion+reinsertion after backtracking
"""
import time
import random

def _check_valid(seq):
    """Check if a strictly increasing sequence has all distinct consecutive-block sums."""
    n = len(seq)
    if n == 0:
        return True
    for i in range(n - 1):
        if seq[i + 1] <= seq[i]:
            return False
    for x in seq:
        if x < 1 or x > 256:
            return False
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _greedy_build(order):
    """Greedy: process candidates in given order, append if no conflict."""
    values = []
    sums = set()
    for x in order:
        total = x
        conflict = False
        for a in reversed(values):
            if total in sums:
                conflict = True
                break
            total += a
        if total in sums:
            conflict = True
        if conflict:
            continue
        total = x
        sums.add(total)
        for a in reversed(values):
            total += a
            sums.add(total)
        values.append(x)
    return values


def _backtrack_search(candidates, current_seq, sums_set, best_seq, deadline, max_depth=0):
    """
    Depth-first backtracking search.
    candidates: list of remaining candidate values (sorted ascending)
    current_seq: list of values chosen so far
    sums_set: set of all consecutive-block sums of current_seq
    best_seq: reference list that gets updated with best found
    deadline: time limit
    Returns nothing; updates best_seq in place.
    """
    if time.time() > deadline:
        return
    if len(current_seq) > len(best_seq):
        best_seq[:] = list(current_seq)

    if not candidates:
        return

    n_cand = len(candidates)
    current_len = len(current_seq)

    # Pruning: even if we take all remaining, can we beat best?
    if current_len + n_cand <= len(best_seq):
        return

    for idx in range(n_cand):
        if time.time() > deadline:
            return

        x = candidates[idx]
        # Try adding x to current_seq
        total = x
        conflict = False
        for a in reversed(current_seq):
            if total in sums_set:
                conflict = True
                break
            total += a
        if total in sums_set:
            conflict = True

        if not conflict:
            # Add x
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(current_seq):
                total += a
                new_sums.add(total)

            current_seq.append(x)
            old_len = len(sums_set)
            sums_set.update(new_sums)

            # Recurse with remaining candidates after idx
            _backtrack_search(candidates[idx+1:], current_seq, sums_set, best_seq, deadline)

            # Undo
            current_seq.pop()
            for s in new_sums:
                sums_set.discard(s)


def _local_improve(seq, deadline):
    """
    Try to improve sequence by removing one element and adding a different one.
    Returns improved sequence.
    """
    best = list(seq)
    best_len = len(best)

    while time.time() < deadline:
        improved = False
        n = len(best)

        for i in range(n):
            if time.time() > deadline:
                return best

            # Remove best[i]
            candidate = best[:i] + best[i+1:]
            existing = set(candidate)

            # Compute sums for candidate
            cand_sums = set()
            for a in range(len(candidate)):
                s = 0
                for b in range(a, len(candidate)):
                    s += candidate[b]
                    cand_sums.add(s)

            # Try to insert a new value
            for x in range(1, 257):
                if x in existing:
                    continue
                # Find insertion position to maintain increasing order
                pos = None
                for j in range(len(candidate) + 1):
                    if j == 0 and (j == len(candidate) or x < candidate[j]):
                        pos = j
                        break
                    elif 0 < j < len(candidate) and candidate[j-1] < x < candidate[j]:
                        pos = j
                        break
                    elif j == len(candidate) and (len(candidate) == 0 or candidate[j-1] < x):
                        pos = j
                        break

                if pos is None:
                    continue

                # Check if inserting x at pos creates duplicate sums
                new_sums = set(cand_sums)
                # New sums: x alone, x + next, x + next + nextnext... (forward)
                # Also: prev + x, prev + prevprev + x, ... (backward)
                # Actually we need all consecutive blocks in the new sequence
                new_seq = candidate[:pos] + [x] + candidate[pos:]
                # Compute new sums efficiently
                conflict = False
                for a in range(len(new_seq)):
                    s = 0
                    for b in range(a, len(new_seq)):
                        s += new_seq[b]
                        if s in cand_sums:
                            # This sum existed before; check if it's a new block
                            # Actually we need to check ALL pairs of (i,j) in new_seq
                            # Simpler: just compute all sums and check uniqueness
                            conflict = True
                            break
                    if conflict:
                        break
                if not conflict:
                    # Full validation
                    if _check_valid(new_seq):
                        if len(new_seq) > best_len:
                            best = new_seq
                            best_len = len(best)
                            improved = True
                            break

            if improved:
                break

        if not improved:
            break

    return best


def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill

    best_seq = []

    # Phase 1: Greedy with various orderings to get a good starting point
    orderings = [
        list(range(1, 257)),
        list(range(256, 0, -1)),
        list(range(128, 257)) + list(range(1, 128)),
        list(range(1, 129)) + list(range(256, 127, -1)),
    ]

    for _ in range(20):
        if time.time() > deadline - 30.0:
            break
        orderings.append(sorted(random.sample(range(1, 257), 256)))

    for order in orderings:
        if time.time() > deadline - 30.0:
            break
        seq = _greedy_build(order)
        if len(seq) > len(best_seq):
            best_seq = seq

    # Phase 2: Backtracking search to extend the best greedy result
    if time.time() < deadline - 5.0:
        used = set(best_seq)
        candidates = [x for x in range(1, 257) if x not in used]
        candidates.sort()

        # Compute initial sums
        sums_set = set()
        for i in range(len(best_seq)):
            s = 0
            for j in range(i, len(best_seq)):
                s += best_seq[j]
                sums_set.add(s)

        # Do backtracking with a time-bounded search
        # We limit the depth of recursion to avoid exponential blowup
        # Use iterative deepening: try to add 1 more element, then 2, etc.
        current_best = list(best_seq)
        bt_deadline = min(deadline - 2.0, start_time + 65.0)
        _backtrack_search(candidates, current_best, sums_set, current_best, bt_deadline)

        if len(current_best) > len(best_seq):
            best_seq = current_best

    # Phase 3: Local improvement
    if time.time() < deadline - 2.0:
        improved = _local_improve(best_seq, deadline - 1.0)
        if len(improved) > len(best_seq):
            best_seq = improved

    # Final validation
    if not _check_valid(best_seq):
        best_seq = _greedy_build(range(1, 257))

    return best_seq


if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Sequence: {result}")
