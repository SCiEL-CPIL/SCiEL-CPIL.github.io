"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 75.0  # safety margin before 90s kill

    N = 256
    # Precompute gcd-based conflict: conflict[a][b] = True if (a*b)%(a+b)==0
    # For efficiency, compute on the fly but cache the "d" structure.
    import math
    from math import gcd

    # Precompute for each a,b whether they conflict.
    # conflict if (a+b) | d^2 where d=gcd(a,b)
    # Build a list of "forbidden partners" using the x+y|d test.
    # Represent selection as a boolean array.

    # Start with all odds (known valid, size 128)
    selected = [False] * (N + 1)
    for i in range(1, N + 1):
        if i % 2 == 1:
            selected[i] = True

    def conflicts_with(i, sel):
        """Return True if i conflicts with any currently selected j != i."""
        for j in range(1, N + 1):
            if j == i or not sel[j]:
                continue
            s = i + j
            if (i * j) % s == 0:
                return True
        return False

    best = [x for x in range(1, N + 1) if selected[x]]

    # Greedily try to add even numbers (and any not selected)
    # Order candidates by "number of conflicts" ascending to add easy ones first.
    # But computing conflicts for each candidate is O(N). Do it.

    def candidate_conflict_count(i, sel):
        c = 0
        for j in range(1, N + 1):
            if j == i or not sel[j]:
                continue
            s = i + j
            if (i * j) % s == 0:
                c += 1
        return c

    # Try adding each unselected number in order of fewest conflicts.
    unselected = [i for i in range(1, N + 1) if not selected[i]]
    # Sort by conflict count
    unselected.sort(key=lambda i: candidate_conflict_count(i, selected))

    for i in unselected:
        if time.time() > deadline:
            break
        if conflicts_with(i, selected):
            continue
        selected[i] = True

    best = [x for x in range(1, N + 1) if selected[x]]
    best_size = len(best)

    # Local search: try to swap out a selected element for an unselected one to gain.
    # Try: for each unselected i, find a selected j whose removal allows i (and maybe more).
    # Simpler hill-climb: try removing one selected and adding one unselected if it increases size.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        unselected = [i for i in range(1, N + 1) if not selected[i]]
        # For each unselected, compute how many selected it conflicts with.
        # If we remove exactly those conflicting selected, we can add i. Net change = 1 - (#conflicts).
        # Only profitable if #conflicts == 0 (already handled) — so single swap never helps unless 0 conflicts.
        # Instead try: remove a set of conflicting selected (k of them) and add i plus maybe others.
        # Keep it simple: try adding i after removing its conflict set, then greedily refill.
        for i in unselected:
            if time.time() > deadline:
                break
            confs = [j for j in range(1, N + 1) if selected[j] and j != i and (i * j) % (i + j) == 0]
            if len(confs) >= 1:
                # Tentatively remove confs, add i, then greedily refill with others.
                for j in confs:
                    selected[j] = False
                selected[i] = True
                # Greedy refill
                others = [k for k in range(1, N + 1) if not selected[k] and k != i]
                others.sort(key=lambda k: candidate_conflict_count(k, selected))
                for k in others:
                    if time.time() > deadline:
                        break
                    if not conflicts_with(k, selected):
                        selected[k] = True
                new_size = sum(selected[1:])
                if new_size > best_size:
                    best_size = new_size
                    best = [x for x in range(1, N + 1) if selected[x]]
                    improved = True
                else:
                    # revert
                    selected[i] = False
                    for j in confs:
                        selected[j] = True
                if time.time() > deadline:
                    break

    return best
# EVOLVE-BLOCK-END
