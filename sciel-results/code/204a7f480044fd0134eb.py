import time
import random

N = 12
DEADLINE = 85.0  # seconds, safety margin below 90s kill


def _threshold(aut):
    """Exact shortest reset word length via BFS on subset automaton.
    Returns (length, word) or (None, None) if not synchronizing."""
    a, b = aut
    start = frozenset(range(N))
    # BFS over subsets
    from collections import deque
    dist = {start: 0}
    parent = {start: (None, None)}
    q = deque([start])
    found = None
    while q:
        s = q.popleft()
        if len(s) == 1:
            found = s
            break
        d = dist[s]
        for letter, trans in ((0, a), (1, b)):
            ns = frozenset(trans[x] for x in s)
            if ns not in dist:
                dist[ns] = d + 1
                parent[ns] = (s, letter)
                q.append(ns)
    if found is None:
        return None, None
    # reconstruct word
    word = []
    cur = found
    while parent[cur][0] is not None:
        word.append(parent[cur][1])
        cur = parent[cur][0]
    word.reverse()
    return len(word), word


def _pawlik():
    """Pawlik-style construction for N states.
    a: cycle 0->1->...->N-1->0
    b: maps most states forward by 1, but with a specific structure
    to maximize reset threshold.
    For N=12, use the known Pawlik pattern."""
    a = [(i + 1) % N for i in range(N)]
    # b: identity except a couple of states mapped to create long reset
    # Classic Pawlik: b(i) = i+1 for i in 0..N-2, b(N-1)=0, but with
    # a twist: b(0)=1, b(1)=0, rest shift
    # Better: use the construction that gives (N-1)^2 for specific N
    # For general N, a good baseline:
    b = [0] * N
    for i in range(N):
        b[i] = (i + 1) % N
    # Modify to create non-trivial reset: map state 0 to 1 under b,
    # and state 1 to 0 under b, creating a "pin" structure
    b[0] = 1
    b[1] = 0
    # This gives a synchronizing automaton with longer threshold
    return [a, b]


def _random_aut():
    """Random complete binary automaton."""
    a = [random.randint(0, N - 1) for _ in range(N)]
    b = [random.randint(0, N - 1) for _ in range(N)]
    return [a, b]


def _is_sync(aut):
    """Quick check if automaton synchronizes (BFS, early exit)."""
    a, b = aut
    from collections import deque
    seen = set()
    start = frozenset(range(N))
    seen.add(start)
    q = deque([start])
    while q:
        s = q.popleft()
        if len(s) == 1:
            return True
        for trans in (a, b):
            ns = frozenset(trans[x] for x in s)
            if ns not in seen:
                seen.add(ns)
                q.append(ns)
    return False


def construct():
    t0 = time.time()
    best_aut = None
    best_thresh = 0

    # Start with Pawlik baseline
    aut = _pawlik()
    th, _ = _threshold(aut)
    if th is not None:
        best_aut = [list(x) for x in aut]
        best_thresh = th

    # Local search with random restarts
    current = [list(x) for x in best_aut] if best_aut else _random_aut()
    cur_th = best_thresh if best_aut else 0
    if cur_th == 0:
        # Try random starts until we find a synchronizing one
        for _ in range(500):
            if time.time() - t0 > DEADLINE:
                break
            cand = _random_aut()
            if _is_sync(cand):
                th, _ = _threshold(cand)
                if th is not None:
                    current = cand
                    cur_th = th
                    if best_aut is None or th > best_thresh:
                        best_aut = [list(x) for x in cand]
                        best_thresh = th
                    break

    # Local search: mutate one transition entry at a time
    stall_count = 0
    while time.time() - t0 < DEADLINE:
        # Pick a random letter and state to mutate
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        old_val = current[letter][state]
        new_val = random.randint(0, N - 1)
        if new_val == old_val:
            continue
        current[letter][state] = new_val

        # Check if still synchronizing and get threshold
        th, _ = _threshold(current)
        if th is None:
            # Not synchronizing, revert
            current[letter][state] = old_val
            stall_count += 1
        elif th > cur_th:
            # Improvement
            cur_th = th
            stall_count = 0
            if th > best_thresh:
                best_thresh = th
                best_aut = [list(x) for x in current]
        else:
            # No improvement, revert (strict improvement only)
            current[letter][state] = old_val
            stall_count += 1

        # Random restart if stalled too long
        if stall_count > 200:
            stall_count = 0
            # Try a new random start, but keep best
            for _ in range(100):
                if time.time() - t0 > DEADLINE:
                    break
                cand = _random_aut()
                if _is_sync(cand):
                    th2, _ = _threshold(cand)
                    if th2 is not None and th2 > cur_th:
                        current = [list(x) for x in cand]
                        cur_th = th2
                        if th2 > best_thresh:
                            best_thresh = th2
                            best_aut = [list(x) for x in cand]
                        break
            else:
                # Reset to best known
                current = [list(x) for x in best_aut]
                cur_th = best_thresh

    # Return best automaton found
    if best_aut is None:
        # Fallback: trivial synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
    return best_aut
