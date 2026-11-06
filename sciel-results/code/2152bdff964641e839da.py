"""Systematic enumeration + refinement for Černý 12-state reset threshold."""
import time
from collections import deque
import itertools

N = 12
DEADLINE = 80.0  # safety margin before 90s kill
_start = time.time()

def _time_left():
    return DEADLINE - (time.time() - _start)

def _bfs_threshold(a_map, b_map):
    """Compute exact shortest reset word length via BFS on subset automaton.
    Returns (threshold, word) or (None, None) if not synchronizing."""
    # Precompute subset transitions
    def apply_letter(subset, letter):
        if letter == 0:
            return frozenset(a_map[s] for s in subset)
        else:
            return frozenset(b_map[s] for s in subset)
    
    full = frozenset(range(N))
    if len(full) == 1:
        return (0, "")
    
    # BFS from full set
    visited = {full: (0, "")}
    queue = deque([full])
    
    while queue:
        cur = queue.popleft()
        dist, word = visited[cur]
        for letter in range(2):
            nxt = apply_letter(cur, letter)
            if nxt not in visited:
                new_word = word + ('a' if letter == 0 else 'b')
                visited[nxt] = (dist + 1, new_word)
                if len(nxt) == 1:
                    return (dist + 1, new_word)
                queue.append(nxt)
    
    return (None, None)

def _is_synchronizing(a_map, b_map):
    """Quick check if automaton synchronizes without computing full threshold."""
    full = frozenset(range(N))
    visited = {full}
    queue = deque([full])
    while queue:
        cur = queue.popleft()
        for letter in range(2):
            if letter == 0:
                nxt = frozenset(a_map[s] for s in cur)
            else:
                nxt = frozenset(b_map[s] for s in cur)
            if len(nxt) == 1:
                return True
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return False

def _make_cycle_with_break(base, k, letter):
    """Cycle on 0..base-1, with state base mapped by 'letter' to 0, rest cycle."""
    a = list(range(1, base)) + [0]
    b = list(range(1, base)) + [0]
    # Extend to N states
    if base < N:
        for s in range(base, N):
            a.append(0)
            b.append(0)
    # Apply break: state k under 'letter' goes to 0
    if letter == 0:
        a[k] = 0
    else:
        b[k] = 0
    return a, b

def _make_pinwheel(base):
    """Pinwheel: a is cycle, b rotates with offset."""
    a = [(i + 1) % base for i in range(base)] + [0] * (N - base)
    b = [(i + 2) % base for i in range(base)] + [0] * (N - base)
    return a, b

def _make_vorel_variant(seed_state, base):
    """Vorel-like: a is cycle, b maps most to 0 but one goes to seed."""
    a = [(i + 1) % base for i in range(base)] + [0] * (N - base)
    b = [0] * N
    b[seed_state] = (seed_state + 1) % base
    return a, b

def construct():
    global _start
    _start = time.time()
    
    best_threshold = 0
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0] * N
    
    # Phase 1: Systematic enumeration of structural families
    # Family 1: Cycle of length L (L from 7 to 12) with break at various positions
    for L in range(7, N + 1):
        if _time_left() < 10:
            break
        for k in range(L):
            for letter in range(2):
                a, b = _make_cycle_with_break(L, k, letter)
                if _is_synchronizing(a, b):
                    thresh, _ = _bfs_threshold(a, b)
                    if thresh is not None and thresh > best_threshold:
                        best_threshold = thresh
                        best_a, best_b = a, b
    
    # Family 2: Pinwheel variants
    for L in range(6, N + 1):
        if _time_left() < 10:
            break
        for offset in range(1, L):
            a = [(i + 1) % L for i in range(L)] + [0] * (N - L)
            b = [(i + offset) % L for i in range(L)] + [0] * (N - L)
            if _is_synchronizing(a, b):
                thresh, _ = _bfs_threshold(a, b)
                if thresh is not None and thresh > best_threshold:
                    best_threshold = thresh
                    best_a, best_b = a, b
    
    # Family 3: Vorel variants
    for L in range(6, N + 1):
        if _time_left() < 10:
            break
        for seed_state in range(L):
            a, b = _make_vorel_variant(seed_state, L)
            if _is_synchronizing(a, b):
                thresh, _ = _bfs_threshold(a, b)
                if thresh is not None and thresh > best_threshold:
                    best_threshold = thresh
                    best_a, best_b = a, b
    
    # Family 4: Two-cycle structure - states 0..L-1 form cycle under a,
    # states L..N-1 mapped differently
    for L in range(5, N):
        if _time_left() < 8:
            break
        for m in range(L, N):  # m = where state L maps under b
            a = [(i + 1) % L for i in range(L)]
            b = [0] * N
            for i in range(L):
                b[i] = 0
            for i in range(L, N):
                a.append(0)
                b.append(m if i == L else (i + 1) % N)
            # Ensure completeness
            while len(a) < N:
                a.append(0)
                b.append(0)
            a = a[:N]
            b = b[:N]
            if _is_synchronizing(a, b):
                thresh, _ = _bfs_threshold(a, b)
                if thresh is not None and thresh > best_threshold:
                    best_threshold = thresh
                    best_a, best_b = a, b
    
    # Phase 2: Hill-climbing refinement
    # Random single-position mutations
    import random
    rng = random.Random(42)
    
    a_cur = list(best_a)
    b_cur = list(best_b)
    
    for iteration in range(50000):
        if _time_left() < 5:
            break
        # Random mutation: change one random mapping
        if rng.random() < 0.5:
            pos = rng.randint(0, N - 1)
            old = a_cur[pos]
            a_cur[pos] = rng.randint(0, N - 1)
            if a_cur[pos] != old:
                if _is_synchronizing(a_cur, b_cur):
                    thresh, _ = _bfs_threshold(a_cur, b_cur)
                    if thresh is not None and thresh > best_threshold:
                        best_threshold = thresh
                        best_a, best_b = list(a_cur), list(b_cur)
                    else:
                        a_cur[pos] = old
                else:
                    a_cur[pos] = old
        else:
            pos = rng.randint(0, N - 1)
            old = b_cur[pos]
            b_cur[pos] = rng.randint(0, N - 1)
            if b_cur[pos] != old:
                if _is_synchronizing(a_cur, b_cur):
                    thresh, _ = _bfs_threshold(a_cur, b_cur)
                    if thresh is not None and thresh > best_threshold:
                        best_threshold = thresh
                        best_a, best_b = list(a_cur), list(b_cur)
                    else:
                        b_cur[pos] = old
                else:
                    b_cur[pos] = old
        
        # Occasionally swap two entries
        if rng.random() < 0.1 and _time_left() > 3:
            p1 = rng.randint(0, N - 1)
            p2 = rng.randint(0, N - 1)
            if p1 != p2:
                a_cur[p1], a_cur[p2] = a_cur[p2], a_cur[p1]
                if _is_synchronizing(a_cur, b_cur):
                    thresh, _ = _bfs_threshold(a_cur, b_cur)
                    if thresh is not None and thresh > best_threshold:
                        best_threshold = thresh
                        best_a, best_b = list(a_cur), list(b_cur)
                else:
                    a_cur[p1], a_cur[p2] = a_cur[p2], a_cur[p1]
    
    # Phase 3: Final verification
    if best_threshold > 0:
        thresh, word = _bfs_threshold(best_a, best_b)
        if thresh is not None:
            return (best_a, best_b)
    
    # Fallback: guaranteed synchronizing
    return ([(i + 1) % N for i in range(N)], [0] * N)
