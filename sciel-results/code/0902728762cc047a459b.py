"""Construct a 12-state binary automaton maximizing the shortest reset word length."""
import time
import random
from collections import deque

N = 12
_DEADLINE = 45.0  # safety margin below the 60s wall-clock kill
_start = time.time()


def _elapsed():
    return time.time() - _start


def _is_synchronizing(a, b):
    """BFS over subsets; return (threshold, word) or (None, None)."""
    start = frozenset(range(N))
    visited = {start: None}  # subset -> (parent_subset, letter)
    q = deque([start])
    while q:
        cur = q.popleft()
        if len(cur) == 1:
            # reconstruct word
            word = []
            s = cur
            while visited[s] is not None:
                parent, letter = visited[s]
                word.append(letter)
                s = parent
            word.reverse()
            return len(word), word
        for letter, trans in ((0, a), (1, b)):
            nxt = frozenset(trans[i] for i in cur)
            if nxt not in visited:
                visited[nxt] = (cur, letter)
                q.append(nxt)
    return None, None


def _threshold_only(a, b):
    """Return the exact shortest reset word length, or None if not synchronizing."""
    # Bitmask representation for speed
    start_mask = (1 << N) - 1
    visited = {start_mask}
    q = deque([(start_mask, 0)])
    
    while q:
        cur, d = q.popleft()
        # Check if single state: power of 2
        if cur & (cur - 1) == 0:
            return d
        
        # Next state under 'a'
        next_a = 0
        for i in range(N):
            if cur & (1 << i):
                next_a |= (1 << a[i])
        if next_a not in visited:
            visited.add(next_a)
            q.append((next_a, d + 1))
            
        # Next state under 'b'
        next_b = 0
        for i in range(N):
            if cur & (1 << i):
                next_b |= (1 << b[i])
        if next_b not in visited:
            visited.add(next_b)
            q.append((next_b, d + 1))
            
    return None


def construct():
    """Return two length-N lists of destination states in 0..N-1.

    Must define a complete binary deterministic automaton that synchronizes.
    Score is the exact shortest reset-word length, maximized.
    """
    best_thresh = -1
    best_a = None
    best_b = None
    rng = random.Random(12345)

    # A known-good fallback (cycle + sink) guarantees synchronization.
    fallback_a = [(i + 1) % N for i in range(N)]
    fallback_b = [0] * N

    while _elapsed() < _DEADLINE:
        # Generate a random automaton. Bias 'a' to be a permutation (cycle-like)
        # and 'b' to be a random map, which tends to produce long reset words.
        a = list(range(N))
        rng.shuffle(a)  # 'a' is a random permutation
        # 'b' is a random map; make it non-permutation to allow synchronization
        b = [rng.randrange(N) for _ in range(N)]
        # Occasionally make 'b' collapse more to ensure synchronizability
        if rng.random() < 0.5:
            # make a couple of states map to the same target
            for _ in range(rng.randint(1, 3)):
                i = rng.randrange(N)
                j = rng.randrange(N)
                b[i] = b[j]

        # Quick check: must synchronize
        thresh = _threshold_only(a, b)
        if thresh is None:
            continue
        if thresh > best_thresh:
            best_thresh = thresh
            best_a = a
            best_b = b
            # If we hit a very high value, we can stop early
            if best_thresh >= 120:
                break

    if best_a is None:
        # Fallback to the known synchronizing automaton
        return fallback_a, fallback_b
    return best_a, best_b
