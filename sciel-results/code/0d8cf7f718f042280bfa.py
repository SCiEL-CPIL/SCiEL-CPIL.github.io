"""Černý problem N=12: maximize shortest reset word length.
Strategy: precompute bitmask transition tables for O(1) BFS steps,
then search structured families known to produce long reset words.
"""
import time
import random
from collections import deque

N = 12
DEADLINE = 80.0  # safety margin before 90s kill

def _bfs_threshold(ta, tb):
    """Compute exact reset threshold via BFS on subset automaton.
    ta, tb: tuples of length N (destination states for each state).
    Returns (threshold, shortest_word) or (None, None) if not synchronizing.
    """
    # Precompute transition for all 2^N subsets
    # Use bitmask representation: subset S is an int bitmask
    # trans_a[S] = image of S under letter a
    # trans_b[S] = image of S under letter b
    
    # Build transition tables
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        # Compute image under a
        img_a = 0
        img_b = 0
        for i in range(N):
            if mask & (1 << i):
                img_a |= (1 << ta[i])
                img_b |= (1 << tb[i])
        trans_a[mask] = img_a
        trans_b[mask] = img_b
    
    full = (1 << N) - 1
    
    # BFS from full set
    # State: (subset_mask, path)
    # We want shortest word that reaches a singleton
    visited = bytearray(1 << N)  # visited[subset] = 1 if seen
    visited[full] = 1
    
    # BFS queue: (mask, path_as_int)
    # Encode path as integer: 0 for 'a', 1 for 'b', least significant bit = first letter
    queue = deque()
    queue.append((full, 0))  # (subset, path_encoding)
    # We also need to track path length implicitly via BFS level
    
    # Actually, let's use a proper BFS with parent tracking for the word
    # But for just the threshold, we can do level-by-level BFS
    # and track the word for the first singleton found
    
    # For efficiency, store (mask, path) where path is encoded
    # path encoding: we append bits. Since max threshold is ~121, 
    # we need up to 121 bits. Use Python int for arbitrary precision.
    
    # Level-by-level BFS
    current_level = [(full, 0)]  # (mask, path_int)
    depth = 0
    max_depth = (N - 1) * (N - 1) + 5  # upper bound safety
    
    while current_level and depth < max_depth:
        next_level = []
        for mask, path in current_level:
            # Check if singleton
            if mask & (mask - 1) == 0 and mask != 0:
                # Decode path
                word = []
                p = path
                for _ in range(depth):
                    word.append(p & 1)
                    p >>= 1
                return (depth, word)
            
            # Try 'a'
            new_mask = trans_a[mask]
            if not visited[new_mask]:
                visited[new_mask] = 1
                next_level.append((new_mask, (path << 1) | 0))
            
            # Try 'b'
            new_mask = trans_b[mask]
            if not visited[new_mask]:
                visited[new_mask] = 1
                next_level.append((new_mask, (path << 1) | 1))
        
        current_level = next_level
        depth += 1
    
    return (None, None)


def _cherny_construction():
    """Classic Černý construction: cycle + fold.
    a: cycle 0->1->...->N-1->0
    b: 0->0, 1->0, ..., N-1->N-2 (fold: i -> max(0, i-1) for i>=1, 0->0)
    This gives threshold (N-1)^2.
    """
    a = list(range(1, N)) + [0]
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
    return tuple(a), tuple(b)


def _double_cycle_construction():
    """Double cycle: a is a 6-cycle repeated, b is a different 6-cycle.
    a: 0->1->2->3->4->5->0, 6->7->8->9->10->11->6
    b: 0->1->2->3->4->5->6->7->8->9->10->11->0 (full cycle)
    Actually let's try:
    a: cycle of length 6 on {0..5}, cycle of length 6 on {6..11}
    b: shifts between the two halves
    """
    # a: two disjoint 6-cycles
    a = []
    for i in range(6):
        a.append((i + 1) % 6)
    for i in range(6, 12):
        a.append((i - 6 + 1) % 6 + 6)
    
    # b: full 12-cycle
    b = [(i + 1) % 12 for i in range(12)]
    
    return tuple(a), tuple(b)


def _perturbed_cycle_search(a_base, b_base, num_perturbations, rng):
    """Try small perturbations of a base automaton."""
    best = (0, None, None)
    a = list(a_base)
    b = list(b_base)
    
    for _ in range(num_perturbations):
        # Perturb a few random transitions
        for _ in range(rng.randint(1, 3)):
            if rng.random() < 0.5:
                i = rng.randint(0, N - 1)
                a[i] = rng.randint(0, N - 1)
            else:
                i = rng.randint(0, N - 1)
                b[i] = rng.randint(0, N - 1)
        
        ta, tb = tuple(a), tuple(b)
        result = _bfs_threshold(ta, tb)
        if result[0] is not None and result[0] > best[0]:
            best = (result[0], ta, tb)
            # Reset to best found
            a = list(ta)
            b = list(tb)
    
    return best


def construct():
    start_time = time.time()
    
    # Best result so far
    best_threshold = 0
    best_a = None
    best_b = None
    
    # 1. Try classic Černý construction
    ta, tb = _cherny_construction()
    result = _bfs_threshold(ta, tb)
    if result[0] is not None and result[0] > best_threshold:
        best_threshold = result[0]
        best_a = ta
        best_b = tb
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return list(best_a), list(best_b)
    
    # 2. Try double cycle
    ta, tb = _double_cycle_construction()
    result = _bfs_threshold(ta, tb)
    if result[0] is not None and result[0] > best_threshold:
        best_threshold = result[0]
        best_a = ta
        best_b = tb
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return list(best_a), list(best_b)
    
    # 3. Try more structured families
    rng = random.Random(42)
    
    # Family: cycle with a single "defect"
    # a is a full cycle, b has one state mapped differently
    base_cycle = tuple([(i + 1) % N for i in range(N)])
    
    for defect_pos in range(N):
        for defect_target in range(N):
            b = list(base_cycle)
            b[defect_pos] = defect_target
            ta, tb = base_cycle, tuple(b)
            
            # Quick check: must be synchronizing
            result = _bfs_threshold(ta, tb)
            if result[0] is not None and result[0] > best_threshold:
                best_threshold = result[0]
                best_a = ta
                best_b = tb
            
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 10:
                return list(best_a), list(best_b)
    
    # 4. Family: near-cycle with fold
    # a: cycle, b: fold-like but with variations
    for shift in range(N):
        b = [0] * N
        for i in range(1, N):
            b[i] = (i - 1 + shift) % N
        b[0] = shift
        ta, tb = base_cycle, tuple(b)
        result = _bfs_threshold(ta, tb)
        if result[0] is not None and result[0] > best_threshold:
            best_threshold = result[0]
            best_a = ta
            best_b = tb
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            return list(best_a), list(best_b)
    
    # 5. Perturbation search around best found
    if best_a is not None:
        remaining = DEADLINE - (time.time() - start_time) - 5
        if remaining > 1:
            # Estimate how many BFS we can do
            # Each BFS takes roughly 0.5-2ms for N=12
            # Try as many as we can
            perturb_count = min(int(remaining * 500), 5000)  # ~2ms each
            best_t, best_ta, best_tb = _perturbed_cycle_search(
                tuple(best_a), tuple(best_b), perturb_count, rng
            )
            if best_t > best_threshold:
                best_threshold = best_t
                best_a = best_ta
                best_b = best_tb
    
    # 6. Try some random structured automata
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed - 3
    if remaining > 0.5 and best_a is not None:
        # Try random perturbations of the Černý construction
        cherny_a, cherny_b = _cherny_construction()
        for _ in range(min(int(remaining * 300), 3000)):
            a = list(cherny_a)
            b = list(cherny_b)
            # Perturb 1-2 transitions in b
            for _ in range(rng.randint(1, 2)):
                i = rng.randint(0, N - 1)
                b[i] = rng.randint(0, N - 1)
            
            ta, tb = tuple(a), tuple(b)
            result = _bfs_threshold(ta, tb)
            if result[0] is not None and result[0] > best_threshold:
                best_threshold = result[0]
                best_a = ta
                best_b = tb
            
            if time.time() - start_time > DEADLINE - 3:
                break
    
    # Ensure we have a valid result
    if best_a is None:
        # Fallback: simple synchronizing automaton
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        return a, b
    
    return list(best_a), list(best_b)
