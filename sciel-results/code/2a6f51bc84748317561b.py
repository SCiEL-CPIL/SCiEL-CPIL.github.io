"""
Černý problem: 12-state binary automata with long reset thresholds.

Strategy:
- Start from a known high-threshold construction (or a good random seed).
- Use simulated annealing to mutate transition tables.
- Fast evaluation via BFS on the subset automaton (memoized subset transitions).
- Time-bounded: return best found before deadline.
"""

import random
import time
import sys

N = 12

# Deadline guard: must finish within 60s (evaluator kills at 90s, but we target 60s)
_DEADLINE_SECONDS = 55.0


def _subset_transition(subset, trans_table):
    """Apply a letter to a subset of states. Returns frozenset of new states."""
    # trans_table is a list of two lists: trans_table[0][i] = state after 'a' from i,
    # trans_table[1][i] = state after 'b' from i.
    # But we need to know which letter. Let's restructure.
    
    pass


def _bfs_reset_length(trans_a, trans_b):
    """
    Compute the shortest reset word length for the automaton defined by
    trans_a (list of N ints) and trans_b (list of N ints).
    Returns the length, or None if not synchronizing.
    Uses BFS on the subset automaton.
    """
    # A subset is represented as a frozenset of state indices, or as a bitmask (int).
    # Using bitmask for speed: bit i set means state i is in the subset.
    
    # Initial subset: all states
    full_mask = (1 << N) - 1
    
    # Target: any singleton
    # We'll BFS from full_mask, looking for any singleton.
    
    from collections import deque
    
    # Precompute subset transitions for each letter
    # For each subset (bitmask), compute where 'a' and 'b' take it.
    # There are 2^N = 4096 subsets, so we can precompute all.
    
    # trans_a_mask[subset_mask] = new_mask after applying 'a'
    # trans_b_mask[subset_mask] = new_mask after applying 'b'
    
    trans_a_mask = [0] * (1 << N)
    trans_b_mask = [0] * (1 << N)
    
    for mask in range(1 << N):
        ma = 0
        mb = 0
        for i in range(N):
            if mask & (1 << i):
                ma |= (1 << trans_a[i])
                mb |= (1 << trans_b[i])
        trans_a_mask[mask] = ma
        trans_b_mask[mask] = mb
    
    # BFS
    visited = set()
    visited.add(full_mask)
    queue = deque()
    queue.append((full_mask, 0))
    
    while queue:
        mask, depth = queue.popleft()
        
        # Check if singleton
        if mask & (mask - 1) == 0:
            return depth
        
        # Try 'a'
        next_a = trans_a_mask[mask]
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, depth + 1))
        
        # Try 'b'
        next_b = trans_b_mask[mask]
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, depth + 1))
    
    return None  # Not synchronizing


def _make_random_trans():
    """Generate a random complete deterministic automaton."""
    trans_a = [random.randint(0, N - 1) for _ in range(N)]
    trans_b = [random.randint(0, N - 1) for _ in range(N)]
    return trans_a, trans_b


def _mutate_trans(trans_a, trans_b, mutation_rate=0.3):
    """Mutate the transition tables by randomly changing some entries."""
    new_a = list(trans_a)
    new_b = list(trans_b)
    for i in range(N):
        if random.random() < mutation_rate:
            new_a[i] = random.randint(0, N - 1)
        if random.random() < mutation_rate:
            new_b[i] = random.randint(0, N - 1)
    return new_a, new_b


def construct():
    """
    Construct a synchronizing automaton with 12 states and two input letters,
    maximizing the shortest reset word length.
    
    Returns: (trans_a, trans_b) where trans_a and trans_b are lists of length N
    with values in 0..N-1.
    """
    start_time = time.time()
    
    # Start with a good initial guess
    # A known construction: use a "Pinsker" type or random search
    # Let's start with a random automaton and then anneal
    
    # Initial: try a few random automata to find a synchronizing one
    best_a, best_b = None, None
    best_length = -1
    
    # Try to find an initial synchronizing automaton
    for _ in range(100):
        ta, tb = _make_random_trans()
        length = _bfs_reset_length(ta, tb)
        if length is not None:
            best_a, best_b = ta, tb
            best_length = length
            break
    
    if best_a is None:
        # Fallback: use the simple cycle + reset
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        best_length = _bfs_reset_length(best_a, best_b)
    
    if best_length is None:
        best_length = 0
    
    # Simulated annealing
    current_a, current_b = list(best_a), list(best_b)
    current_length = best_length
    
    temperature = 1.0
    cooling = 0.9995
    
    while time.time() - start_time < _DEADLINE_SECONDS:
        # Mutate
        new_a, new_b = _mutate_trans(current_a, current_b, mutation_rate=0.2)
        new_length = _bfs_reset_length(new_a, new_b)
        
        if new_length is None:
            # Not synchronizing; reject
            continue
        
        # Acceptance criterion
        if new_length > current_length:
            # Better: always accept
            current_a, current_b = new_a, new_b
            current_length = new_length
            if current_length > best_length:
                best_a, best_b = list(current_a), list(current_b)
                best_length = current_length
        else:
            # Worse or equal: accept with probability exp((new - current) / temperature)
            # Since we want to maximize, delta = new_length - current_length <= 0
            delta = new_length - current_length
            if temperature > 0:
                prob = pow(2.718281828, delta / temperature)
                if random.random() < prob:
                    current_a, current_b = new_a, new_b
                    current_length = new_length
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    return best_a, best_b


# Pre-compute for speed: we can cache BFS results for frequently seen automata
# But since each automaton is unique, caching may not help much.
# Instead, let's optimize the BFS by using arrays instead of sets for visited.


def _bfs_reset_length_fast(trans_a, trans_b):
    """
    Faster BFS using arrays for visited and queue.
    """
    full_mask = (1 << N) - 1
    num_subsets = 1 << N
    
    # Precompute transitions
    trans_a_mask = [0] * num_subsets
    trans_b_mask = [0] * num_subsets
    
    for mask in range(num_subsets):
        ma = 0
        mb = 0
        for i in range(N):
            if mask & (1 << i):
                ma |= (1 << trans_a[i])
                mb |= (1 << trans_b[i])
        trans_a_mask[mask] = ma
        trans_b_mask[mask] = mb
    
    # BFS with array-based visited
    visited = bytearray(num_subsets)  # 0 = unvisited, 1 = visited
    visited[full_mask] = 1
    
    # Queue: use a list with index pointer for efficiency
    queue = [full_mask]
    depths = [0]
    head = 0
    
    while head < len(queue):
        mask = queue[head]
        depth = depths[head]
        head += 1
        
        # Check if singleton
        if mask & (mask - 1) == 0:
            return depth
        
        # Try 'a'
        next_a = trans_a_mask[mask]
        if not visited[next_a]:
            visited[next_a] = 1
            queue.append(next_a)
            depths.append(depth + 1)
        
        # Try 'b'
        next_b = trans_b_mask[mask]
        if not visited[next_b]:
            visited[next_b] = 1
            queue.append(next_b)
            depths.append(depth + 1)
    
    return None


def construct():
    """
    Construct a synchronizing automaton with 12 states and two input letters,
    maximizing the shortest reset word length.
    
    Returns: (trans_a, trans_b) where trans_a and trans_b are lists of length N
    with values in 0..N-1.
    """
    start_time = time.time()
    
    best_a, best_b = None, None
    best_length = -1
    
    # Try to find an initial synchronizing automaton
    for _ in range(200):
        ta, tb = _make_random_trans()
        length = _bfs_reset_length_fast(ta, tb)
        if length is not None:
            best_a, best_b = ta, tb
            best_length = length
            break
    
    if best_a is None:
        # Fallback: use the simple cycle + reset
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        best_length = _bfs_reset_length_fast(best_a, best_b)
        if best_length is None:
            best_length = 0
    
    # Simulated annealing
    current_a, current_b = list(best_a), list(best_b)
    current_length = best_length
    
    temperature = 1.0
    cooling = 0.999
    
    # Track time for early exit
    while time.time() - start_time < _DEADLINE_SECONDS:
        # Mutate
        new_a, new_b = _mutate_trans(current_a, current_b, mutation_rate=0.15)
        new_length = _bfs_reset_length_fast(new_a, new_b)
        
        if new_length is None:
            continue
        
        if new_length > current_length:
            current_a, current_b = new_a, new_b
            current_length = new_length
            if current_length > best_length:
                best_a, best_b = list(current_a), list(current_b)
                best_length = current_length
        else:
            delta = new_length - current_length
            if temperature > 0.01:
                prob = pow(2.718281828, delta / temperature)
                if random.random() < prob:
                    current_a, current_b = new_a, new_b
                    current_length = new_length
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    return best_a, best_b
