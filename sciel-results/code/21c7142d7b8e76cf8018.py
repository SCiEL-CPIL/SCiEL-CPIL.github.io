"""
Construct a 12-state synchronizing automaton maximizing the shortest reset word length.

Strategy:
1. Start from the Černý construction which achieves (N-1)^2 = 121 for N=12.
2. The Černý automaton:
   - Letter 'a' is a cycle: 0->1->2->...->11->0
   - Letter 'b' maps all states to 0, except state 1 maps to 1 (or similar non-injective map)
   
   Actually, the standard Černý construction for N states:
   - a: i -> (i+1) mod N for all i
   - b: 0 -> 0, 1 -> 0, 2 -> 1, ..., N-1 -> N-2
   
   This gives reset word length (N-1)^2.

3. Verify synchronizability and compute exact threshold via BFS on subset automaton.
4. If the Černý construction works, return it. Otherwise, fall back to local search.

For N=12, the Černý construction should give threshold 121.
"""

import time
from collections import deque

N = 12


def _compute_threshold(a_map, b_map):
    """
    Compute the exact shortest reset word length using BFS on the subset automaton.
    Returns (threshold_length, shortest_word) or (-1, None) if not synchronizing.
    """
    # Full set of all states
    full_set = frozenset(range(N))
    
    # BFS from full_set
    # State: frozenset of states
    # We want to reach a singleton
    
    visited = {full_set: (0, "")}
    queue = deque([full_set])
    
    while queue:
        current = queue.popleft()
        dist, word = visited[current]
        
        if len(current) == 1:
            return dist, word
        
        # Try appending 'a' (0)
        next_a = frozenset(a_map[s] for s in current)
        if next_a not in visited:
            visited[next_a] = (dist + 1, word + "0")
            queue.append(next_a)
        
        # Try appending 'b' (1)
        next_b = frozenset(b_map[s] for s in current)
        if next_b not in visited:
            visited[next_b] = (dist + 1, word + "1")
            queue.append(next_b)
    
    return -1, None


def _cherny_construction(n):
    """
    Standard Černý construction for n states.
    a: cycle 0->1->...->n-1->0
    b: 0->0, 1->0, 2->1, ..., n-1->n-2
    """
    a = [(i + 1) % n for i in range(n)]
    b = [0] * n
    for i in range(2, n):
        b[i] = i - 1
    # b[0] = 0, b[1] = 0
    return a, b


def construct():
    """
    Return two length-N lists defining a complete binary deterministic automaton
    with N=12 states that synchronizes, maximizing the shortest reset word length.
    """
    deadline = time.time() + 75  # Safety margin before 90s kill
    
    # Try the Černý construction first
    a, b = _cherny_construction(N)
    threshold, _ = _compute_threshold(a, b)
    
    if threshold > 0:
        return [a, b]
    
    # If Černý construction doesn't work (shouldn't happen for N=12), 
    # try local search / hill climbing
    import random
    random.seed(42)
    
    best_threshold = -1
    best_a, best_b = None, None
    
    # Start from a random synchronizing automaton or the cycle+constant
    a_curr = [(i + 1) % N for i in range(N)]
    b_curr = [0] * N
    
    # Try to improve via local search
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        # Evaluate current
        th, _ = _compute_threshold(a_curr, b_curr)
        if th > best_threshold:
            best_threshold = th
            best_a, best_b = a_curr[:], b_curr[:]
        
        if best_threshold >= 121:
            break
        
        # Random mutation: change one transition
        choice = random.random()
        if choice < 0.5:
            # Mutate a
            idx = random.randint(0, N - 1)
            old_val = a_curr[idx]
            a_curr[idx] = random.randint(0, N - 1)
            # Ensure it's still a function (it is, since we're just changing one output)
        else:
            # Mutate b
            idx = random.randint(0, N - 1)
            b_curr[idx] = random.randint(0, N - 1)
        
        # Occasionally revert to avoid getting stuck
        if random.random() < 0.1:
            # Revert last mutation
            if choice < 0.5:
                a_curr[idx] = old_val
            else:
                # We didn't save old_val for b, so just continue
                pass
    
    if best_a is not None:
        return [best_a, best_b]
    
    # Fallback: return the simple construction
    return [[(i + 1) % N for i in range(N)], [0] * N]
