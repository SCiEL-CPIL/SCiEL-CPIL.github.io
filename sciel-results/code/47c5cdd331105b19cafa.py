"""
Search for a 12-state binary deterministic automaton with maximum reset threshold.
The program performs a randomized search over transition tables, using BFS to
compute the exact shortest reset word length for candidate automata.
A deadline guard ensures the program returns within the time limit.
"""

import time
import random
import collections
import itertools

N = 12

def _bfs_reset_length(trans_a, trans_b):
    """
    Compute the exact shortest reset word length for a binary automaton.
    Returns (length, word) where word is a string of 'a' and 'b'.
    Returns (None, None) if the automaton is not synchronizing.
    """
    # A subset is represented as a frozenset of states, or as an integer bitmask for speed.
    # For N=12, we can use integer bitmasks (12 bits).
    
    def apply(trans, mask):
        """Apply transition letter (trans is a list of destinations) to a bitmask."""
        result = 0
        for i in range(N):
            if mask & (1 << i):
                result |= (1 << trans[i])
        return result
    
    full_mask = (1 << N) - 1
    
    # BFS from full set of states
    # State: (mask, word)
    # We want to reach a singleton (mask with exactly one bit set)
    
    # Use a dict to track shortest path to each mask
    # parent: mask -> (prev_mask, letter)
    parent = {}
    queue = collections.deque()
    queue.append(full_mask)
    parent[full_mask] = (None, None)
    
    # Precompute transitions for all masks? That's 2^12 = 4096 masks, each with 2 transitions.
    # That's fast.
    
    while queue:
        mask = queue.popleft()
        
        # Check if this is a singleton
        if mask & (mask - 1) == 0:  # power of 2
            # Reconstruct word
            word = []
            cur = mask
            while parent[cur][0] is not None:
                prev, letter = parent[cur]
                word.append(letter)
                cur = prev
            word.reverse()
            return len(word), ''.join(word)
        
        for letter, trans in [('a', trans_a), ('b', trans_b)]:
            new_mask = apply(trans, mask)
            if new_mask not in parent:
                parent[new_mask] = (mask, letter)
                queue.append(new_mask)
    
    # Not synchronizing
    return None, None


def _random_automaton(rng):
    """Generate a random complete binary deterministic automaton."""
    trans_a = [rng.randint(0, N - 1) for _ in range(N)]
    trans_b = [rng.randint(0, N - 1) for _ in range(N)]
    return trans_a, trans_b


def _mutate_automaton(trans_a, trans_b, rng, mutation_rate=0.1):
    """Mutate an automaton by changing some transitions."""
    new_a = list(trans_a)
    new_b = list(trans_b)
    for i in range(N):
        if rng.random() < mutation_rate:
            new_a[i] = rng.randint(0, N - 1)
        if rng.random() < mutation_rate:
            new_b[i] = rng.randint(0, N - 1)
    return new_a, new_b


def construct():
    """
    Construct a 12-state binary deterministic automaton that synchronizes,
    maximizing the length of its shortest reset word.
    Returns two length-N lists of destination states in 0..N-1.
    """
    start_time = time.time()
    deadline = start_time + 75  # 75 second deadline (5 sec safety margin)
    
    rng = random.Random(42)
    
    best_length = 0
    best_a = None
    best_b = None
    
    # Initial population: try some structured automata first
    # 1. Cycle plus map to zero (seed)
    a0 = [(i + 1) % N for i in range(N)]
    b0 = [0] * N
    length, _ = _bfs_reset_length(a0, b0)
    if length is not None and length > best_length:
        best_length = length
        best_a = list(a0)
        best_b = list(b0)
    
    # 2. Try some known good structures
    # De Groot type: one letter is a cycle, other is a map that breaks the cycle
    for offset in range(1, N):
        a1 = [(i + offset) % N for i in range(N)]
        b1 = [0] * N
        length, _ = _bfs_reset_length(a1, b1)
        if length is not None and length > best_length:
            best_length = length
            best_a = list(a1)
            best_b = list(b1)
    
    # 3. Try random automata
    max_iter = 50000
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate a random automaton
        trans_a, trans_b = _random_automaton(rng)
        
        length, _ = _bfs_reset_length(trans_a, trans_b)
        if length is not None and length > best_length:
            best_length = length
            best_a = list(trans_a)
            best_b = list(trans_b)
    
    # 4. Local search around the best found
    if best_a is not None and best_b is not None:
        local_start = time.time()
        local_deadline = deadline
        for it in range(20000):
            if time.time() > local_deadline:
                break
            
            # Mutate the best automaton
            mut_a, mut_b = _mutate_automaton(best_a, best_b, rng, mutation_rate=0.05)
            
            length, _ = _bfs_reset_length(mut_a, mut_b)
            if length is not None and length > best_length:
                best_length = length
                best_a = list(mut_a)
                best_b = list(mut_b)
            elif length is not None and length == best_length:
                # Keep with some probability (to explore)
                if rng.random() < 0.1:
                    best_a = list(mut_a)
                    best_b = list(mut_b)
    
    if best_a is None:
        # Fallback: return the seed automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    return best_a, best_b


# EVOLVE-BLOCK-START
# (construct is defined above; this block is kept for compatibility)
# EVOLVE-BLOCK-END
