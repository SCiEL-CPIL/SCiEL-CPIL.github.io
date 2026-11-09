"""Rewritten program for the Černý problem (N=12).

We construct a synchronizing automaton with 12 states and two letters that
maximizes the shortest reset word length. The evaluator will BFS the subset
automaton to find the exact threshold.

Strategy: Use a known construction that yields long reset words. The classic
Černý automaton has a cycle of N states with one letter and a second letter
that maps most states to a fixed point. For N=12, we can try several
constructions and pick the one with the longest reset word (verified by BFS).

We also try the "Pawlik" type construction and variants thereof, as well as
randomized search with local improvement, all within a time budget.
"""

import time
import random
from collections import deque

N = 12

def _bfs_reset_length(trans_a, trans_b):
    """BFS on subset automaton to find shortest reset word length.
    Returns (length, word) or (None, None) if not synchronizing."""
    start = frozenset(range(N))
    if len(start) == 1:
        return 0, ""
    visited = {start: None}  # subset -> (parent, char)
    queue = deque([start])
    while queue:
        subset = queue.popleft()
        # Try both letters
        for char, trans in enumerate([trans_a, trans_b]):
            new_subset = frozenset(trans[s] for s in subset)
            if new_subset not in visited:
                visited[new_subset] = (subset, char)
                if len(new_subset) == 1:
                    # Reconstruct word
                    length = 0
                    s = new_subset
                    word_chars = []
                    while s != start:
                        parent, c = visited[s]
                        word_chars.append(c)
                        s = parent
                        length += 1
                    word_chars.reverse()
                    return length, ''.join(str(c) for c in word_chars)
                queue.append(new_subset)
    return None, None

def _eval_automaton(trans_a, trans_b):
    """Return the reset word length, or -1 if not synchronizing."""
    length, _ = _bfs_reset_length(trans_a, trans_b)
    return length if length is not None else -1

def _cerny_automaton():
    """Classic Černý automaton: a is a cycle, b maps all but one to a fixed point."""
    trans_a = [(i + 1) % N for i in range(N)]
    trans_b = [0] * N
    return trans_a, trans_b

def _pawlik_automaton():
    """Pawlik-type automaton known for long reset words.
    For N states, uses a specific structure."""
    # A known construction: a is a cycle, b has a specific pattern
    # One well-known long-reset construction:
    trans_a = [(i + 1) % N for i in range(N)]
    # b maps: 0->0, 1->0, 2->1, 3->2, ..., N-1->N-2
    # This creates a "sliding" effect
    trans_b = [0] + list(range(N - 1))
    return trans_a, trans_b

def _random_automaton():
    """Generate a random complete binary deterministic automaton."""
    trans_a = [random.randint(0, N - 1) for _ in range(N)]
    trans_b = [random.randint(0, N - 1) for _ in range(N)]
    return trans_a, trans_b

def _local_search(start_a, start_b, deadline):
    """Hill-climbing: mutate one transition at a time, keep if it improves reset length."""
    best_a = list(start_a)
    best_b = list(start_b)
    best_len = _eval_automaton(best_a, best_b)
    if best_len < 0:
        # Not synchronizing; try to fix by making it synchronizing
        # Simple fix: make b constant
        best_b = [0] * N
        best_len = _eval_automaton(best_a, best_b)
    
    iterations = 0
    while time.time() < deadline - 5:
        iterations += 1
        # Pick a random transition to mutate
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        old_val = best_b[state] if letter == 1 else best_a[state]
        new_val = random.randint(0, N - 1)
        if new_val == old_val:
            continue
        
        if letter == 0:
            best_a[state] = new_val
        else:
            best_b[state] = new_val
        
        new_len = _eval_automaton(best_a, best_b)
        
        if new_len >= 0 and (best_len < 0 or new_len > best_len):
            best_len = new_len
        else:
            # Revert
            if letter == 0:
                best_a[state] = old_val
            else:
                best_b[state] = old_val
        
        if best_len >= 120:  # Near theoretical max for N=12
            break
    
    return best_a, best_b, best_len

def construct():
    """Construct a synchronizing automaton with N=12 states maximizing reset word length."""
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    best_a = None
    best_b = None
    best_len = -1
    
    # Try known constructions first
    candidates = [
        _cerny_automaton(),
        _pawlik_automaton(),
    ]
    
    # Add a few more structured candidates
    # Variant: b maps i -> (i-1) % N but with one exception
    for k in range(N):
        ta = [(i + 1) % N for i in range(N)]
        tb = [((i - 1) % N) for i in range(N)]
        tb[k] = 0
        candidates.append((ta, tb))
    
    # Evaluate all candidates
    for ta, tb in candidates:
        if time.time() > deadline - 10:
            break
        length = _eval_automaton(ta, tb)
        if length > best_len:
            best_len = length
            best_a = list(ta)
            best_b = list(tb)
    
    # If we have time, do local search from the best candidate
    if best_a is not None and time.time() < deadline - 15:
        sa, sb, sl = _local_search(best_a, best_b, deadline)
        if sl > best_len:
            best_a, best_b, best_len = sa, sb, sl
    
    # Also try a few random starts with local search if time permits
    if time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            ra, rb = _random_automaton()
            # Ensure it's synchronizing by checking
            rl = _eval_automaton(ra, rb)
            if rl < 0:
                # Make it synchronizing: set b to constant
                rb = [0] * N
                rl = _eval_automaton(ra, rb)
            if rl > best_len:
                best_a, best_b, best_len = list(ra), list(rb), rl
            # Local search from this start
            if time.time() < deadline - 15:
                sa, sb, sl = _local_search(ra, rb, deadline)
                if sl > best_len:
                    best_a, best_b, best_len = sa, sb, sl
    
    # Final safety: if we don't have a valid automaton, use the Černý one
    if best_a is None or best_len < 0:
        best_a, best_b = _cerny_automaton()
    
    return best_a, best_b

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
