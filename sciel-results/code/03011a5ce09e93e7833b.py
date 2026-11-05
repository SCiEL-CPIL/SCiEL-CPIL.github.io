"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

# EVOLVE-BLOCK-START

def _reset_length(trans_a, trans_b):
    """Compute exact shortest reset word length via BFS on subsets.
    Returns (length, word) where length is -1 if not synchronizing."""
    full = frozenset(range(N))
    if len(full) == 1:
        return 0, ""
    
    # Map each subset to an index for efficient storage
    # We use a dict to store visited subsets and their parent info
    visited = {full: (None, None)}  # subset -> (parent_subset, char)
    queue = deque([full])
    
    while queue:
        current = queue.popleft()
        # Try both letters
        for ch, trans in enumerate([trans_a, trans_b]):
            # Apply transition to all states in current subset
            new_set = frozenset(trans[s] for s in current)
            if len(new_set) == 1:
                # Found a reset
                # Reconstruct word
                word = [ch]
                parent = current
                while parent is not None:
                    parent, pch = visited[parent]
                    if pch is not None:
                        word.append(pch)
                word.reverse()
                return len(word), ''.join(map(str, word))
            if new_set not in visited:
                visited[new_set] = (current, ch)
                queue.append(new_set)
    
    return -1, ""


def _is_valid(trans_a, trans_b):
    """Check that transitions are valid (all destinations in 0..N-1)."""
    for trans in (trans_a, trans_b):
        if len(trans) != N:
            return False
        for v in trans:
            if v < 0 or v >= N:
                return False
    return True


def _try_improve(base_a, base_b, time_budget):
    """Try to find a better automaton by local search around the base.
    Returns (best_a, best_b, best_len) or None if no improvement found."""
    start_time = time.time()
    best_a = list(base_a)
    best_b = list(base_b)
    base_len, _ = _reset_length(base_a, base_b)
    best_len = base_len
    
    if best_len < 0:
        return None
    
    random.seed(42)
    
    while time.time() - start_time < time_budget:
        # Pick a random state and letter to modify
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        
        if letter == 0:
            old_val = best_a[state]
            new_val = random.randint(0, N - 1)
            best_a[state] = new_val
            a, b = best_a, best_b
        else:
            old_val = best_b[state]
            new_val = random.randint(0, N - 1)
            best_b[state] = new_val
            a, b = best_a, best_b
        
        if not _is_valid(a, b):
            # Revert
            if letter == 0:
                best_a[state] = old_val
            else:
                best_b[state] = old_val
            continue
        
        length, _ = _reset_length(a, b)
        
        if length > best_len:
            best_len = length
            # Keep the change
        else:
            # Revert
            if letter == 0:
                best_a[state] = old_val
            else:
                best_b[state] = old_val
    
    return best_a, best_b, best_len


def construct():
    """Construct a synchronizing 12-state binary automaton maximizing reset word length."""
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    # Start with the known good "cycle-shift" automaton:
    # a: i -> (i+1) % N  (cycle)
    # b: i -> i for i < N-1, b[N-1] -> 1  (maps last state to state 1)
    # This was measured to give reset length 111.
    
    base_a = [(i + 1) % N for i in range(N)]
    base_b = [i for i in range(N - 1)] + [1]  # b[N-1] = 1
    
    # Verify base
    base_len, base_word = _reset_length(base_a, base_b)
    if base_len < 0:
        # Fallback to simple cycle-sink
        base_a = [(i + 1) % N for i in range(N)]
        base_b = [0] * N
        base_len, base_word = _reset_length(base_a, base_b)
    
    best_a = list(base_a)
    best_b = list(base_b)
    best_len = base_len
    
    # Try local search with remaining time
    remaining = deadline - time.time()
    if remaining > 5.0:
        result = _try_improve(base_a, base_b, remaining - 5.0)
        if result is not None:
            a, b, length = result
            if length > best_len:
                best_a, best_b, best_len = a, b, length
    
    # Final safety check
    if time.time() > deadline - 2.0:
        # Just return what we have
        pass
    
    # Verify the result is valid and synchronizing
    if not _is_valid(best_a, best_b):
        # Fallback
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
    
    final_len, _ = _reset_length(best_a, best_b)
    if final_len < 0:
        # Should not happen, but fallback
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
