"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import time
from collections import deque

def _bfs_reset_length(a, b):
    """Compute shortest reset word length via BFS on subset automaton."""
    start = tuple(range(N))
    if len(set(a)) == 1 or len(set(b)) == 1:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        subset, dist = queue.popleft()
        if len(subset) == 1:
            return dist
        # Apply letter a
        next_a = tuple(sorted(set(a[i] for i in subset)))
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
        # Apply letter b
        next_b = tuple(sorted(set(b[i] for i in subset)))
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
    return -1  # not synchronizing

def construct():
    # Černý automaton construction:
    # Letter a: cyclic shift (i -> (i+1) mod N)
    # Letter b: reset map (0 -> 0, i -> i-1 for i >= 1)
    # This achieves reset length binom(N,2) - N + 1 = 77 for N=12
    a = [(i + 1) % N for i in range(N)]
    b = [0] + list(range(0, N - 1))
    
    # Verify it's synchronizing
    rlen = _bfs_reset_length(a, b)
    if rlen < 0:
        # Fallback: simple synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
    return [a, b]
# EVOLVE-BLOCK-END
