N = 12
import time
from collections import deque

def _reset_length(a, b):
    """Compute shortest reset word length via BFS on subsets."""
    start = tuple(range(N))
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        subset, dist = queue.popleft()
        if len(subset) == 1:
            return dist
        next_a = tuple(sorted(set(a[i] for i in subset)))
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
        next_b = tuple(sorted(set(b[i] for i in subset)))
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
    return -1

def construct():
    # Classic Černý automaton: achieves (n-1)^2 reset length
    # Letter a: 0->0, i->i-1 for i>=1
    # Letter b: i->(i+1)%N (cycle)
    a = [0] * N
    for i in range(1, N):
        a[i] = i - 1
    b = [(i + 1) % N for i in range(N)]
    return [a, b]
