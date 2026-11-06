"""Improved program: min-degree greedy with randomized restarts for max independent set on conflict graph."""
import time
import random
import math

# Precompute conflict structure
# Two elements a, b (a <= b) conflict if a+b is a perfect square.
# We need a subset S of [1,256] such that for all a, b in S with a <= b, a+b is NOT a perfect square.
# This includes a=b, so 2a must not be a perfect square.

N = 256
elements = list(range(1, N + 1))

# Precompute which pairs conflict
# conflict[a][b] = True if a+b is a perfect square
# Use 1-indexed: index 1..256
# For speed, precompute a set of conflicting pairs or adjacency lists

def is_perfect_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

# Build adjacency: adj[i] = set of j such that i+j is a perfect square (for i != j or i==j)
# Actually we need: for a<=b, a+b not a perfect square.
# So conflict between a and b (a<=b) if a+b is perfect square.
# Self-conflict: 2a is perfect square -> a must be excluded.

# Precompute self-conflicts
self_conflict = set()
for a in elements:
    if is_perfect_square(2 * a):
        self_conflict.add(a)

# Precompute adjacency lists for the conflict graph
# adj[i] = set of j (j != i) such that i+j is a perfect square
adj = [set() for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(i + 1, N + 1):
        if is_perfect_square(i + j):
            adj[i].add(j)
            adj[j].add(i)

# Also, elements in self_conflict cannot be in any valid set
available = [a for a in elements if a not in self_conflict]

def greedy_min_degree(available, adj, rng):
    """Greedy: repeatedly pick the available element with minimum degree among available, add it, remove its neighbors."""
    if not available:
        return []
    
    # Work with a mutable copy of available set
    avail = set(available)
    result = []
    
    while avail:
        # Find element with minimum degree in current available set
        min_deg = float('inf')
        min_elem = None
        for v in avail:
            deg = len(adj[v] & avail)
            if deg < min_deg:
                min_deg = deg
                min_elem = v
            elif deg == min_deg and min_elem is not None:
                # Tie-break randomly
                if rng.random() < 0.5:
                    min_elem = v
        result.append(min_elem)
        # Remove min_elem and its neighbors from avail
        avail.discard(min_elem)
        for neighbor in adj[min_elem]:
            avail.discard(neighbor)
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    best = []
    best_size = 0
    
    # First, try a deterministic greedy to get a baseline
    baseline = greedy_min_degree(available, adj, random.Random(0))
    best = baseline
    best_size = len(baseline)
    
    # Now do randomized restarts
    rng = random.Random(42)
    restart = 0
    while time.time() < deadline:
        # Randomize the available set ordering or use random tie-breaking
        # Strategy: shuffle available, then do greedy with random tie-breaking
        shuffled = available[:]
        rng.shuffle(shuffled)
        
        result = greedy_min_degree(shuffled, adj, rng)
        if len(result) > best_size:
            best_size = len(result)
            best = result
            # If we've found a very good solution, we can be more aggressive with restarts
            # but keep going to see if we can do better
        
        restart += 1
    
    return sorted(best)
