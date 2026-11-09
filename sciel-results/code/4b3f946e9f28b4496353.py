"""Program for the suite task; finds largest set of distinct integers in 2..256
such that no three distinct members a,b,c satisfy a | b*c.

The problem is equivalent to finding a maximum independent set in a conflict
graph where edge (i,j) exists if i | j or j | i (pairwise divisibility conflict).
"""
from math import isqrt

def _prime_factors(n):
    """Return the set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

# Precompute prime factors for each number in 2..256
_N = 256
_NUMS = list(range(2, _N + 1))
_FACTORS = {x: _prime_factors(x) for x in _NUMS}
_NN = len(_NUMS)

# Build conflict graph: conflict[i][j] = True if _NUMS[i] and _NUMS[j] share a prime factor
conflict = [[False] * _NN for _ in range(_NN)]
for i in range(_NN):
    fi = _FACTORS[_NUMS[i]]
    for j in range(i + 1, _NN):
        fj = _FACTORS[_NUMS[j]]
        if fi & fj:
            conflict[i][j] = True
            conflict[j][i] = True

# Degree of each vertex
_degree = [sum(conflict[i]) for i in range(_NN)]

# For fast neighbor lookup
_neighbors = [set() for _ in range(_NN)]
for i in range(_NN):
    for j in range(_NN):
        if i != j and conflict[i][j]:
            _neighbors[i].add(j)

# Greedy upper bound using greedy coloring
def _greedy_upper_bound(candidates):
    """Upper bound on the size of an independent set within candidates.
    Uses greedy coloring: the number of colors used is an upper bound."""
    if not candidates:
        return 0
    color = {}
    max_color = 0
    # Sort by degree descending for tighter bounds
    for v in sorted(candidates, key=lambda x: -_degree[x]):
        used = set()
        for u in candidates:
            if u != v and conflict[v][u] and u in color:
                used.add(color[u])
        c = 0
        while c in used:
            c += 1
        color[v] = c
        if c > max_color:
            max_color = c
    return max_color

_best = []
_best_size = 0

def _branch_and_bound(candidates, current):
    """Branch and bound for maximum independent set."""
    global _best, _best_size
    if not candidates:
        if len(current) > _best_size:
            _best_size = len(current)
            _best = list(current)
        return
    # Upper bound
    ub = len(current) + _greedy_upper_bound(candidates)
    if ub <= _best_size:
        return
    # Pick vertex with highest degree in candidate subgraph
    v = max(candidates, key=lambda x: sum(1 for u in candidates if u != x and conflict[x][u]))
    # Branch 1: include v
    current.append(v)
    new_candidates = [u for u in candidates if u != v and not conflict[v][u]]
    _branch_and_bound(new_candidates, current)
    current.pop()
    # Branch 2: exclude v
    new_candidates2 = [u for u in candidates if u != v]
    _branch_and_bound(new_candidates2, current)

def construct():
    global _best, _best_size
    # Initialize with a greedy solution to have a good lower bound
    greedy = []
    remaining = list(range(_NN))
    while remaining:
        v = min(remaining, key=lambda x: sum(1 for u in remaining if u != x and conflict[x][u]))
        greedy.append(v)
        remaining = [u for u in remaining if u != v and not conflict[v][u]]
    _best = list(greedy)
    _best_size = len(greedy)
    
    # Run branch and bound to find the optimal
    all_candidates = list(range(_NN))
    _branch_and_bound(all_candidates, [])
    
    # Convert indices back to values
    result = [_NUMS[i] for i in _best]
    return result
