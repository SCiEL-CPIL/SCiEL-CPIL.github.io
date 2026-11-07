"""Rewritten program using maximum weight independent set via branch-and-bound."""
# EVOLVE-BLOCK-START
from math import isqrt
from functools import lru_cache

# Precompute primes up to 256
def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_up_to(256)

def _prime_factors_set(n):
    """Return set of prime factors of n."""
    s = set()
    for p in _PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            s.add(p)
            while n % p == 0:
                n //= p
    if n > 1:
        s.add(n)
    return s

# Build conflict graph: edge between i and j if gcd(i,j) > 1
# We'll represent this as a list of sets: conflicts[i] = set of j that conflict with i
# Vertices are 1..256. Vertex 1 has no conflicts (gcd(1,x)=1 for all x).
N = 256
_conflicts = [set() for _ in range(N + 1)]
_pf = [set()] * (N + 1)
for i in range(1, N + 1):
    _pf[i] = _prime_factors_set(i)

for i in range(1, N + 1):
    for j in range(i + 1, N + 1):
        if _pf[i] & _pf[j]:  # share a prime factor
            _conflicts[i].add(j)
            _conflicts[j].add(i)

# Weight of each vertex is its value
_weight = list(range(N + 1))  # weight[i] = i

# For branch-and-bound, we need an upper bound on the sum of weights of
# an independent set that can still be added to the current partial solution.
# Greedy coloring upper bound: color the remaining candidate vertices such that
# no two adjacent vertices share a color. The sum of max weight per color gives
# an upper bound on the total additional weight.

def _greedy_color_bound(candidates, available):
    """
    Given a list of candidate vertex indices (all in 'available'),
    compute a greedy coloring upper bound on the sum of weights of
    an independent subset of candidates.
    
    available: set of vertices that are still allowed (not conflicting with chosen).
    candidates: list of vertices to consider (subset of available).
    Returns upper bound on sum of weights of independent set from candidates.
    """
    if not candidates:
        return 0
    
    # Order candidates by weight descending for better pruning
    cands = sorted(candidates, key=lambda v: -_weight[v])
    
    # Greedy coloring: assign colors so that no two adjacent (conflicting) vertices share a color
    color_of = {}
    color_max_weight = []  # color_max_weight[c] = max weight of vertex with color c
    
    for v in cands:
        # Find smallest color not used by any neighbor of v that has already been colored
        neighbor_colors = set()
        for u in _conflicts[v]:
            if u in color_of:
                neighbor_colors.add(color_of[u])
        
        c = 0
        while c in neighbor_colors:
            c += 1
        
        color_of[v] = c
        if c >= len(color_max_weight):
            color_max_weight.append(0)
        if _weight[v] > color_max_weight[c]:
            color_max_weight[c] = _weight[v]
    
    return sum(color_max_weight)


def _solve_mwis():
    """
    Solve maximum weight independent set on the conflict graph (vertices 1..256).
    Uses branch-and-bound with greedy coloring upper bound.
    """
    # Start with all vertices available
    all_vertices = list(range(1, N + 1))
    
    best_sum = 0
    best_set = []
    
    # We'll use a recursive branch-and-bound.
    # State: current chosen set, available set (vertices not conflicting with chosen).
    # To make it efficient, we process vertices in some order.
    
    # Alternative approach: use a well-known MWIS algorithm.
    # Let's use a recursive approach where at each step we pick a vertex,
    # branch on including or excluding it, and use the coloring bound to prune.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def branch(available, current_sum, current_set):
        nonlocal best_sum, best_set
        
        if not available:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(current_set)
            return
        
        # Upper bound: current_sum + greedy_color_bound(available)
        # But computing the full bound every time might be expensive.
        # Let's compute it.
        ub = current_sum + _greedy_color_bound(list(available), available)
        if ub <= best_sum:
            return
        
        # Pick a vertex to branch on. Choose the one with highest weight for better pruning.
        v = max(available, key=lambda x: _weight[x])
        
        # Branch 1: include v
        new_available = available - {v} - _conflicts[v]
        current_set.add(v)
        branch(new_available, current_sum + _weight[v], current_set)
        current_set.discard(v)
        
        # Branch 2: exclude v
        new_available = available - {v}
        branch(new_available, current_sum, current_set)
    
    branch(set(all_vertices), 0, set())
    return best_set


# Solve once at module load time
_SOLUTION = _solve_mwis()

def construct():
    """Return a list of pairwise-coprime integers from 1..256 maximizing their sum."""
    result = list(_SOLUTION)
    # Ensure it's a nonempty list of Python ints
    if not result:
        result = [1]
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
