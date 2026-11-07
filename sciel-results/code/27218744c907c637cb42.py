<<<<<<< SEARCH
"""
Maximize sum of pairwise-coprime interval integers from 1..256.

The optimal set consists of 1 and, for every prime p <= 256, the largest power p^k <= 256.
These numbers are pairwise coprime because each has a unique prime factor (the base p).
Any composite number formed by multiplying primes from this set would be less than the
sum of their individual maximum powers (for small primes) or impossible (for large primes
where p*q > 256).
Thus, this greedy approach yields the global maximum.
"""

import time

def _sieve(n):
    """Return list of primes up to n."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def _largest_prime_power(p, n):
    """Return the largest power of p that is <= n."""
    val = p
    while val * p <= n:
        val *= p
    return val

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
    n = 256
    
    # Find all primes up to n
    primes = _sieve(n)
    
    # For each prime, find the largest power p^k <= n
    selected = []
    for p in primes:
        val = _largest_prime_power(p, n)
        selected.append(val)
        
    # Include 1, which is coprime with all integers
    result = [1] + selected
    
    # Ensure types are strictly Python int
    result = [int(x) for x in result]
    
    return result
=======
"""
Maximize sum of pairwise-coprime interval integers from 1..256.

Uses branch-and-bound on the conflict graph where vertices are 2..256 and
edges connect non-coprime pairs. A greedy coloring provides a tight upper
bound for pruning. The number 1 is always included since it's coprime with all.
"""

import time
import math
import sys

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _build_conflict_graph(n):
    """Build adjacency lists for the conflict graph on vertices 2..n."""
    # Map number to index: number x -> index x-2
    num_vertices = n - 1  # vertices for 2..n
    adj = [set() for _ in range(num_vertices)]
    
    for i in range(2, n + 1):
        idx_i = i - 2
        for j in range(i + 1, n + 1):
            idx_j = j - 2
            if _gcd(i, j) != 1:
                adj[idx_i].add(idx_j)
                adj[idx_j].add(idx_i)
    
    return adj

def _greedy_coloring(vertices, adj):
    """
    Greedy coloring of the subgraph induced by vertices.
    Returns list of colors for each vertex in vertices.
    """
    if not vertices:
        return []
    
    # Map vertex index to position in vertices list
    vert_set = set(vertices)
    color_of = {}
    color_count = 0
    
    # Order vertices by degree in the induced subgraph (descending)
    def degree_in_subgraph(v):
        return len(adj[v] & vert_set)
    
    ordered = sorted(vertices, key=degree_in_subgraph, reverse=True)
    
    for v in ordered:
        # Find the smallest color not used by neighbors
        neighbor_colors = set()
        for u in adj[v]:
            if u in vert_set and u in color_of:
                neighbor_colors.add(color_of[u])
        
        c = 0
        while c in neighbor_colors:
            c += 1
        color_of[v] = c
        color_count = max(color_count, c + 1)
    
    return color_count

def _branch_and_bound(adj, weights, deadline):
    """
    Branch and bound for maximum weight independent set.
    adj: list of sets (adjacency lists)
    weights: list of weights for each vertex
    Returns (best_weight, best_set)
    """
    n_vertices = len(adj)
    
    # Initial solution: greedy by weight/degree ratio
    def greedy_initial():
        selected = set()
        remaining = set(range(n_vertices))
        total_w = 0
        
        # Try multiple orderings
        best_sel = set()
        best_w = 0
        
        # Order by weight descending
        order = sorted(remaining, key=lambda v: -weights[v])
        for v in order:
            if all(u not in selected for u in adj[v]):
                selected.add(v)
                total_w += weights[v]
        if total_w > best_w:
            best_w = total_w
            best_sel = set(selected)
        
        # Order by weight/degree descending
        selected = set()
        total_w = 0
        order = sorted(remaining, key=lambda v: -weights[v] / (1 + len(adj[v])))
        for v in order:
            if all(u not in selected for u in adj[v]):
                selected.add(v)
                total_w += weights[v]
        if total_w > best_w:
            best_w = total_w
            best_sel = set(selected)
        
        return best_w, best_sel
    
    best_weight, best_set = greedy_initial()
    
    # Precompute weight sums for quick upper bounds
    # Use recursive branch and bound with coloring bound
    
    def bb_search(candidates, current_weight, current_set):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(current_set)
            return
        
        # Upper bound: current_weight + sum of weights of all candidates
        # Better: use greedy coloring bound
        cand_list = list(candidates)
        color_bound = _greedy_coloring(cand_list, adj)
        
        # The sum of the top `color_bound` weights among candidates is an upper bound
        # Actually, a tighter bound: sum of max weight per color class
        # For simplicity, use: current_weight + sum of all candidate weights
        # But that's loose. Let's use the coloring properly.
        
        # Recompute: assign colors, then for each color class, take max weight
        vert_set = set(cand_list)
        color_of = {}
        def degree_in_subgraph(v):
            return len(adj[v] & vert_set)
        ordered = sorted(cand_list, key=degree_in_subgraph, reverse=True)
        
        for v in ordered:
            neighbor_colors = set()
            for u in adj[v]:
                if u in vert_set and u in color_of:
                    neighbor_colors.add(color_of[u])
            c = 0
            while c in neighbor_colors:
                c += 1
            color_of[v] = c
        
        # Sum max weight per color
        color_max = {}
        for v in cand_list:
            c = color_of[v]
            if c not in color_max or weights[v] > color_max[c]:
                color_max[c] = weights[v]
        
        upper_bound = current_weight + sum(color_max.values())
        
        if upper_bound <= best_weight:
            return
        
        # Choose a vertex to branch on: pick the one with highest weight
        v = max(cand_list, key=lambda x: weights[x])
        
        # Branch 1: include v
        new_candidates_include = cand_list - adj[v] - {v}
        bb_search(new_candidates_include, current_weight + weights[v], 
                 current_set | {v})
        
        # Branch 2: exclude v
        new_candidates_exclude = cand_list - {v}
        bb_search(new_candidates_exclude, current_weight, current_set)
    
    all_vertices = set(range(n_vertices))
    bb_search(all_vertices, 0, set())
    
    return best_weight, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    n = 256
    
    # Build conflict graph for 2..256
    adj = _build_conflict_graph(n)
    
    # Weights are the vertex values
    weights = [i + 2 for i in range(len(adj))]
    
    # Solve maximum weight independent set
    best_weight, best_set = _branch_and_bound(adj, weights, deadline)
    
    # Convert set of indices back to numbers
    result = [1]  # Always include 1
    for idx in best_set:
        result.append(idx + 2)
    
    # Ensure types are strictly Python int
    result = [int(x) for x in result]
    
    return result
>>>>>>> REPLACE
