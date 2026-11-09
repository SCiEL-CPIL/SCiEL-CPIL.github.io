"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import product
from functools import lru_cache

def _build_candidate_graph(n_max=255):
    """Build adjacency for candidates that share a factor with 255."""
    # 255 = 3 * 5 * 17
    candidates = [x for x in range(1, n_max + 1) 
                  if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    cand_set = set(candidates)
    n = len(candidates)
    # Map candidate to index
    idx_map = {v: i for i, v in enumerate(candidates)}
    # Build adjacency: edge if gcd > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    return candidates, adj, idx_map

def _max_clique_branch_and_bound(candidates, adj, must_include, deadline):
    """
    Find maximum clique in the graph using branch and bound.
    must_include: set of indices that must be in the clique.
    """
    n = len(candidates)
    best_clique = set(must_include)
    best_size = len(must_include)
    
    # Greedy coloring upper bound for branch and bound
    # Order vertices by degree (descending) for better pruning
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    # We'll use a recursive branch and bound
    # State: current clique, candidate set (vertices that can be added)
    
    def color_bound(cand_set):
        """Upper bound on clique size in cand_set using greedy coloring."""
        if not cand_set:
            return 0
        # Greedy coloring: assign colors to vertices in cand_set
        # Vertices of same color form an independent set, so at most 1 per color in clique
        colors = {}
        color_count = 0
        for v in sorted(cand_set, key=lambda x: -degrees[x]):
            used = set()
            for u in adj[v] & cand_set:
                if u in colors:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            color_count = max(color_count, c + 1)
        return color_count
    
    def branch(clique_set, cand_set):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        if not cand_set:
            if len(clique_set) > best_size:
                best_size = len(clique_set)
                best_clique = set(clique_set)
            return
        # Upper bound
        ub = len(clique_set) + color_bound(cand_set)
        if ub <= best_size:
            return
        # Try adding vertices
        # Order by degree in cand_set for better search
        for v in sorted(cand_set, key=lambda x: -len(adj[x] & cand_set)):
            if time.time() > deadline:
                return
            # Check if v can be added (connected to all in clique)
            if all(u in adj[v] for u in clique_set):
                new_clique = clique_set | {v}
                new_cand = cand_set & adj[v] & {u for u in cand_set if u != v}
                branch(new_clique, new_cand)
            cand_set.discard(v)
    
    # Start: must_include vertices are in the clique
    # Candidates that can be added: connected to all must_include vertices
    initial_cand = set()
    for i in range(n):
        if i in must_include:
            continue
        if all(m in adj[i] for m in must_include):
            initial_cand.add(i)
    
    branch(set(must_include), initial_cand)
    
    return [candidates[i] for i in best_clique]

def construct():
    """Return a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all distinct pairs. Maximize cardinality."""
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # Baseline: multiples of 3
    base_set = [x for x in range(1, 256) if x % 3 == 0]
    best_size = len(base_set)
    best_result = base_set
    
    # Build graph
    candidates, adj, idx_map = _build_candidate_graph(255)
    
    # 255 must be included
    idx_255 = idx_map[255]
    
    # Try branch and bound
    if time.time() < deadline - 10:
        try:
            result = _max_clique_branch_and_bound(
                candidates, adj, {idx_255}, deadline - 5
            )
            if len(result) > best_size:
                best_size = len(result)
                best_result = result
        except Exception:
            pass
    
    # Verify and return
    best_result = sorted(best_result)
    if 255 not in best_result:
        best_result.append(255)
    
    return best_result
# EVOLVE-BLOCK-END
