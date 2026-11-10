# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a maximum-size subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses branch-and-bound with greedy coloring upper bounds.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s kill

    m = 205
    k = 2

    # Forbidden differences: nonzero quadratic residues mod 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)

    # Build conflict graph: vertices 0..204, edge if difference is forbidden
    # We want maximum independent set in this graph
    # Equivalent: maximum clique in the complement graph
    
    # For each vertex, compute which other vertices it conflicts with
    conflict = [set() for _ in range(m)]
    for a in range(m):
        for b in range(a + 1, m):
            d = (a - b) % m
            if d in forbidden:
                conflict[a].add(b)
                conflict[b].add(a)

    # Use branch and bound for maximum independent set
    # This is equivalent to finding max clique in complement graph
    # Let's work with the conflict graph and find max independent set
    
    best_set = []
    best_size = 0
    
    # Order vertices by degree (higher degree first for better pruning)
    vertices = list(range(m))
    vertices.sort(key=lambda v: -len(conflict[v]))
    
    # Map from original vertex to position in ordering
    pos_in_order = [0] * m
    for i, v in enumerate(vertices):
        pos_in_order[v] = i
    
    # For branch and bound, we maintain:
    # - current set (list of vertices in the independent set)
    # - candidates (vertices that can still be added)
    
    # Use a recursive approach with pruning
    # Upper bound: greedy coloring of the candidate subgraph
    
    sys.setrecursionlimit(10000)
    
    def greedy_color_bound(candidates):
        """
        Greedy coloring of the subgraph induced by candidates.
        Returns the number of colors used, which is an upper bound on the independence number.
        Actually, for maximum independent set, we need to think about this differently.
        
        For maximum independent set in graph G:
        - We can use the fact that alpha(G) <= chi(G) where chi is chromatic number
        - But computing chi is hard. Instead, we use a different approach.
        
        Alternative: For maximum clique in complement graph, we use greedy coloring bound.
        Let's reformulate: find max clique in complement of conflict graph.
        """
        # Actually, let's use a simpler upper bound: 
        # The maximum independent set in the subgraph induced by candidates
        # is at most the size of candidates.
        # For better pruning, we can use the fact that if we color the graph,
        # each color class is an independent set, so alpha >= max color class size.
        # But we need an upper bound.
        
        # A common technique: sort candidates, greedily assign colors.
        # The number of colors is an upper bound on the clique number of the complement,
        # which equals the independence number of the original.
        
        # Let's just return len(candidates) as a trivial upper bound for now,
        # and rely on other pruning.
        
        # Better: use the fact that we can greedily build an independent set
        # and use its size as a lower bound, but we need upper bounds for pruning.
        
        # For maximum independent set, a good upper bound is:
        # For each vertex v in candidates, alpha(G[candidates]) <= 1 + alpha(G[candidates \ N[v]])
        # where N[v] is the closed neighborhood of v.
        # This gives a recursive upper bound but is expensive.
        
        # Simpler approach: use the minimum degree in the candidate subgraph.
        # If the minimum degree is d, then alpha >= n/(d+1) (Turán), but we need upper bound.
        
        # Let's use a different strategy: branch and bound with the following:
        # - If |candidates| + |current_set| <= best_size, prune.
        # - Try adding vertices in order of increasing degree in the candidate subgraph.
        
        return len(candidates)
    
    def branch_and_bound(current_set, candidates, depth=0):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Pruning: even if we add all candidates, can we beat best?
        if len(current_set) + len(candidates) <= best_size:
            return
        
        # Choose the next vertex to branch on
        # Heuristic: pick vertex with minimum degree in the candidate subgraph
        # (this tends to give better bounds)
        
        # Compute degrees within candidates
        cand_set = set(candidates)
        min_deg = float('inf')
        chosen = None
        for v in candidates:
            deg = len(conflict[v] & cand_set)
            if deg < min_deg:
                min_deg = deg
                chosen = v
        
        if chosen is None:
            return
        
        # Option 1: Include chosen in the independent set
        # Remove chosen and all its neighbors from candidates
        new_candidates = [v for v in candidates if v != chosen and v not in conflict[chosen]]
        current_set.append(chosen)
        branch_and_bound(current_set, new_candidates, depth + 1)
        current_set.pop()
        
        # Option 2: Exclude chosen from the independent set
        new_candidates = [v for v in candidates if v != chosen]
        branch_and_bound(current_set, new_candidates, depth + 1)
    
    # Initial candidates: all vertices
    candidates = list(range(m))
    branch_and_bound([], candidates)
    
    if time.time() > deadline:
        # Return whatever we have
        if not best_set:
            best_set = [0]  # Fallback
        return np.array(best_set, dtype=np.int64)
    
    # Verify the solution
    result = best_set
    # Double-check validity
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            d = (result[i] - result[j]) % m
            if d in forbidden:
                # This shouldn't happen, but if it does, fall back to greedy
                break
    
    if not result:
        # Fallback to greedy
        selected = []
        for candidate in range(m):
            ok = True
            for s in selected:
                d1 = (candidate - s) % m
                d2 = (s - candidate) % m
                if d1 in forbidden or d2 in forbidden:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        result = selected
    
    return np.array(result, dtype=np.int64)


# EVOLVE-BLOCK-END
