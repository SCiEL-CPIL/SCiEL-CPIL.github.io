"""Optimized search for the largest pairwise non-coprime subset of [1, 255] containing 255."""

import math
import time

_N = 255
_TARGET = 255

def _is_valid_clique(S):
    """Check that S is pairwise non-coprime and contains 255."""
    if _TARGET not in S:
        return False
    S = list(S)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if math.gcd(S[i], S[j]) == 1:
                return False
    return True

def _build_candidate_graph():
    """
    Build a graph where edges represent gcd(a,b) > 1 for a, b in [1, 255].
    Returns adjacency as a set of frozensets for speed, and a list of candidates
    that can coexist with 255 (i.e., gcd(x, 255) > 1).
    """
    candidates = [x for x in range(1, _N + 1) if math.gcd(x, _TARGET) > 1]
    
    adj = {x: set() for x in candidates}
    for i in range(len(candidates)):
        a = candidates[i]
        for j in range(i + 1, len(candidates)):
            b = candidates[j]
            if math.gcd(a, b) > 1:
                adj[a].add(b)
                adj[b].add(a)
    
    return candidates, adj

def _max_clique_branch_and_bound(adj, max_time=50.0):
    """
    Branch and bound to find the maximum clique in the candidate graph.
    Uses a simple coloring bound.
    """
    best_clique = set()
    
    # Initial solution: Greedy from 255
    # Also try specific prime multiples
    candidates = list(adj.keys())
    
    # Hardcoded strong lower bounds
    mult3 = set(x for x in candidates if x % 3 == 0)
    mult5 = set(x for x in candidates if x % 5 == 0)
    mult17 = set(x for x in candidates if x % 17 == 0)
    
    best_size = max(len(mult3), len(mult5), len(mult17))
    best_set = mult3 if len(mult3) == best_size else (mult5 if len(mult5) == best_size else mult17)
    
    # Sort vertices by degree (descending) for better branching
    vertices = sorted(adj.keys(), key=lambda x: len(adj[x]), reverse=True)
    
    deadline = time.time() + max_time
    
    def greedy_coloring(P):
        """Color the subgraph induced by P using greedy coloring.
        Returns (coloring_dict, num_colors) where coloring_dict[v] = color.
        """
        color = {}
        u = 0
        P_sorted = sorted(P, key=lambda v: len(adj[v] & set(P)), reverse=True)
        
        for v in P_sorted:
            used = set()
            for w in adj[v]:
                if w in color:
                    used.add(color[w])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            u = max(u, c + 1)
        
        return color, u
    
    def expand(R, P):
        nonlocal best_size, best_set, deadline
        if time.time() > deadline:
            return
        
        if len(R) + len(P) <= best_size:
            return
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_set = set(R)
            return
        
        color, ub = greedy_coloring(P)
        
        if len(R) + ub <= best_size:
            return
        
        P_sorted = sorted(P, key=lambda v: color[v])
        
        for i, v in enumerate(P_sorted):
            if time.time() > deadline:
                return
            if len(R) + ub - color[v] <= best_size:
                continue
            
            R.append(v)
            P_new = [x for x in P_sorted[i+1:] if x in adj[v]]
            expand(R, set(P_new))
            R.pop()
            
            if len(R) + len(P_new) <= best_size:
                break
        
    expand([], set(adj.keys()))
    
    return best_set

def construct():
    t0 = time.time()
    deadline = t0 + 55.0
    
    candidates, adj = _build_candidate_graph()
    
    best = set()
    
    # Try simple prime multiples
    mult3 = set(x for x in candidates if x % 3 == 0)
    mult5 = set(x for x in candidates if x % 5 == 0)
    mult17 = set(x for x in candidates if x % 17 == 0)
    
    best = max([mult3, mult5, mult17], key=len)
    
    if time.time() < deadline - 5:
        search_result = _max_clique_branch_and_bound(adj, max_time=min(deadline - time.time() - 2, 45.0))
        if len(search_result) > len(best):
            best = search_result
    
    if _is_valid_clique(best):
        return list(best)
    
    return list(mult3)
