"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import sys
from collections import deque

def _has_q3(edge_set):
    """Check if the graph defined by edge_set (set of frozenset({u,v})) contains Q3 as a subgraph."""
    n = 16
    adj = [set() for _ in range(n)]
    for e in edge_set:
        u, v = tuple(e)
        adj[u].add(v)
        adj[v].add(u)
    
    # For each ordered triple of edges that form a path of length 2 (u-v-w),
    # and for each vertex x in N(u) \ {v} and y in N(w) \ {v},
    # check if {u,v,w,x,y} plus appropriate edges form Q3.
    # Actually, Q3 is bipartite with parts of size 4.
    # A subgraph isomorphic to Q3 exists if there are 8 vertices that induce Q3 (or contain Q3 as subgraph).
    # Q3 has 8 vertices and 12 edges.
    # We can enumerate all 8-vertex subsets and check if they contain Q3.
    # C(16,8) = 12870. For each, check if the induced subgraph contains Q3.
    # To check if an 8-vertex graph contains Q3:
    # Q3 is the unique bipartite graph on 8 vertices with 12 edges and girth 4 that is 3-regular.
    # So if the induced subgraph on 8 vertices has >= 12 edges and is bipartite with parts of size 4, 
    # and contains a 3-regular bipartite subgraph on 8 vertices...
    # Actually, since we are looking for a subgraph (not induced), we need to check if there's a subset of edges forming Q3.
    # But since Q3 is 3-regular on 8 vertices, any subgraph containing Q3 must have at least 12 edges among those 8 vertices.
    # So: for each 8-subset S, if the number of edges within S is >= 12, then check if Q3 is a subgraph.
    
    # Let's precompute the 8-subsets that have >= 12 edges.
    vertices = list(range(n))
    for combo in itertools.combinations(vertices, 8):
        combo_set = set(combo)
        # Count edges in induced subgraph
        edge_count = 0
        for i, u in enumerate(combo):
            for v in combo[i+1:]:
                if (min(u,v), max(u,v)) in {(a,b) for a,b in [tuple(e) for e in edge_set]}:
                    # This is inefficient. Let's use adj.
                    pass
        # Better:
        cnt = 0
        for u in combo:
            for v in combo:
                if u < v and v in adj[u]:
                    cnt += 1
        if cnt < 12:
            continue
        
        # Check if Q3 is a subgraph of the induced subgraph on `combo`.
        # Q3 is bipartite. Let's try all ways to split `combo` into two sets of 4.
        # There are C(8,4)/2 = 35 ways (up to complement).
        combo_list = list(combo)
        for A in itertools.combinations(combo_list, 4):
            B = tuple(v for v in combo_list if v not in A)
            A_set = set(A)
            B_set = set(B)
            # Check if the bipartite graph between A and B (using edges from the original graph) contains Q3.
            # Q3 as a bipartite graph: it's the incidence graph of Fano plane? No.
            # Q3 is the 3-dimensional hypercube. It's bipartite with parts being even/odd parity vertices.
            # The bipartite graph of Q3 has each vertex of degree 3.
            # So we need a 3-regular bipartite subgraph between A and B.
            # Since |A|=|B|=4, and total edges in Q3 is 12, we need exactly 12 edges present, and each vertex has degree 3.
            # So: count edges between A and B. If exactly 12, and every vertex has degree 3, it's Q3.
            # Actually, if there are 12 edges between A and B, and each vertex in A has degree <= 3 (since it connects to at most 4 vertices in B, but Q3 requires exactly 3), 
            # and same for B, then if all degrees are exactly 3, it's Q3.
            # But wait, Q3 is 3-regular. So yes.
            
            degs_a = [0]*4
            degs_b = [0]*4
            edge_cnt = 0
            for i, u in enumerate(A):
                for j, v in enumerate(B):
                    if v in adj[u]:
                        degs_a[i] += 1
                        degs_b[j] += 1
                        edge_cnt += 1
            if edge_cnt == 12 and all(d == 3 for d in degs_a) and all(d == 3 for d in degs_b):
                return True
    return False

def _edge_to_idx(u, v):
    if u > v:
        u, v = v, u
    return u * 16 + v  # 0..120

def _idx_to_edge(idx):
    u = idx // 16
    v = idx % 16
    if u == v:
        return None
    return (u, v)

def _all_edges():
    edges = []
    for u in range(16):
        for v in range(u+1, 16):
            edges.append((u, v))
    return edges

def _remove_q3_edges(edge_set, max_iterations=1000000, deadline=None):
    """
    Start with a set of edges, and iteratively remove edges to break all Q3 subgraphs.
    Returns a Q3-free set of edges.
    """
    edge_set = set(edge_set)
    # Convert to set of frozensets for fast lookup
    fs_edge_set = set(frozenset(e) for e in edge_set)
    
    it = 0
    while True:
        if deadline and time.time() > deadline:
            break
        it += 1
        # Find a Q3
        if not _has_q3(fs_edge_set):
            break
        
        # Find which edges to remove. We need to find an actual Q3 and remove one of its edges.
        # Modify _has_q3 to return the Q3 edges? Or just do a simpler approach:
        # For each edge, remove it and check if Q3 is gone. This is O(E * check_time) which is too slow.
        
        # Let's find a Q3 explicitly.
        q3_found = _find_q3(fs_edge_set)
        if q3_found is None:
            break
        # q3_found is a set of 12 edges forming a Q3
        # Remove one edge (randomly or the first one)
        edge_to_remove = next(iter(q3_found))
        fs_edge_set.remove(edge_to_remove)
        
    return set(tuple(sorted(e)) for e in fs_edge_set)

def _find_q3(edge_set):
    """Find a Q3 subgraph in the graph defined by edge_set (set of frozensets). Return set of 12 edges or None."""
    n = 16
    adj = [set() for _ in range(n)]
    for e in edge_set:
        u, v = tuple(e)
        adj[u].add(v)
        adj[v].add(u)
    
    vertices = list(range(n))
    for combo in itertools.combinations(vertices, 8):
        # Count edges
        cnt = 0
        for u in combo:
            for v in combo:
                if u < v and v in adj[u]:
                    cnt += 1
        if cnt < 12:
            continue
        
        combo_list = list(combo)
        for A in itertools.combinations(combo_list, 4):
            B = tuple(v for v in combo_list if v not in A)
            degs_a = [0]*4
            degs_b = [0]*4
            edge_cnt = 0
            for i, u in enumerate(A):
                for j, v in enumerate(B):
                    if v in adj[u]:
                        degs_a[i] += 1
                        degs_b[j] += 1
                        edge_cnt += 1
            if edge_cnt == 12 and all(d == 3 for d in degs_a) and all(d == 3 for d in degs_b):
                # Found it! Reconstruct the 12 edges.
                q3_edges = set()
                for i, u in enumerate(A):
                    for j, v in enumerate(B):
                        if v in adj[u]:
                            q3_edges.add(frozenset((u, v)))
                return q3_edges
    return None

def construct():
    deadline = time.time() + 110  # Safety margin
    
    # Start with a dense graph. K16 has 120 edges.
    # But K16 contains many Q3s.
    # Let's start with K16 and iteratively remove edges.
    
    all_edges = _all_edges()
    edge_set = set(frozenset(e) for e in all_edges)
    
    # Greedy removal
    it = 0
    while time.time() < deadline:
        if not _has_q3(edge_set):
            break
        q3 = _find_q3(edge_set)
        if q3 is None:
            break
        # Remove a random edge from the Q3
        edge_to_remove = random.choice(list(q3))
        edge_set.remove(edge_to_remove)
    
    # Convert to list of [u,v]
    result = []
    for e in edge_set:
        u, v = tuple(e)
        if u > v:
            u, v = v, u
        result.append([u, v])
    
    return result
# EVOLVE-BLOCK-END
