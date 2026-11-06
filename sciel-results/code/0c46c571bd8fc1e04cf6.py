# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np

def _find_c4(adj, n):
    """Find a C4 in the graph represented by adjacency sets. Return (a,b,c,d) or None."""
    # For each edge (u,v), check if there's a common neighbor pair
    # A C4 is u-x-v-y-u where x,y are distinct common neighbors... no.
    # C4: a-b-c-d-a. So a and c are opposite, b and d are opposite.
    # a-b-c-d-a means: a~b, b~c, c~d, d~a, and a!~c, b!~d (for simple C4, but even if a~c it's still a C4 subgraph)
    # Actually C4 just means 4 distinct vertices forming a cycle of length 4.
    # For each pair of vertices (a,c) that are NOT adjacent, check if they have 2+ common neighbors.
    # If a and c have common neighbors b and d, and b~d, then a-b-c-d-a is a C4.
    # More simply: for each edge (b,d), check if b and d have 2+ common neighbors a,c.
    # Then a-b-c-d-a or a-b-d-c-a... 
    # Let me think: a C4 is four vertices v1,v2,v3,v4 with edges v1v2, v2v3, v3v4, v4v1.
    # v1 and v3 are opposite, v2 and v4 are opposite.
    # For a pair of non-adjacent vertices (v1,v3), if they share two common neighbors v2,v4, 
    # and v2~v4, then v1-v2-v3-v4-v1 is a C4.
    # Actually even if v2 and v4 are not adjacent, v1-v2-v3-v4-v1 is still a C4 as long as all 4 edges exist.
    # The edges needed: v1v2, v2v3, v3v4, v4v1. v2v4 is NOT required.
    # So: for non-adjacent pair (v1,v3), if |N(v1) ∩ N(v3)| >= 2, pick two common neighbors v2,v4.
    # Then v1-v2-v3-v4-v1 is a C4 (edges v1v2, v2v3, v3v4, v4v1 all exist).
    # Wait: v4v1 exists because v4 in N(v1). v3v4 exists because v4 in N(v3). Yes!
    # So any non-adjacent pair with 2+ common neighbors creates a C4.
    
    for a in range(n):
        for c in range(a+1, n):
            if c not in adj[a]:  # non-adjacent
                common = adj[a] & adj[c]
                if len(common) >= 2:
                    common_list = list(common)
                    b = common_list[0]
                    d = common_list[1]
                    if b != d:
                        return (a, b, c, d)
    return None

def _has_c4(adj, n):
    return _find_c4(adj, n) is not None

def _remove_edge(adj, u, v):
    adj[u].discard(v)
    adj[v].discard(u)

def _construct_c4_free(n, min_deg, time_limit):
    """
    Construct C4-free graph on n vertices with min degree >= min_deg.
    Strategy: Start with random graph at appropriate density, then remove C4s.
    """
    start_time = time.time()
    
    for attempt in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Target density: min_deg / (n-1)
        target_density = min_deg / (n - 1)
        # Start slightly above to have room
        p = min(target_density * 1.3, 0.5)
        
        # Initialize random graph
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # Ensure min degree by adding edges (carefully)
        # First, try to fix min degree
        for fix_iter in range(n * 10):
            if time.time() - start_time > time_limit:
                break
            min_d = min(len(adj[i]) for i in range(n))
            if min_d >= min_deg:
                break
            # Find vertex with min degree
            u = min(range(n), key=lambda i: len(adj[i]))
            if len(adj[u]) >= min_deg:
                break
            # Try to add edges to u
            candidates = [v for v in range(n) if v != u and v not in adj[u]]
            random.shuffle(candidates)
            added = False
            for v in candidates:
                # Check if adding u-v creates C4
                # Adding u-v creates C4 iff there exist x in N(u), y in N(v) with x~y
                nu = adj[u]
                nv = adj[v]
                if len(nu) > len(nv):
                    nu, nv = nv, nu
                creates = False
                for x in nu:
                    if adj[x] & nv:
                        creates = True
                        break
                if not creates:
                    adj[u].add(v)
                    adj[v].add(u)
                    added = True
                    break
            if not added:
                break
        
        # Now remove C4s
        for remove_iter in range(n * n * 5):
            if time.time() - start_time > time_limit:
                break
            
            c4 = _find_c4(adj, n)
            if c4 is None:
                break
            
            a, b, c, d = c4
            # Remove one edge from the C4. Choose the edge whose removal
            # least hurts min degree.
            edges_in_c4 = [(a,b), (b,c), (c,d), (d,a)]
            # For each edge, check degree impact
            best_edge = None
            best_score = -1
            for (u, v) in edges_in_c4:
                du = len(adj[u])
                dv = len(adj[v])
                # Prefer removing edge where both endpoints have high degree
                score = min(du, dv)
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
            
            if best_edge:
                u, v = best_edge
                _remove_edge(adj, u, v)
        
        # Check if we succeeded
        min_d = min(len(adj[i]) for i in range(n))
        if min_d >= min_deg and not _has_c4(adj, n):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    
    return None

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # n=47 is impossible by C4 bound: d(d-1) <= n-1 => 8*7=56 > 46
    # Try n=46, min_deg=7
    res = _construct_c4_free(46, 7, 130.0)
    if res:
        return res
    
    # Fallback: n=45, min_deg=6
    remaining = deadline - (time.time() - start_time)
    if remaining > 5.0:
        res = _construct_c4_free(45, 6, remaining - 5.0)
        if res:
            return res
    
    # Fallback: n=44, min_deg=5
    remaining = deadline - (time.time() - start_time)
    if remaining > 2.0:
        res = _construct_c4_free(44, 5, remaining - 2.0)
        if res:
            return res
    
    # Absolute fallback
    return {'n': 1, 'edges': []}

# EVOLVE-BLOCK-END
