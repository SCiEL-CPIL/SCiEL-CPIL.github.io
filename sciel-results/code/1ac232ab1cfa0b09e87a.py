"""Construct a 4-regular graph of girth 10 (the (4,10)-cage)."""
import numpy as np
from collections import deque
import time
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph of girth at least g.
    
    For k=4, g=10, uses randomized local search to find a small cage.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    if k == 4 and g == 10:
        return _construct_4_10_cage_random(start_time, deadline)
    else:
        return _construct_general(k, g)

def _construct_4_10_cage_random(start_time, deadline):
    """Construct the (4,10)-cage using randomized local search."""
    # The known best is 320. We try to find a graph with n=320.
    # If we can't, we try larger n.
    # Moore bound is 161.
    
    # We will try n = 320 first.
    # If it fails, we might try 322, etc.
    # But the score is 1/n, so smaller is better.
    
    target_n = 320
    max_n = 340 # Try a few above 320 if 320 fails
    
    for n in range(target_n, max_n + 1):
        if time.time() > deadline:
            break
        # Try to find a valid graph of size n
        graph = _find_cage_random(n, 4, 10, deadline)
        if graph is not None:
            edges = graph
            return n, edges
            
    # Fallback: Return a valid graph if we found one in a previous step? 
    # No, the loop returns immediately.
    # If we exit the loop without returning, we need a fallback.
    # The LW graph is a safe fallback, though large.
    return _construct_4_10_cage_lw()

def _find_cage_random(n, k, g, deadline):
    """
    Randomized local search for a k-regular graph of girth >= g on n vertices.
    Returns list of edges or None.
    """
    # 1. Generate random k-regular graph
    # Using configuration model
    # 2. Check girth
    # 3. If girth < g, find short cycle, 2-switch.
    
    # Configuration model
    # Each vertex has k stubs.
    # Total stubs = n * k
    # Randomly pair stubs.
    
    # To avoid self-loops and multi-edges, we use a retry loop or a specific generator.
    # For k=4, n=320, random pairing often creates multi-edges.
    # We will use a simple rejection sampling for the initial graph, 
    # or just generate and fix.
    
    # Better: Start with a known 4-regular graph structure, e.g., a grid or cycle, 
    # and randomize edges.
    # Or just use networkx if available? No, standard library + numpy/scipy.
    
    # Let's generate a random 4-regular graph by swapping edges.
    # Start with a fixed 4-regular graph (e.g., a circulant graph).
    # C_320 with chords?
    # Let's use a random permutation-based construction or just random stub pairing with validation.
    
    def generate_random_k_regular(n, k):
        # Try to generate a simple k-regular graph
        # Use stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        while True:
            random.shuffle(stubs)
            # Pair stubs
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((min(u, v), max(u, v)))
            
            if valid:
                # Convert to list
                return list(edges)
            # If not valid, retry. 
            # For n=320, k=4, this might take a while?
            # Probability of self-loop is 1/n. 
            # Probability of multi-edge is small.
            # But retrying the whole shuffle is expensive.
            # We can just break and try again.
            # Given 115s budget, and we need many iterations, 
            # we should optimize this.
            # Actually, for k=4, the number of 4-regular graphs is huge.
            # Rejection sampling is fine for the first step.
            
    edges = generate_random_k_regular(n, k)
    
    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Function to find shortest cycle
    def find_shortest_cycle(adj, n, max_len):
        # BFS from each node
        # To speed up, we can stop if we find a cycle < max_len
        # Actually, we just need to know if girth < g.
        # And find one such cycle to break.
        
        # For efficiency, we can do BFS from a random node first.
        # If girth is large, we might need to check many.
        # But if girth is small, we find it fast.
        
        for start in range(n):
            # BFS
            dist = [-1] * n
            prev = [-1] * n
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                if dist[u] >= max_len // 2: # We only care about cycles < g
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        q.append(v)
                    else:
                        # Check if this closes a cycle
                        # If v is not parent of u and u is not parent of v
                        if prev[u] != v and prev[v] != u:
                            # Cycle length
                            # We need to find the LCA of u and v in the BFS tree?
                            # Or just use the fact that in BFS, 
                            # if we see a visited node, the cycle length is 
                            # dist[u] + dist[v] + 1?
                            # This is only true if the path from start to u and start to v 
                            # diverge at the LCA.
                            # Actually, a simpler check:
                            # If we find a cycle, we can reconstruct it.
                            # But for the purpose of "is girth < g", 
                            # we can just check if dist[u] + dist[v] + 1 < g 
                            # AND the paths don't share edges in a way that makes it smaller?
                            # Actually, the shortest cycle through u and v 
                            # in a BFS tree from start is not necessarily dist[u]+dist[v]+1 
                            # unless the paths are disjoint until start?
                            # No, the cycle is formed by the path u->...->start->...->v->u?
                            # No, the edge (u,v) closes the cycle.
                            # The cycle is u->...->LCA->...->v->u.
                            # Length = dist[u] + dist[v] - 2*dist[LCA] + 1.
                            # To find LCA, we need to trace back.
                            
                            # Simpler: Just do BFS from start. 
                            # If we encounter a visited node v (not parent), 
                            # we have found a cycle.
                            # The length is at most dist[u] + dist[v] + 1.
                            # If this is < g, we have a short cycle.
                            # We can return the cycle edges.
                            
                            if dist[u] + dist[v] + 1 < g:
                                # Reconstruct cycle
                                # Trace back from u to start
                                path_u = []
                                curr = u
                                while curr != -1:
                                    path_u.append(curr)
                                    curr = prev[curr]
                                path_u.reverse()
                                
                                # Trace back from v to start
                                path_v = []
                                curr = v
                                while curr != -1:
                                    path_v.append(curr)
                                    curr = prev[curr]
                                path_v.reverse()
                                
                                # Find LCA
                                # The LCA is the last common node in path_u and path_v
                                # Since both end at start, they share start.
                                # Find the longest common suffix of path_u and path_v?
                                # No, they are paths from start.
                                # Find the first common node from the end?
                                # Actually, just iterate from start?
                                # No, the LCA is the node where the paths diverge.
                                
                                # Let's just find the cycle by tracing back from u and v 
                                # until they meet.
                                set_u = set(path_u)
                                lca = None
                                for node in path_v:
                                    if node in set_u:
                                        lca = node
                                        break
                                
                                if lca is None:
                                    # Should not happen if connected
                                    continue
                                    
                                # Cycle: u -> ... -> lca -> ... -> v -> u
                                # Path u to lca
                                idx_u = path_u.index(lca)
                                path_u_lca = path_u[:idx_u+1]
                                # Path v to lca
                                idx_v = path_v.index(lca)
                                path_v_lca = path_v[:idx_v+1]
                                
                                # Cycle edges
                                cycle_edges = []
                                for i in range(len(path_u_lca) - 1):
                                    u1, u2 = path_u_lca[i], path_u_lca[i+1]
                                    cycle_edges.append((min(u1, u2), max(u1, u2)))
                                for i in range(len(path_v_lca) - 1):
                                    v1, v2 = path_v_lca[i], path_v_lca[i+1]
                                    cycle_edges.append((min(v1, v2), max(v1, v2)))
                                # Edge (u, v)
                                cycle_edges.append((min(u, v), max(u, v)))
                                
                                return cycle_edges
                            
        return None

    # Local Search Loop
    iterations = 0
    max_iterations = 50000
    last_improvement = time.time()
    
    while time.time() < deadline - 5.0:
        if time.time() - last_improvement > 10.0:
            # Restart if no improvement for 10s
            # Or just break and try next n?
            # For now, break and try next n
            break
            
        # Find a short cycle
        cycle_edges = find_shortest_cycle(adj, n, g)
        
        if cycle_edges is None:
            # Girth >= g
            return edges
            
        # Perform a 2-switch
        # Pick a random edge (u, v) in the cycle
        e1 = random.choice(cycle_edges)
        # Pick a random edge (x, y) NOT in the cycle
        # To ensure we don't create a multi-edge or self-loop
        
        # We need to pick an edge (x, y) such that 
        # swapping (u,v) and (x,y) with (u,x) and (v,y) 
        # (or u,y and v,x) is valid.
        
        # Candidate edges: all edges not in cycle_edges
        # But we must ensure (u,x) and (v,y) are not existing edges.
        
        # Let's pick a random edge from the whole graph, excluding cycle_edges
        # And check validity.
        
        # To speed up, we can pick a random edge from a random vertex's neighborhood.
        
        success = False
        for _ in range(100):
            # Pick a random edge
            # We can pick a random vertex and one of its neighbors
            u_rand = random.randint(0, n-1)
            if not adj[u_rand]:
                continue
            v_rand = random.choice(adj[u_rand])
            
            e2 = (min(u_rand, v_rand), max(u_rand, v_rand))
            
            if e2 in cycle_edges:
                continue
                
            # Try to swap
            # Remove e1, e2
            # Add e3, e4
            # e1 = (u1, v1)
            # e2 = (u2, v2)
            # Options:
            # (u1, u2) and (v1, v2)
            # (u1, v2) and (v1, u2)
            
            u1, v1 = e1
            u2, v2 = e2
            
            # Check if (u1, u2) is valid
            if (u1, u2) not in adj[u1] and u1 != u2:
                e3 = (min(u1, u2), max(u1, u2))
                e4 = (min(v1, v2), max(v1, v2))
                if (v1, v2) not in adj[v1] and v1 != v2:
                    # Check if e3, e4 are new (not e1, e2)
                    # And not existing
                    if e3 != e1 and e3 != e2 and e4 != e1 and e4 != e2:
                        # Check if e3, e4 are not already in graph
                        if e3 not in adj[u1] and e4 not in adj[v1]:
                            # Perform swap
                            # Remove e1, e2
                            adj[u1].remove(v1)
                            adj[v1].remove(u1)
                            adj[u2].remove(v2)
                            adj[v2].remove(u2)
                            
                            # Add e3, e4
                            adj[u1].append(u2)
                            adj[u2].append(u1)
                            adj[v1].append(v2)
                            adj[v2].append(v1)
                            
                            # Update edges list
                            # This is O(E) to remove. 
                            # For n=320, E=640. It's fine.
                            edges.remove(e1)
                            edges.remove(e2)
                            edges.append(e3)
                            edges.append(e4)
                            
                            success = True
                            last_improvement = time.time()
                            break
                            
            # Try other pairing
            if (u1, v2) not in adj[u1] and u1 != v2:
                e3 = (min(u1, v2), max(u1, v2))
                e4 = (min(v1, u2), max(v1, u2))
                if (v1, u2) not in adj[v1] and v1 != u2:
                    if e3 != e1 and e3 != e2 and e4 != e1 and e4 != e2:
                        if e3 not in adj[u1] and e4 not in adj[v1]:
                            # Perform swap
                            adj[u1].remove(v1)
                            adj[v1].remove(u1)
                            adj[u2].remove(v2)
                            adj[v2].remove(u2)
                            
                            adj[u1].append(v2)
                            adj[v2].append(u1)
                            adj[v1].append(u2)
                            adj[u2].append(v1)
                            
                            edges.remove(e1)
                            edges.remove(e2)
                            edges.append(e3)
                            edges.append(e4)
                            
                            success = True
                            last_improvement = time.time()
                            break
        
        if not success:
            # Try next iteration
            continue

    return None

def _construct_4_10_cage_lw():
    """Fallback: Construct LW graph."""
    q = 4
    d = 5
    
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    def gf_add(x, y):
        return x ^ y
    
    def gf_mul(x, y):
        return mul_table[x][y]
    
    rule = {
        2: (1, 1),
        3: (2, 1),
        4: (1, 2),
        5: (1, 3)
    }
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_add(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, tuple([0] * (d + 1)))
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _construct_general(k, g):
    """General construction for other parameters."""
    n = 2 * k * (g - 1)
    edges = []
    for i in range(n):
        for j in range(1, k // 2 + 1):
            edges.append((i, (i + j) % n))
    return n, edges
# EVOLVE-BLOCK-END
