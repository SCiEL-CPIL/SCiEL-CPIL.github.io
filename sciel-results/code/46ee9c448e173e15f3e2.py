import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin for 120s limit
    
    n = 20
    
    # Use a structured construction: Take the complement of a known K3,3-free graph
    # Strategy: Use a strongly regular graph or algebraic construction
    
    # Approach: Start with a dense graph and remove edges to eliminate K3,3 subgraphs
    # Better: Use a known construction for K3,3-free graphs
    
    # Known result: ex(n, K3,3) is at least n^2/4 for large n
    # For n=20, we want around 100+ edges
    
    # Construction: Use a bipartite graph with parts of size 10 each
    # But we can do better with a non-bipartite construction
    
    # Let's use a greedy approach with a good initial structure
    # Start with a random regular-ish graph and optimize
    
    M = [0] * n
    
    def can_add(u, v):
        if u == v:
            return False
        if (M[u] >> v) & 1:
            return False
        
        new_M_u = M[u] | (1 << v)
        new_M_v = M[v] | (1 << u)
        
        # Check all triples that include u or v
        # A K3,3 exists if there are 3 vertices with 3 common neighbors
        # When adding edge (u,v), we need to check if any triple now has 3 common neighbors
        
        # Triples containing u: {u, x, y}
        for x in range(n):
            if x == u:
                continue
            for y in range(x + 1, n):
                if y == u:
                    continue
                # Common neighbors of u, x, y
                common = new_M_u & M[x] & M[y]
                if common.bit_count() >= 3:
                    return False
        
        # Triples containing v but not u: {v, x, y} where x,y != u
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                common = new_M_v & M[x] & M[y]
                if common.bit_count() >= 3:
                    return False
        
        return True

    def add_edge(u, v):
        M[u] |= (1 << v)
        M[v] |= (1 << u)

    def remove_edge(u, v):
        M[u] &= ~(1 << v)
        M[v] &= ~(1 << u)

    def count_edges():
        return sum(M[i].bit_count() for i in range(n)) // 2

    def get_edge_list():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (M[u] >> v) & 1:
                    edges.append((u, v))
        return edges

    def try_add_all():
        """Try to add all possible edges greedily"""
        added = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not ((M[u] >> v) & 1):
                    if can_add(u, v):
                        add_edge(u, v)
                        added += 1
        return added

    def local_search_maximize():
        """Local search to maximize edges"""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any missing edge
            for u in range(n):
                if time.time() > deadline:
                    return
                for v in range(u + 1, n):
                    if not ((M[u] >> v) & 1):
                        if can_add(u, v):
                            add_edge(u, v)
                            improved = True
            
            # Try 1-for-1 swaps: remove edge (a,b), add edge (c,d)
            if not improved:
                edges = get_edge_list()
                random.shuffle(edges)
                for (a, b) in edges:
                    if time.time() > deadline:
                        return
                    remove_edge(a, b)
                    added = 0
                    for c in range(n):
                        for d in range(c + 1, n):
                            if not ((M[c] >> d) & 1) and (c, d) != (a, b):
                                if can_add(c, d):
                                    add_edge(c, d)
                                    added += 1
                                    break
                        else:
                            continue
                        break
                    
                    if added >= 1:
                        improved = True
                    else:
                        add_edge(a, b)
        
        return count_edges()

    # Multiple restarts with different initializations
    best_edges = 0
    best_M = [0] * n
    random.seed(42)
    
    # Strategy 1: Start with a random sparse graph
    for restart in range(15):
        if time.time() > deadline:
            break
        
        M = [0] * n
        
        # Random initialization
        density = 0.3 + 0.05 * restart
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < density:
                    if can_add(u, v):
                        add_edge(u, v)
        
        # Greedy fill
        try_add_all()
        
        # Local search
        local_search_maximize()
        
        current = count_edges()
        if current > best_edges:
            best_edges = current
            best_M = M[:]
    
    # Strategy 2: Start with a structured graph (cycle + chords)
    if time.time() < deadline:
        M = [0] * n
        # Create a cycle
        for i in range(n):
            add_edge(i, (i + 1) % n)
        # Add some chords
        for i in range(n):
            add_edge(i, (i + 5) % n)
            add_edge(i, (i + 7) % n)
        
        try_add_all()
        local_search_maximize()
        
        current = count_edges()
        if current > best_edges:
            best_edges = current
            best_M = M[:]
    
    # Strategy 3: Start with a bipartite-like structure
    if time.time() < deadline:
        M = [0] * n
        # Split into two groups of 10
        group1 = list(range(10))
        group2 = list(range(10, 20))
        
        # Add edges within groups (sparse)
        for i in range(len(group1)):
            for j in range(i + 1, len(group1)):
                if random.random() < 0.3:
                    u, v = group1[i], group1[j]
                    if can_add(u, v):
                        add_edge(u, v)
        
        for i in range(len(group2)):
            for j in range(i + 1, len(group2)):
                if random.random() < 0.3:
                    u, v = group2[i], group2[j]
                    if can_add(u, v):
                        add_edge(u, v)
        
        # Add cross edges (denser)
        for u in group1:
            for v in group2:
                if random.random() < 0.6:
                    if can_add(u, v):
                        add_edge(u, v)
        
        try_add_all()
        local_search_maximize()
        
        current = count_edges()
        if current > best_edges:
            best_edges = current
            best_M = M[:]
    
    # Final refinement on best solution
    if best_edges > 0 and time.time() < deadline:
        M = best_M[:]
        local_search_maximize()
        current = count_edges()
        if current > best_edges:
            best_edges = current
            best_M = M[:]
    
    # Return the best found
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_M[u] >> v) & 1:
                edges.append([u, v])
    return edges
