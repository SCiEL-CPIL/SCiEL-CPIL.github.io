"""Rewritten program: C6-free graph on 40 vertices, maximizing edges."""
import random
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a C6-free graph on 40 vertices with many edges.
    
    Strategy:
    1. Initialize with 8 disjoint K5s.
    2. Use bitset-based local search to add edges and perform swaps.
    
    Returns a list of edges [u, v].
    """
    n = 40
    rng = random.Random(42)
    
    # Initialize adjacency as bitsets (integers)
    adj = [0] * n
    edge_set = set()
    edges = []
    
    # Step 1: Initialize with 8 disjoint K5s
    # This gives 8 * 10 = 80 edges.
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = base + a, base + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
                edges.append([u, v])
    
    def can_add_edge(u, v):
        """
        Check if adding edge (u,v) creates a C6.
        A C6 would be u - a - b - c - d - v - u.
        We need to check if there exists a path a-b-c-d of length 3
        in G \ {u,v} such that a in N(u) and d in N(v).
        """
        # Neighbors of u excluding v
        nu = adj[u] & ~(1 << v)
        # Neighbors of v excluding u
        nv = adj[v] & ~(1 << u)
        
        # If no neighbors, no C6
        if nu == 0 or nv == 0:
            return True
            
        # Expand from N(u) by 1 step
        # S1 = { b | exists a in N(u), (a,b) in E, b != u, b != v }
        s1 = 0
        temp = nu
        while temp:
            lsb = temp & -temp
            a = lsb.bit_length() - 1
            s1 |= adj[a]
            temp ^= lsb
        s1 &= ~(1 << u)
        s1 &= ~(1 << v)
        
        # Early termination: if S1 is empty, no C6
        if s1 == 0:
            return True
        
        # Expand from S1 by 1 step
        # S2 = { c | exists b in S1, (b,c) in E, c != u, c != v }
        s2 = 0
        temp = s1
        while temp:
            lsb = temp & -temp
            b = lsb.bit_length() - 1
            s2 |= adj[b]
            temp ^= lsb
        s2 &= ~(1 << u)
        s2 &= ~(1 << v)
        
        # Early termination: if S2 is empty, no C6
        if s2 == 0:
            return True
        
        # Expand from S2 by 1 step
        # S3 = { d | exists c in S2, (c,d) in E, d != u, d != v }
        s3 = 0
        temp = s2
        while temp:
            lsb = temp & -temp
            c = lsb.bit_length() - 1
            s3 |= adj[c]
            temp ^= lsb
        s3 &= ~(1 << u)
        s3 &= ~(1 << v)
        
        # Check if S3 intersects N(v)
        intersection = s3 & nv
        if intersection == 0:
            return True
            
        # If intersection exists, we need to verify if a simple path exists
        # Iterate through possible d in intersection
        temp_d = intersection
        while temp_d:
            lsb_d = temp_d & -temp_d
            d = lsb_d.bit_length() - 1
            temp_d ^= lsb_d
            
            # Find c in N(d) that is in S2
            temp_c = adj[d] & s2
            while temp_c:
                lsb_c = temp_c & -temp_c
                c = lsb_c.bit_length() - 1
                temp_c ^= lsb_c
                
                # Find b in N(c) that is in S1
                temp_b = adj[c] & s1
                while temp_b:
                    lsb_b = temp_b & -temp_b
                    b = lsb_b.bit_length() - 1
                    temp_b ^= lsb_b
                    
                    # Find a in N(b) that is in N(u)
                    temp_a = adj[b] & nu
                    while temp_a:
                        lsb_a = temp_a & -temp_a
                        a = lsb_a.bit_length() - 1
                        temp_a ^= lsb_a
                        
                        # Check all distinct
                        if a != b and b != c and c != d and a != c and a != d and b != d:
                            if a != u and a != v and b != u and b != v and c != u and c != v and d != u and d != v:
                                return False
                            
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v))
        edges.append([u, v])

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edge_set.discard((u, v))
        edge_set.discard((v, u))
        # Remove from edges list
        for i, e in enumerate(edges):
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                edges.pop(i)
                break

    # Local Search
    max_iters = 10000
    no_improve = 0
    
    for _ in range(max_iters):
        improved = False
        
        # 1. Try to add an edge
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    non_edges.append((u, v))
        
        rng.shuffle(non_edges)
        
        added_any = False
        for u, v in non_edges:
            if can_add_edge(u, v):
                add_edge(u, v)
                improved = True
                added_any = True
                break
        
        if improved:
            no_improve = 0
            continue
        else:
            no_improve += 1
            
        # 2. Try Swap: Remove one edge, try to add two
        if edges:
            idx = rng.randint(0, len(edges) - 1)
            u, v = edges[idx]
            remove_edge(u, v)
            
            added_count = 0
            added_edges = []
            
            non_edges2 = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        non_edges2.append((i, j))
            
            rng.shuffle(non_edges2)
            
            for a, b in non_edges2:
                if can_add_edge(a, b):
                    add_edge(a, b)
                    added_edges.append((a, b))
                    added_count += 1
                    if added_count == 2:
                        break
            
            if added_count == 2:
                improved = True
                no_improve = 0
            else:
                for a, b in added_edges:
                    remove_edge(a, b)
                add_edge(u, v)
                
    return edges
