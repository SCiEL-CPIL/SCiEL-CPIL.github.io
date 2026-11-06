"""Rewritten program: C6-free graph on 40 vertices, maximizing edges."""
import random
import numpy as np
from itertools import combinations

def construct():
    n = 40
    rng = random.Random(42)
    
    # Initialize adjacency as bitsets
    adj = [0] * n
    edge_set = set()
    edges = []

    def add_edge(u, v):
        if u == v: return
        if u > v: u, v = v, u
        if (u, v) in edge_set: return
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v))
        edges.append([u, v])

    def remove_edge(u, v):
        if u == v: return
        if u > v: u, v = v, u
        if (u, v) not in edge_set: return
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edge_set.remove((u, v))
        for i, e in enumerate(edges):
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                edges.pop(i)
                break

    def can_add_edge(u, v):
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        if nu == 0 or nv == 0:
            return True
            
        s1 = 0
        temp = nu
        while temp:
            lsb = temp & -temp
            a = lsb.bit_length() - 1
            s1 |= adj[a]
            temp ^= lsb
        s1 &= ~(1 << u)
        s1 &= ~(1 << v)
        
        s2 = 0
        temp = s1
        while temp:
            lsb = temp & -temp
            b = lsb.bit_length() - 1
            s2 |= adj[b]
            temp ^= lsb
        s2 &= ~(1 << u)
        s2 &= ~(1 << v)
        
        s3 = 0
        temp = s2
        while temp:
            lsb = temp & -temp
            c = lsb.bit_length() - 1
            s3 |= adj[c]
            temp ^= lsb
        s3 &= ~(1 << u)
        s3 &= ~(1 << v)
        
        intersection = s3 & nv
        if intersection == 0:
            return True
            
        temp_d = intersection
        while temp_d:
            lsb_d = temp_d & -temp_d
            d = lsb_d.bit_length() - 1
            temp_d ^= lsb_d
            
            temp_c = adj[d] & s2
            while temp_c:
                lsb_c = temp_c & -temp_c
                c = lsb_c.bit_length() - 1
                temp_c ^= lsb_c
                
                temp_b = adj[c] & s1
                while temp_b:
                    lsb_b = temp_b & -temp_b
                    b = lsb_b.bit_length() - 1
                    temp_b ^= lsb_b
                    
                    temp_a = adj[b] & nu
                    while temp_a:
                        lsb_a = temp_a & -temp_a
                        a = lsb_a.bit_length() - 1
                        temp_a ^= lsb_a
                        
                        if a != b and b != c and c != d and a != c and a != d and b != d:
                            return False
        return True

    # Step 1: Spectral Initialization
    p = 0.15
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < p:
                A[i, j] = 1
                A[j, i] = 1
                
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    principal_eigenvector = eigenvectors[:, -1]
    if principal_eigenvector.sum() < 0:
        principal_eigenvector = -principal_eigenvector
        
    scores = []
    for i in range(n):
        for j in range(i+1, n):
            score = principal_eigenvector[i] * principal_eigenvector[j]
            scores.append((score, i, j))
            
    scores.sort(reverse=True)
    
    for score, i, j in scores:
        if not (adj[i] & (1 << j)):
            if can_add_edge(i, j):
                add_edge(i, j)
                
    # Step 2: Local Search
    no_improve = 0
    import time
    start_time = time.time()
    
    while no_improve < 500 and (time.time() - start_time) < 110:
        improved = False
        
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
                    
        rng.shuffle(non_edges)
        
        added_any = False
        for i, j in non_edges:
            if can_add_edge(i, j):
                add_edge(i, j)
                improved = True
                added_any = True
                break
                
        if improved:
            no_improve = 0
            continue
        else:
            no_improve += 1
            
        if edge_set:
            edge_to_remove = rng.choice(list(edge_set))
            u, v = edge_to_remove
            remove_edge(u, v)
            
            added_count = 0
            added_edges = []
            
            non_edges2 = []
            for i in range(n):
                for j in range(i+1, n):
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
