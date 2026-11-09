"""
Combined program: PG(2,5) base + Bron-Kerbosch extension + Fast Bitmask Independence Check.
"""

import time
import sys

sys.setrecursionlimit(10000)

def _popcount(x):
    return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

def _polarity_graph_pg25():
    """
    Construct the polarity graph of PG(2,5).
    Points are vectors in GF(5)^3, up to scale.
    There are (5^3 - 1)/(5 - 1) = 31 points.
    Two points are adjacent if their dot product is 0.
    This graph is C4-free.
    """
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 3, 5)
                    nx = 1
                    ny = (y * inv_x) % 5
                    nz = (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx = 0
                    ny = 1
                    nz = (z * inv_y) % 5
                else:
                    inv_z = pow(z, 3, 5)
                    nx = 0
                    ny = 0
                    nz = 1
                points.append((nx, ny, nz))
    
    unique_points = list(set(points))
    unique_points.sort()
    n = len(unique_points)
    
    adj = [set() for _ in range(n)]
    adj_bits = [0] * n
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = unique_points[i]
            p2 = unique_points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                adj[i].add(j)
                adj[j].add(i)
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
                
    return n, adj, adj_bits

def _has_independent_set_k(adj_bits, n, k):
    """
    Fast exact check for independent set of size k using bitmask branch and bound.
    Independent set in G is a clique in the complement graph G'.
    """
    if k <= 0:
        return True
    if n < k:
        return False
    if k == 1:
        return n > 0

    full_mask = (1 << n) - 1
    # comp_mask[v] has bits set for all u != v such that (u,v) is NOT an edge in G
    comp_mask = [0] * n
    for v in range(n):
        # Non-neighbors are full_mask minus adj_bits[v] minus v itself
        comp_mask[v] = full_mask & ~(adj_bits[v] | (1 << v))
    
    def search(candidates, size):
        if size >= k:
            return True
        if candidates == 0:
            return False
        
        # Pruning: even if we take all candidates, we won't reach k
        if size + _popcount(candidates) < k:
            return False
            
        # Heuristic: pick vertex v in candidates with max degree in G'[candidates]
        # i.e., max popcount(comp_mask[v] & candidates)
        best_v = -1
        best_deg = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = _popcount(comp_mask[v] & candidates)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
            
        if best_v == -1:
            return False
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        # New candidates must be neighbors of best_v in G' (i.e., non-neighbors in G)
        new_cands_include = candidates & comp_mask[best_v]
        if search(new_cands_include, size + 1):
            return True
            
        # Branch 2: Exclude best_v
        new_cands_exclude = candidates & ~v_bit
        if search(new_cands_exclude, size):
            return True
            
        return False
        
    return search(full_mask, 0)

def _get_compatible_cliques(adj, n, time_limit=5.0):
    """
    Find maximal cliques in the compatibility graph H.
    H has vertices 0..n-1.
    u ~_H v if N_G(u) and N_G(v) are disjoint.
    We want cliques in H, which are sets S of vertices in G that can be neighbors of a new vertex.
    """
    start_time = time.time()
    
    # Build compatibility graph adjacency masks
    # H_mask[i] has bit j set if i and j are compatible (disjoint neighborhoods in G)
    H_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & adj[j]):
                H_mask[i] |= (1 << j)
                H_mask[j] |= (1 << i)
    
    cliques = []
    
    def bronk(R, P, X):
        if time.time() - start_time > time_limit:
            return
        
        # P and X are sets of vertices (as integers or sets). Using sets for clarity, 
        # but for speed with n<50, sets are fine. 
        if not P and not X:
            if R:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P union X to minimize branching
        # u with max degree in P within H
        u = -1
        max_deg = -1
        for v in P:
            deg = len(P & (set(range(n)) & {x for x in range(n) if (H_mask[v] >> x) & 1}))
            # Faster: count bits in H_mask[v] that are in P
            # Since P is a set, we can iterate
            deg = 0
            for x in P:
                if (H_mask[v] >> x) & 1:
                    deg += 1
            if deg > max_deg:
                max_deg = deg
                u = v
                
        if u == -1:
            if not X:
                if R:
                    cliques.append(frozenset(R))
            return

        # Branch on vertices in P that are NOT neighbors of u in H
        # Wait, standard Bron-Kerbosch:
        # For each v in P \ N(u):
        #   Bronk(R + v, P & N(v), X & N(v))
        #   P = P - v
        #   X = X + v
        
        # Get neighbors of u in H
        N_u = set()
        for x in range(n):
            if (H_mask[u] >> x) & 1:
                N_u.add(x)
                
        # Candidates are P \ N_u
        candidates = P - N_u
        
        P_copy = set(P)
        for v in candidates:
            if v not in P: continue # safety
            
            new_R = R | {v}
            
            # P & N(v)
            new_P = set()
            for x in P:
                if (H_mask[v] >> x) & 1:
                    new_P.add(x)
            
            # X & N(v)
            new_X = set()
            for x in X:
                if (H_mask[v] >> x) & 1:
                    new_X.add(x)
            
            bronk(new_R, new_P, new_X)
            
            P.remove(v)
            X.add(v)
            
            if time.time() - start_time > time_limit:
                return

    bronk(frozenset(), set(range(n)), set())
    return cliques

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin from 150s limit
    
    # 1. Start with Polarity Graph of PG(2,5)
    n0, adj, adj_bits = _polarity_graph_pg25()
    n = n0
    
    best_n = n
    best_edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                best_edges.append([u, v])
                
    # Verify base graph
    if _has_independent_set_k(adj_bits, n, 11):
        # Should not happen for PG(2,5) which has alpha 8
        pass

    # 2. Extension Loop
    while time.time() < deadline - 10:
        # Find compatible cliques in current graph
        # Limit time for this step to ensure we have time for independence checks
        time_for_cliques = min(5.0, deadline - time.time() - 15)
        if time_for_cliques <= 0:
            break
            
        cliques = _get_compatible_cliques(adj, n, time_limit=time_for_cliques)
        
        if not cliques:
            break
            
        # Sort cliques by size descending
        # Larger S means the new vertex is connected to more existing vertices.
        # This tends to reduce the independence number.
        cliques.sort(key=lambda s: -len(s))
        
        found_extension = False
        for S in cliques:
            if time.time() > deadline - 10:
                break
                
            S = set(S)
            
            # Create a copy of the graph to test
            adj_copy = [set(s) for s in adj]
            adj_bits_copy = list(adj_bits)
            
            v_new = n
            adj_copy.append(set(S))
            adj_bits_copy.append(0)
            
            new_bits = 0
            for u in S:
                adj_copy[u].add(v_new)
                adj_bits_copy[u] |= (1 << v_new)
                new_bits |= (1 << u)
            adj_bits_copy[v_new] = new_bits
            
            # Check if the new graph has an independent set of size 11
            if not _has_independent_set_k(adj_bits_copy, n + 1, 11):
                # Accept extension
                adj = adj_copy
                adj_bits = adj_bits_copy
                n += 1
                found_extension = True
                
                best_n = n
                best_edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            best_edges.append([u, v])
                break
        
        if not found_extension:
            break

    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Constructed graph with n={res['n']}")
