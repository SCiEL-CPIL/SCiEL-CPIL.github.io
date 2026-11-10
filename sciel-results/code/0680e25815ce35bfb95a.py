import time
import random
import sys

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def count_bits(x):
        return x.bit_count()

    def greedy_partition(adj, n, deadline):
        # Work on a copy of the adjacency list
        rem_adj = adj[:]
        cliques = []
        
        # Pre-compute a static ordering of edges to ensure determinism 
        # and avoid O(E^2) overhead of scanning all edges repeatedly in a naive way.
        # However, since edges are removed, we just scan u, v and check if edge exists.
        
        # To make it faster, we can prioritize vertices with high remaining degree.
        # But a simple double loop is O(n^2) per check, and we do this until no edges.
        # Total edges ~ n^2/2. If we remove a clique of size k, we remove k(k-1)/2 edges.
        
        # Optimization: Instead of scanning all u,v every time, 
        # we can maintain a list of active vertices or just scan and check.
        # Given n=64, n^2=4096, which is small.
        
        while True:
            if time.time() > deadline:
                break
                
            # Find an edge that still exists
            found_edge = False
            start_u = -1
            start_v = -1
            
            # We can scan in a specific order. 
            # To improve packing, maybe pick the edge with the "best" potential?
            # For simplicity and speed, pick the first available edge.
            # But "first" might lead to small cliques if the graph is sparse in that corner.
            # Let's pick the edge (u,v) such that u has max remaining degree, 
            # then v has max remaining degree among neighbors of u.
            
            max_u_deg = -1
            best_u = -1
            for u in range(n):
                deg = count_bits(rem_adj[u])
                if deg > max_u_deg:
                    max_u_deg = deg
                    best_u = u
            
            if best_u == -1 or max_u_deg == 0:
                break # No edges left
            
            start_u = best_u
            # Find best neighbor v of start_u
            max_v_deg = -1
            best_v = -1
            candidates = rem_adj[start_u]
            while candidates:
                lsb = candidates & -candidates
                v = lsb.bit_length() - 1
                deg = count_bits(rem_adj[v])
                if deg > max_v_deg:
                    max_v_deg = deg
                    best_v = v
                candidates ^= lsb
            
            if best_v == -1:
                break # Should not happen if max_u_deg > 0
                
            # Now we have an edge (start_u, best_v). 
            # Build a clique containing this edge.
            clique = [start_u, best_v]
            # Candidates for adding to clique: common neighbors of all current clique members
            common = rem_adj[start_u] & rem_adj[best_v]
            # Also ensure we don't add vertices that don't connect to all existing clique members
            # 'common' already handles this for the first two.
            
            # Greedily add vertices to the clique
            while common:
                # Pick vertex with max degree in 'common' (or in rem_adj)
                # Using degree in 'common' (induced subgraph) is better for clique growth
                best_cand = -1
                best_deg = -1
                c = common
                while c:
                    lsb = c & -c
                    v = lsb.bit_length() - 1
                    # Degree in the common subgraph
                    deg = count_bits(rem_adj[v] & common)
                    if deg > best_deg:
                        best_deg = deg
                        best_cand = v
                    c ^= lsb
                
                if best_cand == -1:
                    break
                    
                clique.append(best_cand)
                # Update common neighbors
                common &= rem_adj[best_cand]
                
                # Time check inside inner loop? 
                # n is small, clique size is small, so this is fast.
            
            # Remove all edges in the clique
            for i in range(len(clique)):
                u = clique[i]
                for j in range(i + 1, len(clique)):
                    v = clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
            
            cliques.append(clique)
            
            # Check if any edges remain
            has_edges = False
            for u in range(n):
                if rem_adj[u]:
                    has_edges = True
                    break
            if not has_edges:
                break

        # Safety: If we broke due to timeout or bug, ensure all edges are covered.
        # This loop is O(n^2) and fast.
        for u in range(n):
            for v in range(u + 1, n):
                if rem_adj[u] & (1 << v):
                    cliques.append([u, v])
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
                    
        return cliques

    start_time = time.time()
    # Reserve 10s for final processing/safety
    deadline = start_time + 140
    
    adj_1 = instance(32, 0.6, 601)
    adj_2 = instance(64, 0.55, 602)
    
    cliques_1 = greedy_partition(adj_1, 32, deadline)
    cliques_2 = greedy_partition(adj_2, 64, deadline)
    
    return [cliques_1, cliques_2]

if __name__ == "__main__":
    res = construct()
    print(f"Instance 1: {len(res[0])} cliques")
    print(f"Instance 2: {len(res[1])} cliques")
    print(f"Total: {len(res[0]) + len(res[1])}")
