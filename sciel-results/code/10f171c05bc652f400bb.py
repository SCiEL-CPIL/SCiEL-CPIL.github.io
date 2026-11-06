import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 33
    
    # Precompute valid 5-vertex subgraphs (those NOT containing a C5)
    # A C5 is a 5-cycle. In a 5-vertex graph, a C5 exists if the subgraph contains a 5-cycle.
    # Note: "contains a C5 as a subgraph" means the induced subgraph on those 5 vertices must not have a C5?
    # No, "subgraph" usually means not necessarily induced.
    # If a graph H contains a C5 as a subgraph, it means there are 5 vertices that form a C5.
    # For a 5-vertex set S, G[S] contains a C5 subgraph if and only if G[S] has a C5 as a spanning subgraph?
    # Actually, if G[S] has more than 5 edges, it might contain a C5.
    # If G[S] has exactly 5 edges, it is a C5 iff it is a 5-cycle.
    # If G[S] has > 5 edges, does it always contain a C5?
    # No. K5 contains C5. K4 + isolated vertex does not.
    # We need to check if G[S] contains ANY C5.
    # Since |S|=5, a C5 must use all 5 vertices.
    # So G[S] contains a C5 iff G[S] has a spanning 5-cycle subgraph.
    # This is equivalent to checking if the 5 vertices can be ordered v1..v5 such that (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v1) are edges in G.
    
    # Let's precompute for all 1024 possible edge sets on 5 vertices whether they contain a C5.
    # Vertices 0..4. Edges (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4).
    # Map edge (i,j) to bit index.
    edge_indices = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    edge_to_bit = {}
    for i, (u, v) in enumerate(edge_indices):
        edge_to_bit[(u, v)] = i
        edge_to_bit[(v, u)] = i
        
    valid_5_graphs = [False] * 1024
    for mask in range(1024):
        adj = [[False]*5 for _ in range(5)]
        for i in range(10):
            if mask & (1 << i):
                u, v = edge_indices[i]
                adj[u][v] = True
                adj[v][u] = True
        # Check if there is a C5
        # Permutations of 1,2,3,4 (0 is fixed)
        has_c5 = False
        for p in [(1,2,3,4), (1,2,4,3), (1,3,2,4), (1,3,4,2), (1,4,2,3), (1,4,3,2)]:
            if adj[0][p[0]] and adj[p[0]][p[1]] and adj[p[1]][p[2]] and adj[p[2]][p[3]] and adj[p[3]][0]:
                has_c5 = True
                break
        valid_5_graphs[mask] = not has_c5
        
    # Local search
    # Represent graph as a list of bitmasks for adjacency, or just a set of edges.
    # For n=33, a set of edges is fine, but bitsets are faster for checking.
    # Let's use a list of ints for adjacency masks. adj[i] is a bitmask of neighbors of i.
    
    def is_valid(adj):
        # Check C5
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        for m in range(l+1, n):
                            # Extract edges
                            mask = 0
                            # (i,j)
                            if adj[i] & (1 << j): mask |= 1 << 0
                            # (i,k)
                            if adj[i] & (1 << k): mask |= 1 << 1
                            # (i,l)
                            if adj[i] & (1 << l): mask |= 1 << 2
                            # (i,m)
                            if adj[i] & (1 << m): mask |= 1 << 3
                            # (j,k)
                            if adj[j] & (1 << k): mask |= 1 << 4
                            # (j,l)
                            if adj[j] & (1 << l): mask |= 1 << 5
                            # (j,m)
                            if adj[j] & (1 << m): mask |= 1 << 6
                            # (k,l)
                            if adj[k] & (1 << l): mask |= 1 << 7
                            # (k,m)
                            if adj[k] & (1 << m): mask |= 1 << 8
                            # (l,m)
                            if adj[l] & (1 << m): mask |= 1 << 9
                            
                            if not valid_5_graphs[mask]:
                                return False, (i,j,k,l,m)
        # Check alpha < 9 (i.e., no K9 in complement)
        # Complement adjacency: ~adj[i] & ((1<<n)-1) & ~(1<<i)
        # Check for K9 in complement graph.
        # Use backtracking or simple check for n=33?
        # Simple check: iterate all 9-sets? C(33,9) is huge.
        # Use bitset clique finding.
        # Max clique in complement graph.
        # We just need to know if there is a clique of size 9.
        # Let's use a simple recursive search with pruning.
        
        comp_adj = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp_adj[i] = all_mask & ~(1 << i) & ~adj[i]
            
        def find_clique_size(adj_list, target_size):
            # Returns True if clique of size target_size exists
            # adj_list is list of bitmasks
            # Use recursive search
            # To speed up, we can use a greedy coloring bound or just simple backtracking.
            # For n=33, simple backtracking might be slow if many candidates.
            # But we only call this if C5 check passes, which is rare in random search?
            # Actually, we want to find a valid graph.
            
            # Let's use a simpler check: 
            # If the complement graph is sparse, K9 is unlikely.
            # We can try to find a K9.
            
            # Initialize candidates
            # We'll do a DFS.
            # To make it fast, we can pick a vertex, then intersect neighbors.
            
            # This is a standard max clique algorithm (Bron-Kerbosch variant or just bounded search).
            # Since we only care about size 9, we can stop early.
            
            # Let's use a simple iterative deepening or just check a few random 9-sets? No, must be exact.
            
            # Efficient check:
            # If max degree < 8, no K9.
            # If any vertex has degree >= 8, check its neighbors.
            
            # Let's implement a fast K9 checker using bitsets.
            # Pseudo-code:
            # def has_k9(adj):
            #   candidates = all vertices
            #   for v in candidates:
            #     if popcount(adj[v]) < 8: continue
            #     next_candidates = candidates & adj[v]
            #     if has_k8(next_candidates, adj): return True
            #   return False
            
            # We can implement this recursively.
            
            def has_kk(adj_list, k, candidates):
                if k == 0: return True
                if candidates == 0: return False
                # Pick a vertex v from candidates
                # To prune, we should pick v with high degree in the induced subgraph?
                # Or just iterate.
                # Optimization: if popcount(candidates) < k: return False
                if bin(candidates).count('1') < k: return False
                
                # Iterate through vertices in candidates
                c = candidates
                while c:
                    lsb = c & -c
                    v = lsb.bit_length() - 1
                    c ^= lsb
                    # Intersect with neighbors
                    new_cand = candidates & adj_list[v]
                    # Remove v and previous candidates to avoid duplicates?
                    # Standard trick: new_cand = new_cand & (candidates - (lsb - 1))? No.
                    # We just need to find ONE clique.
                    if has_kk(adj_list, k-1, new_cand):
                        return True
                return False

            return has_kk(comp_adj, 9, all_mask)

        if find_clique_size(comp_adj, 9):
            return False, None
        return True, None

    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Initialization: Start with a graph likely to be valid.
    # 8 x K4 is valid for n=32.
    # For n=33, let's try to extend it.
    # Or start random and repair.
    
    # Let's try a few restarts.
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Initialize adj
        # Strategy: Start with 8 K4s and one extra vertex connected to make it work?
        # Or just random graph with density ~0.5?
        # C5-free graphs are usually sparse or structured.
        # Let's try starting with the 32-vertex solution and adding a vertex.
        adj = [0] * n
        # Build 8 K4s
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i+1, 4):
                    u = base + i
                    v = base + j
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        
        # Vertex 32 is isolated so far.
        # Connect vertex 32 to some vertices.
        # To avoid C5, we should connect it to a clique or independent set carefully.
        # If we connect 32 to all vertices of K4_0, we get K5, which has C5.
        # If we connect 32 to 1 vertex in each K4, we might get C5s involving 32.
        
        # Let's just randomize the connections of vertex 32.
        for i in range(32):
            if random.random() < 0.5:
                adj[32] |= 1 << i
                adj[i] |= 1 << 32
        
        # Local search
        valid, violation = is_valid(adj)
        if valid:
            return {'n': n, 'edges': get_edges(adj)}
        
        # If not valid, try to fix.
        # This is complex. Let's just do random restarts with different initializations.
        # Maybe increase density or change structure.
        
        # Try a different initialization: Random graph with p=0.5
        if restart % 2 == 0:
             adj = [0] * n
             for i in range(n):
                 for j in range(i+1, n):
                     if random.random() < 0.5:
                         adj[i] |= 1 << j
                         adj[j] |= 1 << i
             valid, violation = is_valid(adj)
             if valid:
                 return {'n': n, 'edges': get_edges(adj)}
        
        # Try p=0.3
        else:
             adj = [0] * n
             for i in range(n):
                 for j in range(i+1, n):
                     if random.random() < 0.3:
                         adj[i] |= 1 << j
                         adj[j] |= 1 << i
             valid, violation = is_valid(adj)
             if valid:
                 return {'n': n, 'edges': get_edges(adj)}

    # Fallback to n=32
    n = 32
    adj = [0] * n
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i+1, 4):
                u = base + i
                v = base + j
                adj[u] |= 1 << v
                adj[v] |= 1 << u
    return {'n': n, 'edges': get_edges(adj)}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
