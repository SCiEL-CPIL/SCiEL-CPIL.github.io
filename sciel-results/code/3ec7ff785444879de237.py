import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 290

    def time_left():
        return deadline - time.time()

    # Fallback: Paley Graph P(37)
    def get_paley_37():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return q, edges

    # Try to find a graph of size N using Local Search
    def try_find_graph(N, time_limit):
        # Adjacency as list of bitmasks
        # adj[i] is a bitmask of neighbors of i
        adj = [0] * N
        edges = []
        
        def init_random():
            nonlocal adj, edges
            adj = [0] * N
            edges = []
            for i in range(N):
                for j in range(i + 1, N):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        edges.append((i, j))
            
        def count_violations():
            # Count K5s and Ind5s
            # To be fast, we can just count them.
            # But for local search, we need delta.
            # Let's implement a full count first for initialization.
            k5_count = 0
            ind5_count = 0
            
            # Optimized count using bitsets
            # Iterate over all 5-combinations? No, too slow in Python.
            # Use the property:
            # A set {u,v,w,x,y} is a K5 if all pairs are edges.
            # It is an Ind5 if no pairs are edges.
            
            # We can count K5s by iterating over triangles?
            # Or just use a simple check for all 5-subsets if N is small (<=42).
            # C(42, 5) = 850668.
            # 850k iterations * 10 checks = 8.5M ops.
            # In Python, this might take ~1-2 seconds.
            # We need to do this for initial state.
            # For delta, we use the incremental formula.
            
            import itertools
            for combo in itertools.combinations(range(N), 5):
                # Check if K5
                is_k5 = True
                is_ind5 = True
                for i in range(5):
                    for j in range(i+1, 5):
                        u = combo[i]
                        v = combo[j]
                        if adj[u] & (1 << v):
                            is_ind5 = False
                        else:
                            is_k5 = False
                        if not is_k5 and not is_ind5:
                            break
                    if not is_k5 and not is_ind5:
                        break
                if is_k5:
                    k5_count += 1
                if is_ind5:
                    ind5_count += 1
            return k5_count + ind5_count

        def delta_flip(u, v):
            # Returns (delta_k5, delta_ind5) for flipping edge (u,v)
            # Currently edge exists?
            edge_exists = bool(adj[u] & (1 << v))
            
            # Common neighbors
            common_neigh = adj[u] & adj[v]
            # Common non-neighbors (excluding u and v)
            # All vertices except u, v
            all_mask = (1 << N) - 1
            u_mask = 1 << u
            v_mask = 1 << v
            others_mask = all_mask & ~u_mask & ~v_mask
            
            # Non-neighbors of u among others
            non_neigh_u = others_mask & ~adj[u]
            # Non-neighbors of v among others
            non_neigh_v = others_mask & ~adj[v]
            common_non_neigh = non_neigh_u & non_neigh_v
            
            num_common_neigh = bin(common_neigh).count("1")
            num_common_non_neigh = bin(common_non_neigh).count("1")
            
            if edge_exists:
                # Removing edge (u,v)
                # K5s that used (u,v) are destroyed.
                # A K5 containing (u,v) requires 3 other vertices that are neighbors to both u and v.
                # Number of such triples is C(num_common_neigh, 3)
                # Wait, the other 3 vertices must form a triangle among themselves?
                # Yes, for {u,v,a,b,c} to be a K5, a,b,c must be connected to u,v AND each other.
                # So we need to count triangles in the subgraph induced by common_neigh.
                # This is expensive to calculate exactly for every flip.
                # Approximation or simpler metric?
                # The task requires *no* K5 and *no* Ind5.
                # If we use a soft metric, we might get stuck.
                # Let's stick to the exact count if possible, or a good heuristic.
                
                # Actually, for N=42, C(42,5) is small enough that we can just re-count?
                # No, re-counting takes ~1s. We want ~1000 flips/sec.
                
                # Let's use a different approach:
                # Random restart local search on the *vertex set* is hard.
                # Let's try the "Maximal Independent Set" style? No.
                
                # Let's go back to the Paley graph extension.
                # The previous code failed because it was slow.
                # What if we just use a faster check for the Paley extension?
                # The Paley graph P(37) is known.
                # Can we add 5 vertices?
                # The known 42-vertex graph is not a supergraph of P(37).
                
                # Let's try a different seed: Random Graph G(n, p) and Local Search.
                # To make delta calculation fast, we can approximate the number of K5s/Ind5s.
                # Or, we can just use the full count if we limit the number of restarts.
                # 300s / 1s per count = 300 restarts.
                # In each restart, we do 1000 flips.
                # Total 300,000 flips.
                # This might be enough to find a solution if we are lucky.
                
                # Let's optimize the count.
                # We can precompute for each pair (u,v) the number of K5s and Ind5s it participates in?
                # That's expensive to update.
                
                # Alternative: Use a SAT solver? Not available.
                
                # Let's try to improve the Paley extension code with a much faster check.
                # The check for a new vertex mask M against existing graph G:
                # Violations = (# K4s in G that become K5 with M) + (# Ind4s in G that become Ind5 with M)
                # + (# K4s formed by M and 3 vertices of G) ... wait.
                # If we add vertex v with neighborhood S in G:
                # New K5s:
                # 1. v + 4 vertices from S that form a K4 in G.
                # 2. v + 3 vertices from S + 1 vertex from \bar{S} ... No, v is not connected to \bar{S}.
                # So any K5 containing v must have the other 4 vertices in S.
                # And those 4 vertices must form a K4 in G.
                # So, # New K5s = # K4s in G[S].
                # Similarly, # New Ind5s = # Ind4s in G[\bar{S}].
                
                # This is much faster!
                # We just need to count K4s in the induced subgraph on S and Ind4s in \bar{S}.
                # S and \bar{S} are subsets of the 37 vertices.
                # Counting K4s in a subgraph of size <= 37.
                # We can precompute K4s of the base graph?
                # There are many K4s.
                # But we can just iterate over the 4-subsets of S and check if they are K4s.
                # |S| can be up to 37. C(37,4) ~ 66k.
                # This is fast enough to do for each candidate mask.
                
                # So, the strategy:
                # 1. Start with P(37).
                # 2. Try to add vertex 37.
                # 3. Candidate mask M (37 bits).
                # 4. Cost(M) = Count_K4(G[M]) + Count_Ind4(G[~M]).
                # 5. Local search on M to minimize Cost(M).
                # 6. If Cost(M) == 0, add vertex.
                # 7. Update G.
                # 8. Repeat.
                
                # This is the same as the original code, but the cost function is:
                # Count K4s in the subgraph induced by neighbors.
                # The original code iterated over *all* K4s of G and checked if they are subset of M.
                # That is O(#K4s). #K4s in P(37) is large.
                # My new method iterates over 4-subsets of M.
                # If M is sparse, this is fast. If M is dense, it's slower.
                # But we can optimize Count_K4(G[S]) by precomputing a list of K4s in G and checking membership?
                # Or just iterate 4-subsets of S.
                
                # Let's implement this improved Paley extension.
                
                pass # Placeholder for logic in the actual edit

        # The above analysis suggests the "Improved Paley Extension" is the way to go.
        # The key is the cost function:
        # cost(mask) = num_k4s_in_induced_subgraph(mask) + num_ind4s_in_induced_subgraph(~mask)
        
        # We can precompute all K4s and Ind4s of the *current* graph G.
        # Then, for a candidate mask M (neighborhood of new vertex):
        # num_k4s_in_G[M] = count of precomputed K4s that are subsets of M.
        # num_ind4s_in_G[~M] = count of precomputed Ind4s that are subsets of ~M.
        
        # This is what the original code tried to do with `k4_containing`.
        # Why did it fail?
        # 1. `count_violations` iterated over ALL k4_masks.
        # 2. `try_add_vertex` used a random mask and local search.
        # 3. The local search used `k4_containing` to compute delta.
        
        # The problem might be that P(37) is not extendable by simple vertex addition?
        # Or the search space is too rugged.
        
        # Let's try a different seed: The (5,5)-free graph on 42 vertices is known.
        # Can we find it by random search on N=42?
        
        # Let's stick to the "Local Search on Edge Set for N=42" but with a better delta.
        # Since exact delta is hard, we can use a "Greedy" approach:
        # Minimize the number of "Almost K5s" and "Almost Ind5s".
        # Or just use the full count every K steps.
        
        # Given the time constraints and the difficulty, let's try the "Improved Paley Extension" with a more robust local search.
        # We will use the `cost(mask)` defined above.
        
        return None # Placeholder

    # Fallback
    return get_paley_37()
