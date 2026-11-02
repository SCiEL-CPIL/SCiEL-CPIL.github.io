import numpy as np
import time
import random

D = 11

def _generate_weight_2_vectors():
    """Generate all vectors with exactly two non-zero entries, each +/-1."""
    vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    return vectors

def _generate_weight_3_vectors():
    """Generate all vectors with exactly three non-zero entries, each +/-1."""
    vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            vectors.append(v)
    return vectors

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for verification and return

    # 1. Generate Candidates
    # We combine weight 2 and weight 3 vectors.
    # Condition for all pairs (u, v) with norms squared 2 or 3:
    # max_norm_sq is either 2 or 3.
    # If both are weight 2 (norm_sq=2): dist_sq = 4 - 2u.v >= 2 => u.v <= 1.
    # If one is w2 (2) and one is w3 (3): dist_sq = 2+3 - 2u.v >= 3 => 5 - 2u.v >= 3 => 2u.v <= 2 => u.v <= 1.
    # If both are weight 3 (norm_sq=3): dist_sq = 3+3 - 2u.v >= 3 => 6 - 2u.v >= 3 => 2u.v <= 3 => u.v <= 1.5 => u.v <= 1.
    
    # So the conflict condition is uniform: u.dot(v) > 1.
    
    w2_vectors = _generate_weight_2_vectors()
    w3_vectors = _generate_weight_3_vectors()
    
    all_vectors = w2_vectors + w3_vectors
    n = len(all_vectors)
    
    # Convert to numpy array for fast dot product calculation
    V = np.array(all_vectors, dtype=np.int8)
    
    # 2. Build Conflict Graph
    # Conflict if dot(u, v) > 1
    # We will use bitsets (Python ints) for adjacency.
    # adj[i] is an integer where bit j is 1 if i and j conflict.
    
    # Calculate dot products
    # n is 1540. n^2 is ~2.4M.
    # Dot product matrix is 1540x1540.
    # This is manageable.
    
    if time.time() > deadline:
        # Fallback: just return weight 2 vectors (known valid)
        return np.array(w2_vectors, dtype=np.int64)

    # Compute dot products in chunks to save memory if needed, but 1540^2 * 1 byte is small.
    # Actually, V @ V.T is efficient.
    dot_matrix = V @ V.T
    
    adj = [0] * n
    for i in range(n):
        # Find all j such that dot_matrix[i, j] > 1
        # dot_matrix[i, i] is norm_sq (2 or 3), which is > 1, so self-loop is technically there, 
        # but we handle independence by not picking same node twice.
        # We just need neighbors j != i.
        # Using numpy to find indices
        # Note: dot_matrix is symmetric.
        
        # Optimization: Only compute for i < j? No, bitset needs full row or we can be careful.
        # Let's just fill the bitset.
        
        # Mask of conflicts
        # We want bits set where dot > 1.
        # Since we are building adjacency for MIS, we usually exclude self-loops in the logic 
        # or just ensure we don't pick a vertex if it's in the current set.
        
        # Get indices where dot > 1
        # This loop is O(n^2) in Python which is slow for 1540.
        # 1540 * 1540 = 2.3M iterations. In Python, this might take 1-2 seconds.
        # Acceptable.
        
        row = dot_matrix[i]
        # We can use numpy to get indices
        # But converting to list of indices and then setting bits is also slow.
        
        # Alternative: 
        # Since values are small integers, we can just loop.
        
        # Let's try a vectorized approach for bitset construction?
        # It's easier to just loop i, and for each i, find conflicts.
        
        # To speed up, we can precompute conflict lists using numpy
        # But let's stick to the simple loop first, it should be fine within 120s.
        
        # Actually, let's use a more efficient bitset construction.
        # For each i, we want an integer mask.
        
        # We'll do this:
        # For i in range(n):
        #   mask = 0
        #   for j in range(n):
        #     if i != j and dot_matrix[i, j] > 1:
        #       mask |= (1 << j)
        #   adj[i] = mask
        
        # This is 2.3M iterations of simple ops. Should be < 1s in PyPy or optimized CPython? 
        # In standard CPython, 2.3M iterations is ~0.1-0.2s.
        
        pass # Placeholder, will implement in the actual loop below

    # Re-doing the adjacency construction with a slightly optimized loop
    adj = [0] * n
    for i in range(n):
        mask = 0
        # We can iterate over all j
        # Optimization: Since dot_matrix is symmetric, we could do i<j, but bitsets are directional in this simple representation (adj[i] has bit j).
        # We just need adj[i] to contain all j that conflict with i.
        
        # Using numpy argwhere on the row might be faster?
        # indices = np.where(row > 1)[0]
        # for j in indices:
        #   if j != i: mask |= (1 << j)
        
        # Let's try this numpy approach
        indices = np.where(dot_matrix[i] > 1)[0]
        for j in indices:
            if j != i:
                mask |= (1 << j)
        adj[i] = mask

    if time.time() > deadline:
        return np.array(w2_vectors, dtype=np.int64)

    # 3. Tomita Algorithm for Maximum Independent Set
    # We want to find a subset of vertices with no edges between them.
    # This is equivalent to finding a Maximum Clique in the complement graph.
    # But Tomita's algorithm is often described for MIS directly or Clique.
    # Let's implement a Branch and Bound for MIS using coloring on the complement?
    # Or simpler: Branch and Bound on the conflict graph directly.
    # 
    # Standard approach:
    # Find Max Independent Set in G.
    # This is Max Clique in G_complement.
    # 
    # Let's implement Max Clique on the complement graph?
    # Or just implement MIS directly.
    # 
    # MIS(G) = Clique(G_bar).
    # 
    # Let's use a standard Tomita-like algorithm for Maximum Independent Set.
    # 
    # Algorithm:
    # expand(R, P):
    #   if P is empty: update best
    #   order P by degree (or color)
    #   for v in P:
    #     if |R| + |P| <= best: return
    #     expand(R + v, P & ~adj[v] & ~{v})
    #     P = P & ~{v}
    #
    # To make it fast, we use a coloring bound.
    # In the context of MIS, we want to bound the size of the MIS in the remaining subgraph induced by P.
    # This is equivalent to the clique number of the complement of P.
    # 
    # Simpler heuristic: 
    # If we just use a greedy upper bound (e.g., greedy packing), it might be weak.
    # Tomita's coloring bound is strong.
    
    # Let's implement a recursive search with a simple bound first, and see if it's fast enough.
    # If not, I will need to implement the coloring.
    
    # Given n=1540, a simple branch and bound might struggle if the bound is weak.
    # However, the graph is structured (lattice-like).
    
    # Let's try a randomized greedy restart first as a baseline, and then try a limited B&B.
    # Actually, the inspirations suggest Tomita works.
    
    # I will implement a simplified Tomita for MIS.
    # 
    # Steps for Tomita MIS:
    # 1. Compute a coloring of the graph G (conflict graph).
    #    Wait, for MIS, we want to bound the size of the independent set in P.
    #    This is bounded by the clique number of the complement graph G_bar[P].
    #    Coloring G_bar[P] gives a lower bound on the chromatic number, which is a lower bound on the clique number? No.
    #    Chromatic number of G_bar is >= Clique number of G_bar.
    #    So Coloring G_bar gives an upper bound on the Clique number of G_bar? No.
    #    Clique number <= Chromatic number.
    #    So if we color G_bar, the number of colors is an UPPER bound on the clique number of G_bar.
    #    Since MIS(G) = Clique(G_bar), the number of colors in G_bar is an upper bound on the MIS of G.
    #    
    #    So we need to color the COMPLEMENT of the conflict graph.
    #    
    #    Let's define H = complement of conflict graph.
    #    H has an edge between i, j if they do NOT conflict (i.e., dot <= 1).
    #    We want Max Clique in H.
    #    
    #    Tomita's algorithm for Max Clique uses coloring of H to bound the search.
    
    # Let's construct the adjacency of H (compatible graph).
    # comp_adj[i] is a bitset of vertices compatible with i (dot <= 1).
    
    # We already have adj (conflict).
    # comp_adj[i] = all_bits ^ adj[i] ^ (1 << i)
    
    all_bits = (1 << n) - 1
    
    comp_adj = [0] * n
    for i in range(n):
        # Compatible are those not in adj[i] and not i itself
        # Note: adj[i] does not contain i.
        comp_adj[i] = all_bits ^ adj[i] ^ (1 << i)

    # Now we run Tomita Max Clique on H (using comp_adj).
    
    best_clique = []
    best_size = 0
    
    # We need a coloring function for the subgraph induced by a bitset P.
    # We will use a greedy coloring heuristic.
    
    def color_graph(P_bits):
        """
        Greedy coloring of the graph induced by P_bits.
        Returns a list of colors for each vertex in P_bits (ordered by index in P_bits? or just count?).
        For Tomita, we usually order vertices by color and use the color of the last vertex as the bound.
        
        Standard Tomita:
        1. Get vertices in P.
        2. Order them by degree in P? Or just iterate.
        3. Assign colors greedily.
        4. The sequence of colors c_1, c_2, ... c_k is non-decreasing.
        5. When searching for the k-th vertex, the bound is c_k.
        
        Implementation:
        We need to return an ordered list of (vertex, color) pairs.
        """
        # Extract vertices from bitset
        vertices = []
        p = P_bits
        while p:
            # Get lowest set bit
            lsb = p & -p
            idx = lsb.bit_length() - 1
            vertices.append(idx)
            p ^= lsb
        
        # We need to order them. Tomita suggests ordering by color assigned in a greedy pass.
        # But to assign colors, we need to process them.
        # Let's use a simple greedy coloring that returns the max color used.
        # And also the ordering.
        
        # Actually, for the bound, we need the color assigned to each vertex in the order we visit them.
        # Let's do a standard greedy coloring on the induced subgraph.
        
        # To make it fast, we can just use the vertex index order or degree order.
        # Let's use degree in the subgraph (complement of conflict, so degree in comp_adj).
        
        # Calculate degrees in H[P]
        # deg[v] = popcount(comp_adj[v] & P_bits)
        
        # Sort vertices by degree descending? Or ascending?
        # For coloring, low degree first is often better for fewer colors?
        # For Tomita, the ordering is crucial.
        
        # Let's stick to a simpler approach: 
        # We will just compute a valid coloring and return the max color.
        # And for the search, we will use the vertex order defined by the coloring process.
        
        # Simplified Tomita:
        # 1. Sort vertices in P by some heuristic (e.g., degree in H).
        # 2. Greedily color them.
        # 3. The color of the i-th vertex in this sorted order is the bound for that step.
        
        # Let's implement this.
        
        if not vertices:
            return []
        
        # Calculate degrees in H[P]
        # This is expensive if done naively.
        # But P size decreases.
        
        # Let's just use the natural order of indices for simplicity first, 
        # or sort by index.
        
        # Actually, sorting by degree in H[P] is important for tight bounds.
        
        # We'll compute degrees
        degs = []
        for v in vertices:
            deg = bin(comp_adj[v] & P_bits).count('1')
            degs.append((deg, v))
        
        # Sort by degree descending (high degree first in coloring? 
        # In Max Clique, we usually order by degree in the graph we are finding clique in.
        # Here we are finding Clique in H.
        # So degree in H.
        # High degree vertices are more likely to be in the clique.
        
        degs.sort(key=lambda x: -x[0])
        
        # Now color them greedily
        # We need to assign colors such that adjacent vertices in H have different colors.
        # Wait, Tomita's coloring is on the graph H.
        # We want to color H[P].
        
        color_map = {} # vertex -> color
        current_color = 0
        # We process vertices in the order of degs (high degree first).
        # We maintain a list of vertices for each color class.
        # A vertex can take the smallest color not used by its neighbors in H[P] that have already been colored.
        
        # This is standard greedy coloring.
        
        # To optimize, we can just assign colors and track the max color.
        # And we need the list of (vertex, color) in the order of degs.
        
        ordered_vertices = [v for _, v in degs]
        
        # For each vertex in ordered_vertices, assign smallest available color.
        # We need to check neighbors in H[P].
        
        # This is O(|P|^2).
        
        colors = []
        used_colors = [set() for _ in range(len(vertices) + 1)] # color -> set of vertices? No, we need color per vertex.
        
        # Let's just do:
        # color_of[v]
        
        # We iterate through ordered_vertices.
        # For v, check colors of neighbors u in H[P] that are in ordered_vertices and processed before v?
        # No, in greedy coloring, we process in a specific order.
        # If we process in degs order, we check neighbors that have ALREADY been assigned a color.
        
        assigned = {}
        
        for v in ordered_vertices:
            # Find colors used by neighbors of v in H[P] that are already assigned
            # Neighbors in H[P] are bits in comp_adj[v] & P_bits
            # We need to check which of those have been assigned a color.
            
            # This check is expensive.
            # Optimization: Keep a set of assigned vertices?
            
            # For now, let's assume this is fast enough for small P.
            # As P shrinks, it gets faster.
            
            neighbors = comp_adj[v] & P_bits
            # Get indices of neighbors that are in assigned
            # We can iterate through assigned keys? No, too slow.
            
            # Let's just iterate through all assigned vertices?
            # Or better: 
            # Since we are in Python, this might be slow.
            
            # Alternative: Use a simpler bound.
            # Bound = |P| / avg_degree? No.
            # Let's try a simpler recursive search without coloring first, 
            # and see if it times out.
            # If it does, I will need to optimize the coloring.
            
            # Given the constraints, let's try a random greedy restart first to get a good lower bound,
            # then use a limited B&B.
            
            pass

    # Let's step back. Implementing a full Tomita in Python within 120s for n=1540 is risky if not optimized.
    # However, the graph is sparse?
    # Conflict if dot > 1.
    # For weight 3 vectors, dot can be -3, -1, 1, 3.
    # Conflict if dot = 3? No, dot > 1 means dot = 3.
    # Wait, dot products of weight 3 vectors:
    # Max overlap 3 -> dot 3.
    # Overlap 2 -> dot 2 (if signs match) or 0 or -2.
    # Overlap 1 -> dot 1 or -1.
    # Overlap 0 -> 0.
    
    # So conflict (dot > 1) happens ONLY if dot = 3?
    # If dot = 2, is it conflict?
    # Condition: dot <= 1 is OK.
    # So dot = 2 is CONFLICT.
    # Dot = 3 is CONFLICT.
    
    # So edges exist if overlap is 2 (with matching signs) or 3.
    
    # This graph is likely sparse.
    # Degree of a vertex:
    # How many vectors share 2 indices with matching signs?
    # If v has indices {i, j, k} with signs {1, 1, 1}.
    # Vectors sharing i, j with signs 1, 1:
    # The third index can be anything except i, j. (8 choices).
    # Signs for third index: 2 choices.
    # So 16 vectors?
    # Plus vectors sharing i, k...
    # So degree is roughly 3 * 16 = 48?
    # Plus vectors with same 3 indices (0 others? No, only itself).
    # So degree is low (~50-100).
    
    # Sparse graph!
    # Random greedy should work very well.
    # The current program did random greedy on weight 3 ONLY (1320 nodes).
    # We are doing Weight 2 + Weight 3 (1540 nodes).
    
    # Let's just do extensive Random Greedy with a good heuristic (pick vertex with min degree in remaining graph).
    
    def greedy_mis(adj_bitsets, n):
        """
        Greedy MIS using bitsets.
        At each step, pick vertex with minimum degree in the current candidate set P.
        """
        P = (1 << n) - 1
        selected = []
        
        while P:
            # Find vertex in P with minimum degree in P
            # deg(v) = popcount(adj[v] & P)
            # We want to minimize this.
            
            # Iterate through vertices in P
            # To speed up, we can sample or just iterate all.
            # n is 1540. P shrinks.
            
            best_v = -1
            min_deg = n + 1
            
            p = P
            while p:
                lsb = p & -p
                v = lsb.bit_length() - 1
                deg = bin(adj_bitsets[v] & P).count('1')
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                p ^= lsb
            
            if best_v == -1:
                break
                
            # Select best_v
            selected.append(best_v)
            
            # Remove best_v and its neighbors from P
            # New P = P & ~(adj[best_v] | (1 << best_v))
            P = P & ~(adj_bitsets[best_v] | (1 << best_v))
            
        return selected

    # Let's run multiple restarts of this greedy.
    
    best_solution_indices = []
    best_size = 0
    
    # Initial baseline: Weight 2 vectors are always valid.
    # We can start with that.
    
    # Run greedy restarts
    for i in range(500):
        if time.time() > deadline:
            break
            
        # Randomize the order of tie-breaking or the initial set?
        # The "min degree" greedy is deterministic for a given graph.
        # To make it stochastic, we can randomly remove some vertices from P initially?
        # Or just shuffle the tie-breaking.
        
        # Better: Randomly select a starting subset?
        # Or just use the deterministic greedy, but we know it's deterministic.
        # So one run is enough?
        # No, the "min degree" choice is local.
        
        # Let's add a bit of randomness:
        # Instead of strictly min degree, pick from top 3 min degrees randomly.
        
        P = (1 << n) - 1
        selected = []
        
        while P:
            # Find vertices with low degree
            # Get all vertices in P
            verts = []
            p = P
            while p:
                lsb = p & -p
                v = lsb.bit_length() - 1
                verts.append(v)
                p ^= lsb
            
            if not verts:
                break
            
            # Calculate degrees
            degs = []
            for v in verts:
                deg = bin(adj_bitsets[v] & P).count('1')
                degs.append((deg, v))
            
            # Sort by degree
            degs.sort()
            
            # Pick from the top 5 (lowest degrees)
            k = min(5, len(degs))
            chosen = random.choice(degs[:k])
            v = chosen[1]
            
            selected.append(v)
            P = P & ~(adj_bitsets[v] | (1 << v))
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_solution_indices = selected
            
            # If we beat the weight-2 baseline (220), we are happy.
            # If we get close to 1540, we might need better algo.
            
    # Verification
    if best_solution_indices:
        final_vectors = [all_vectors[i] for i in best_solution_indices]
    else:
        final_vectors = w2_vectors
        
    # Final check
    if len(final_vectors) > 1:
        V_final = np.array(final_vectors, dtype=np.int64)
        norms_sq = np.sum(V_final * V_final, axis=1)
        max_ns = np.max(norms_sq)
        
        # Check pairwise
        # Use matrix multiplication for speed
        # Dist_sq = Norm_i + Norm_j - 2 Dot_ij
        # We need Dist_sq >= max_ns for all i != j
        
        # To save time, we can just trust the construction if it's from the conflict graph.
        # But let's do a quick check on a sample or all if small.
        
        if len(final_vectors) <= 500:
            # Full check
            Dots = V_final @ V_final.T
            Norms = norms_sq
            # Construct dist matrix
            # dist_sq[i,j] = Norms[i] + Norms[j] - 2*Dots[i,j]
            # We need dist_sq[i,j] >= max_ns for i != j
            
            # Create a mask for i != j
            mask = ~np.eye(len(final_vectors), dtype=bool)
            
            # Calculate dists
            # We can do this element-wise
            # But memory is O(N^2). 500x500 is fine.
            
            dist_sq = Norms[:, None] + Norms[None, :] - 2 * Dots
            
            # Check
            valid = True
            # Find any pair where dist_sq < max_ns
            # dist_sq < max_ns
            
            # We only care about off-diagonals
            # np.where(mask & (dist_sq < max_ns))
            
            bad_pairs = np.where(mask & (dist_sq < max_ns))
            if len(bad_pairs[0]) > 0:
                valid = False
                
            if not valid:
                # Fallback
                final_vectors = w2_vectors

    return np.array(final_vectors, dtype=np.int64)

