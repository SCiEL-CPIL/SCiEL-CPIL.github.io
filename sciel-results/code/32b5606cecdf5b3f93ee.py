"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Generate Candidate Pool
    # We need a sufficiently large and diverse pool of integer vectors.
    # The condition is ||v - w||^2 >= max(||v||^2, ||w||^2)
    # This is equivalent to 2 * v.dot(w) <= min(||v||^2, ||w||^2)
    
    candidates = []
    
    # Norm 1: e_i, -e_i
    for i in range(D):
        v = np.zeros(D, dtype=np.int64); v[i] = 1; candidates.append(v)
        v = np.zeros(D, dtype=np.int64); v[i] = -1; candidates.append(v)
        
    # Norm 2: e_i +/- e_j
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si; v[j] = sj
                    candidates.append(v)
                    
    # Norm 3: e_i +/- e_j +/- e_k
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        for sk in [1, -1]:
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si; v[j] = sj; v[k] = sk
                            candidates.append(v)
                            
    # Norm 4: Various types
    # Type 1: 4 non-zeros of magnitude 1
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for l in range(k+1, D):
                    for si in [1, -1]:
                        for sj in [1, -1]:
                            for sk in [1, -1]:
                                for sl in [1, -1]:
                                    v = np.zeros(D, dtype=np.int64)
                                    v[i] = si; v[j] = sj; v[k] = sk; v[l] = sl
                                    candidates.append(v)
    # Type 2: 2 non-zeros of magnitude 2
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si * 2; v[j] = sj * 2
                    candidates.append(v)
    # Type 3: 1 non-zero of magnitude 2, 1 non-zero of magnitude 1? No, norm^2 = 4+1=5. 
    # Let's stick to integer norms squared.
    # Norm 4 vectors have norm^2 = 4.
    # 2^2 + 0... = 4.
    # 1^2 + 1^2 + 1^2 + 1^2 = 4.
    # Are there others? No, for norm^2=4, max component is 2.
    # If one component is 2, others must be 0. (Covered by Type 2? No, Type 2 has two 2s which is norm^2=8).
    # Wait.
    # Norm^2 = 4.
    # Components can be:
    # (2, 0, ...) -> 1 vector per axis * 2 signs = 22 vectors.
    # (1, 1, 1, 1, 0...) -> 4 non-zeros.
    # (1, 1, 0, ...) -> Norm^2 = 2.
    # So for Norm^2=4:
    # - 2 on one axis: 22 vectors.
    # - 1 on four axes: 11 choose 4 * 2^4 = 330 * 16 = 5280 vectors.
    
    # Let's add the single 2s for Norm^2=4
    for i in range(D):
        for si in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = si * 2
            candidates.append(v)
            
    # Type 4: 3 non-zeros of magnitude 1? No, that's Norm^2=3.
    # Type 5: 2 non-zeros of magnitude 1? No, that's Norm^2=2.
    
    # What about Norm^2 = 5?
    # (2, 1, 0...) -> 2^2 + 1^2 = 5.
    # Number of such vectors:
    # Choose axis for 2: 11
    # Choose axis for 1: 10
    # Signs: 2 * 2 = 4
    # Total: 11 * 10 * 4 = 440 vectors.
    for i in range(D):
        for j in range(D):
            if i == j: continue
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si * 2
                    v[j] = sj * 1
                    candidates.append(v)
                    
    # What about Norm^2 = 8?
    # (2, 2, 0...) -> 2^2 + 2^2 = 8.
    # Choose 2 axes: C(11,2) = 55
    # Signs: 4
    # Total: 55 * 4 = 220 vectors.
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si * 2
                    v[j] = sj * 2
                    candidates.append(v)
    # (2, 1, 1, 0...) -> 4 + 1 + 1 = 6. No.
    # (1, 1, 1, 1, 1, 1, 1, 1) -> 8 ones. C(11,8)*2^8 = 165 * 256 = 42240. Too many?
    # Let's check if Norm^2=8 with 8 ones is useful.
    # It might be. But let's keep the pool manageable for the graph construction.
    # Current pool size:
    # N1: 22
    # N2: 220
    # N3: 1980
    # N4 (4 ones): 5280
    # N4 (1 two): 22
    # N5 (1 two, 1 one): 440
    # N8 (2 twos): 220
    # Total ~ 8184.
    # This is a good size for a sparse graph.

    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # 2. Build Conflict Graph
    # Two vectors v, w conflict if 2 * v.dot(w) > min(||v||^2, ||w||^2)
    # We want the Maximum Independent Set of this graph.
    
    # Compute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Compute dot products matrix
    # n_cand x n_cand
    # 8000x8000 = 64M floats. 512 MB. Might be tight but okay.
    # If memory is an issue, we can do it in blocks.
    # Let's try direct first.
    
    dots = candidates @ candidates.T
    
    # Compute limits
    # limits[i, j] = min(norms_sq[i], norms_sq[j])
    # We can use broadcasting
    # norms_sq is (n, 1) and (1, n)
    limits = np.minimum(norms_sq[:, None], norms_sq[None, :])
    
    # Conflict condition: 2 * dots > limits
    # Diagonal: 2 * norms_sq > norms_sq -> True. So diagonal is conflict.
    # We want Independent Set, so we want NO conflicts.
    # In MIS, edges represent conflicts.
    # So edge (i, j) exists if 2 * dots[i, j] > limits[i, j]
    
    conflicts = 2 * dots > limits
    
    # Free memory
    del dots
    del limits
    
    # Convert to sparse adjacency matrix
    # csr_matrix expects COO or similar.
    # conflicts is boolean.
    # We need to exclude self-loops for standard MIS definitions? 
    # Actually, MIS on a graph with self-loops is usually defined on the simple graph.
    # But here, a vector always "conflicts" with itself in the sense that we can't pick it twice.
    # Standard MIS algorithms assume no self-loops. We just don't pick the same index twice.
    # So we can ignore the diagonal or set it to False.
    np.fill_diagonal(conflicts, False)
    
    # Convert to CSR
    # scipy.sparse.csr_matrix(conflicts)
    # This might be slow for 8kx8k boolean.
    # Let's use the coordinates.
    row_idx, col_idx = np.where(conflicts)
    data = np.ones(len(row_idx), dtype=np.int8)
    
    # Ensure we handle the case where there are no conflicts (unlikely)
    if len(row_idx) == 0:
         # All compatible? Return all
         return candidates
    
    adj = csr_matrix((data, (row_idx, col_idx)), shape=(n_cand, n_cand))
    
    # 3. Solve MIS using Tomita Algorithm (BB-MCS)
    # We will implement a simplified version of the Tomita algorithm.
    # It uses a coloring bound to prune the search tree.
    
    # The algorithm works on the remaining vertices P.
    # It colors P to get an upper bound on the size of any independent set in P.
    # If current_size + color_bound <= best_size, prune.
    
    # We need to represent the graph efficiently for the recursion.
    # Using bitsets (Python integers) is very fast for graphs up to ~1000-2000 nodes.
    # For 8000 nodes, Python integers are still fast for bitwise ops, but recursion depth and overhead might be high.
    # However, 8000 nodes is small enough for bitsets.
    
    # Convert adjacency to bitsets
    # adj_bits[i] is an integer where bit j is set if (i, j) is an edge.
    adj_bits = [0] * n_cand
    for i in range(n_cand):
        neighbors = adj.indices[adj.indptr[i]:adj.indptr[i+1]]
        b = 0
        for j in neighbors:
            b |= (1 << j)
        adj_bits[i] = b
        
    # We also need to handle the "remaining set" P.
    # P is a bitmask.
    
    # Greedy Coloring
    # Returns the number of colors used, which is an upper bound on the independence number.
    # We also want to return the order of vertices to process (reverse color order).
    # Standard Tomita uses a specific coloring heuristic.
    
    def color_bound(P, adj_bits):
        """
        Returns (num_colors, order)
        order is a list of vertices in P, ordered such that vertices with lower color come first.
        In Tomita, we process vertices in reverse order of coloring (highest color first? or lowest?).
        Actually, the bound is the max color index.
        The search iterates through vertices.
        """
        uncolored = P
        color = 0
        order = []
        
        while uncolored:
            color += 1
            # Greedy coloring: pick a vertex, color it, remove it and its neighbors from uncolored
            # To get a tight bound, we should pick vertices that are "central" or just first available?
            # Standard Tomita uses a specific ordering.
            # Let's use a simple greedy: pick lowest index in uncolored?
            # Or better: pick the vertex with max degree in the remaining subgraph?
            # For speed, let's just pick the first set bit.
            
            # Get lowest set bit
            # v = (uncolored & -uncolored).bit_length() - 1
            v = (uncolored & -uncolored).bit_length() - 1
            
            # Add v to this color class
            order.append(v)
            
            # Remove v and its neighbors from uncolored
            # Neighbors of v are adj_bits[v] & uncolored
            # v itself is in uncolored.
            # We want to remove v and N(v) from uncolored.
            # So uncolored = uncolored & ~(adj_bits[v] | (1 << v))
            
            uncolored &= ~(adj_bits[v] | (1 << v))
            
        return color, order

    # Tomita Search
    best_size = 0
    best_set = set()
    
    # Initial P is all vertices
    P = (1 << n_cand) - 1
    
    # To speed up, we can sort vertices by degree initially?
    # The Tomita algorithm usually works best if the graph is ordered by degree.
    # Let's compute degrees and reorder vertices if necessary.
    # Given the time constraint, let's just run it on the current order.
    # But 8000 nodes might be too deep for pure Python recursion without optimization.
    
    # Optimization: Use iterative deepening or a better heuristic?
    # Let's try a standard recursive implementation with the coloring bound.
    
    sys.setrecursionlimit(20000)
    
    def bb_mcs(P, current_set):
        nonlocal best_size, best_set
        
        if not P:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        # Get coloring bound
        num_colors, order = color_bound(P, adj_bits)
        
        # Prune if current_size + num_colors <= best_size
        if len(current_set) + num_colors <= best_size:
            return
            
        # Process vertices in reverse order of the coloring order?
        # Tomita processes vertices in the order they were colored, but checks them in reverse?
        # Actually, the standard implementation iterates through the vertices in `order`.
        # For each vertex v in order:
        #   if v is in P:
        #      add v to set
        #      P = P & ~adj_bits[v] & ~(1<<v)  # Remove v and neighbors
        #      recurse(P, set + v)
        #      remove v from set
        #      P = P & ~(1<<v) # Remove v from P for subsequent iterations
        
        # Wait, the standard Tomita algorithm is:
        # 1. Color P.
        # 2. For each vertex v in P, in reverse order of color:
        #    a. If v is still in P:
        #       i. Add v to current independent set.
        #       ii. Let P' = P \ (N[v]) (remove v and its neighbors).
        #       iii. Recurse on P'.
        #       iv. Remove v from current independent set.
        #       v. P = P \ {v} (remove v from P for the next iteration of the loop).
        
        # Let's implement this.
        # We need to iterate through `order`.
        # `order` contains vertices from the last color class to the first?
        # In my color_bound, I append to `order` as I color them.
        # Color 1 vertices are appended first. Color 2 next.
        # So `order` is [C1_v1, C1_v2, ..., C2_v1, ...]
        # We should process them in reverse: highest color first?
        # Actually, the bound is the max color.
        # The pruning condition is len(current) + num_colors <= best.
        # When we pick a vertex v from color k, the remaining graph has max color <= k-1?
        # Not exactly, but the coloring of the subgraph P' will have at most num_colors-1 colors?
        # The standard Tomita algorithm relies on the property that if we remove a vertex from color k, the remaining vertices in P' can be colored with at most k-1 colors?
        # No, that's not true in general.
        # However, the algorithm is proven to work.
        
        # Let's iterate `order` in reverse.
        for v in reversed(order):
            if not (P & (1 << v)):
                continue
                
            # Check if we can still beat the best
            # The remaining vertices after removing v and N(v) will have a bound.
            # We can just recurse and let the base case/pruning handle it.
            
            current_set.add(v)
            
            # New P: remove v and its neighbors
            # N[v] = adj_bits[v] | (1 << v)
            new_P = P & ~(adj_bits[v] | (1 << v))
            
            bb_mcs(new_P, current_set)
            
            current_set.remove(v)
            
            # Remove v from P for the next iteration
            P &= ~(1 << v)
            
            # Additional pruning: if len(current_set) + (number of colors remaining in P) <= best_size
            # We can re-color P, but that's expensive.
            # A simpler bound: if len(current_set) + popcount(P) <= best_size: return
            # popcount(P) is an upper bound on the independent set size of P.
            if len(current_set) + bin(P).count("1") <= best_size:
                return

    # Run the solver
    # To avoid timeout, we might need to limit the search or use a heuristic to find a good initial solution.
    # Let's find a greedy independent set first to set best_size.
    
    # Greedy MIS
    P_greedy = (1 << n_cand) - 1
    greedy_set = set()
    while P_greedy:
        # Pick vertex with min degree in remaining graph?
        # Or just first one.
        # Let's pick one with min degree to keep options open.
        # Computing degree for all vertices in P is expensive.
        # Let's just pick the first one.
        v = (P_greedy & -P_greedy).bit_length() - 1
        greedy_set.add(v)
        P_greedy &= ~(adj_bits[v] | (1 << v))
        
    best_size = len(greedy_set)
    best_set = greedy_set
    
    # Now run Tomita
    # We need to be careful with time.
    # If n_cand is large, this might be slow.
    # 8000 nodes.
    # Let's add a time check inside the recursion?
    # Python doesn't easily allow interrupting recursion.
    # We can use a global flag.
    
    time_up = [False]
    
    def bb_mcs_timed(P, current_set):
        nonlocal best_size, best_set, time_up
        
        if time_up[0]:
            return
            
        if not P:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        if time.time() > deadline:
            time_up[0] = True
            return
            
        num_colors, order = color_bound(P, adj_bits)
        
        if len(current_set) + num_colors <= best_size:
            return
            
        for v in reversed(order):
            if time_up[0]:
                return
                
            if not (P & (1 << v)):
                continue
                
            current_set.add(v)
            
            new_P = P & ~(adj_bits[v] | (1 << v))
            
            bb_mcs_timed(new_P, current_set)
            
            current_set.remove(v)
            
            P &= ~(1 << v)
            
            if len(current_set) + bin(P).count("1") <= best_size:
                return

    # Start search
    P_init = (1 << n_cand) - 1
    bb_mcs_timed(P_init, set())
    
    # Extract result
    if best_set:
        indices = list(best_set)
        final_vecs = candidates[indices]
        return final_vecs
    else:
        # Fallback
        return candidates[:1]

# EVOLVE-BLOCK-END
