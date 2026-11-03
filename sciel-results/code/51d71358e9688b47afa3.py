import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    # 1. Generate all vectors
    all_vectors = []
    support_map = {}
    
    idx = 0
    for comb in itertools.combinations(range(D), 4):
        support = comb
        support_map[support] = []
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in zip(support, signs):
                v[k] = s
            all_vectors.append((idx, support, signs, v))
            support_map[support].append(idx)
            idx += 1
            
    N = len(all_vectors)
    
    # 2. Build Conflict Graph
    # adj[i] is a list of neighbors
    adj = [[] for _ in range(N)]
    
    supports = list(support_map.keys())
    sup_to_vecs = {s: support_map[s] for s in supports}
    
    # Conflict Condition:
    # 1. Same vector (handled by unique idx, so no self-loops needed, but we don't add self)
    # 2. Intersection size 3 and signs on intersection are identical.
    
    for i_s, s1 in enumerate(supports):
        for i_t, s2 in enumerate(supports):
            if i_t <= i_s: continue
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                # Get indices in s1 and s2 corresponding to inter
                # s1 and s2 are tuples of length 4
                # inter is a set of 3 elements
                
                # Map element to index in s1
                pos_in_s1 = {x: i for i, x in enumerate(s1)}
                pos_in_s2 = {x: i for i, x in enumerate(s2)}
                
                # Get the specific indices in the sign tuples
                p1 = [pos_in_s1[x] for x in inter]
                p2 = [pos_in_s2[x] for x in inter]
                
                vecs_s1 = sup_to_vecs[s1]
                vecs_s2 = sup_to_vecs[s2]
                
                for idx1 in vecs_s1:
                    signs1 = all_vectors[idx1][2]
                    # Check if signs on intersection are all same
                    # We need to compare signs1[p1[0]], signs1[p1[1]], signs1[p1[2]]
                    # with signs2[p2[0]], signs2[p2[1]], signs2[p2[2]]
                    
                    # Optimization: Precompute signature for each vector on this intersection?
                    # Since inter is fixed for this pair of supports, we can just extract.
                    sig1_0 = signs1[p1[0]]
                    sig1_1 = signs1[p1[1]]
                    sig1_2 = signs1[p1[2]]
                    
                    for idx2 in vecs_s2:
                        signs2 = all_vectors[idx2][2]
                        sig2_0 = signs2[p2[0]]
                        sig2_1 = signs2[p2[1]]
                        sig2_2 = signs2[p2[2]]
                        
                        if sig1_0 == sig2_0 and sig1_1 == sig2_1 and sig1_2 == sig2_2:
                            # Conflict
                            adj[idx1].append(idx2)
                            adj[idx2].append(idx1)

    # 3. Solve MIS using Tomita's Algorithm (Branch and Bound with Greedy Coloring)
    
    # Greedy Coloring for Upper Bound
    def greedy_color(candidates):
        """
        Returns a list of color classes.
        candidates: list of vertices
        """
        color_classes = []
        uncolored = set(candidates)
        
        while uncolored:
            color_class = []
            # Pick a vertex
            # Heuristic: pick vertex with max degree in uncolored?
            # For speed, just pick arbitrary (min id)
            v = min(uncolored)
            uncolored.remove(v)
            color_class.append(v)
            
            # Remove neighbors of v from uncolored for this color class
            # Neighbors of v that are in uncolored cannot share this color
            to_remove = set()
            for u in uncolored:
                if u in adj[v]:
                    to_remove.add(u)
            
            uncolored -= to_remove
            
            # The rest of uncolored can potentially be in this color class?
            # No, greedy coloring:
            # A color class is an independent set.
            # We select v. Then we cannot select any neighbor of v.
            # So we remove N(v) from the pool of candidates for *this* color.
            # The remaining uncolored vertices are candidates for *subsequent* colors.
            # Wait, the standard greedy coloring for MIS bound:
            # We want to partition the graph into independent sets.
            # 1. Pick a vertex v.
            # 2. Put v in color 1.
            # 3. Remove v and N(v) from the graph? No.
            # 3. Remove N(v) from the *current* color candidate set?
            # Actually, the algorithm is:
            # C = []
            # U = candidates
            # while U:
            #   pick v in U
            #   C.append({v})
            #   U = U \ (N(v) U {v})  # This is for finding a Maximal Independent Set, not coloring.
            
            # Correct Greedy Coloring for Bound:
            # We want to color the graph such that adjacent vertices have different colors.
            # The number of colors is an upper bound on the size of the MIS.
            # Algorithm:
            # 1. Start with empty color classes.
            # 2. While there are uncolored vertices:
            #    a. Start a new color class.
            #    b. Iterate through uncolored vertices.
            #    c. If a vertex is not adjacent to any vertex already in the current color class, add it.
            #    d. Mark it as colored.
            
            # Let's implement this properly.
            pass # Re-implementing below

    def greedy_color_bound(candidates):
        """
        Returns the number of colors used.
        candidates: list of vertices
        """
        color_count = 0
        uncolored = set(candidates)
        
        while uncolored:
            color_count += 1
            # Start a new color class
            # Pick a vertex from uncolored
            # To make it deterministic and fast, pick min
            # We will greedily add vertices to this color class if they don't conflict with existing ones in this class
            current_color_class = []
            # We need to iterate through a copy of uncolored to avoid modification during iteration
            # Or just build the class and then remove from uncolored
            
            # Optimization: 
            # Instead of checking against the whole current_color_class for each candidate,
            # we can just pick one, remove its neighbors, pick next, etc?
            # No, that's finding an independent set.
            # Coloring:
            # For each vertex in uncolored:
            #   If it is not adjacent to any vertex in current_color_class, add it.
            
            # This is O(V^2) per color.
            # V=5280. Colors ~ 20-30?
            # 5280^2 * 30 is too slow in Python.
            
            # Alternative: 
            # The bound is just the number of colors.
            # A simpler bound: 
            # The size of the MIS is at most the size of the maximum independent set of the subgraph.
            # Greedy Independent Set size is a lower bound.
            # Upper bound: 
            # If we color the graph, the number of colors is an upper bound.
            
            # Let's use a simpler, faster upper bound:
            # The trivial bound is len(candidates).
            # A better bound: 
            # We can compute a greedy independent set size (lower bound) and use that to update best.
            # For upper bound, we can use the fact that the graph is sparse.
            # However, Tomita's algorithm relies on coloring.
            
            # Let's try to implement a fast greedy coloring.
            # We maintain a list of available vertices for the current color.
            # Initially available = uncolored.
            # While available:
            #   pick v (e.g., max degree in available? or just min)
            #   add v to color class
            #   remove v and N(v) from available
            # This finds a *Maximal* Independent Set, which is not a coloring.
            # A coloring requires that *all* vertices are assigned a color.
            
            # Actually, for the bound, we just need *any* valid coloring.
            # A greedy algorithm that picks a vertex, assigns it a color, and removes its neighbors from the *current color pool* is incorrect for coloring.
            # Correct Greedy Coloring:
            # 1. Sort vertices (e.g., by degree).
            # 2. For each vertex, assign the smallest available color not used by its neighbors.
            # This is O(V*E) or similar.
            
            # Given the time constraints and Python speed, let's use a simpler heuristic for the bound:
            # Bound = len(candidates) // 2 ? No.
            
            # Let's stick to the Branch and Bound with:
            # 1. Lower bound update via Greedy MIS.
            # 2. Upper bound via simple count (len(candidates)).
            # 3. Branching on max degree vertex.
            # This is what the current code does, but with lists.
            # To improve, we need a tighter upper bound.
            
            # Let's try to implement a fast greedy coloring.
            # We can use a bitset for the current color class to check conflicts quickly?
            # No, let's just use the simple approach and hope the pruning is good enough.
            # Or, we can use the fact that the graph is very sparse.
            
            # Let's revert to the Bitmask approach but with a better branching heuristic?
            # The previous code was slow because of the recursion depth and Python overhead.
            # Using lists might be faster if we avoid the bit operations.
            
            # Let's implement the solver with lists and a simple greedy coloring bound.
            
            # To speed up coloring, we can pre-sort vertices by degree.
            # But the subgraph changes.
            
            # Let's try a different approach: 
            # Use `networkx`? No, standard library only.
            
            # Let's optimize the list-based B&B.
            # 1. Greedy MIS to update best.
            # 2. If len(candidates) + current <= best, prune.
            # 3. Pick vertex v with max degree in candidates.
            # 4. Branch include v (candidates = candidates \ N[v])
            # 5. Branch exclude v (candidates = candidates \ v)
            
            # This is the same as before. Why was it slow?
            # Maybe the conflict graph construction was the bottleneck?
            # Or the recursion limit?
            
            # Let's ensure the graph construction is fast.
            # And use `sys.setrecursionlimit`? No, iterative is better.
            
            # Let's try to implement the Tomita coloring bound properly.
            # It is known to be very effective.
            
            # Fast Greedy Coloring:
            # We maintain a list of uncolored vertices.
            # We repeatedly pick a vertex, assign it to the current color, and remove it and its neighbors from the *uncolored* set?
            # No, that's not coloring.
            
            # Correct Greedy Coloring for Bound:
            # 1. Order vertices by degree (descending).
            # 2. For each vertex, assign the lowest color index such that no neighbor has that color.
            # 3. The max color index is the bound.
            
            # Complexity: O(V * MaxDegree).
            # V=5280, MaxDegree ~ 100?
            # 5280 * 100 = 528,000 ops per bound calculation.
            # If we call this at each node of the tree, it might be slow.
            # But it prunes much better.
            
            # Let's implement this.
            
            pass

    # Implementation of Tomita's Algorithm
    
    # Pre-sort vertices by degree for coloring heuristic?
    # Degrees change in subgraphs.
    
    best_size = 0
    best_set = []
    
    def greedy_mis(candidates):
        """Find a maximal independent set greedily."""
        selected = []
        remaining = set(candidates)
        while remaining:
            # Pick vertex with min degree in remaining (for MIS size) or max?
            # Min degree usually yields larger independent sets.
            min_deg = float('inf')
            v = -1
            for u in remaining:
                deg = len([x for x in adj[u] if x in remaining])
                if deg < min_deg:
                    min_deg = deg
                    v = u
            selected.append(v)
            # Remove v and its neighbors
            to_remove = set(adj[v])
            to_remove.add(v)
            remaining -= to_remove
        return selected

    def color_bound(candidates):
        """Compute greedy coloring bound."""
        if not candidates:
            return 0
        
        # Sort by degree in the subgraph?
        # For simplicity, use global degree or just arbitrary order.
        # Arbitrary order is fast but might give loose bound.
        # Let's use degree in the subgraph.
        
        # Calculate degrees
        cand_set = set(candidates)
        degrees = {}
        for u in candidates:
            deg = 0
            for v in adj[u]:
                if v in cand_set:
                    deg += 1
            degrees[u] = deg
            
        # Sort by degree descending
        sorted_cands = sorted(candidates, key=lambda x: degrees[x], reverse=True)
        
        colors = {}
        max_color = 0
        
        for u in sorted_cands:
            # Find smallest color not used by neighbors
            used_colors = set()
            for v in adj[u]:
                if v in colors:
                    used_colors.add(colors[v])
            
            c = 1
            while c in used_colors:
                c += 1
                
            colors[u] = c
            if c > max_color:
                max_color = c
                
        return max_color

    def solve(candidates, current_set, current_size):
        nonlocal best_size, best_set
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
        
        # Update best with greedy solution
        g_set = greedy_mis(candidates)
        if current_size + len(g_set) > best_size:
            best_size = current_size + len(g_set)
            best_set = current_set + g_set
            
        # Upper bound
        bound = color_bound(candidates)
        if current_size + bound <= best_size:
            return
        
        # Select vertex to branch on
        # Heuristic: Max degree in candidates
        cand_set = set(candidates)
        max_deg = -1
        v = -1
        for u in candidates:
            deg = len([x for x in adj[u] if x in cand_set])
            if deg > max_deg:
                max_deg = deg
                v = u
                
        if v == -1:
            return
            
        # Branch 1: Include v
        # New candidates: candidates \ (N(v) U {v})
        new_cands_1 = [u for u in candidates if u != v and u not in adj[v]]
        solve(new_cands_1, current_set + [v], current_size + 1)
        
        # Branch 2: Exclude v
        new_cands_2 = [u for u in candidates if u != v]
        solve(new_cands_2, current_set, current_size)

    initial_candidates = list(range(N))
    solve(initial_candidates, [], 0)
    
    if best_size == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result_vectors = []
    for i in best_set:
        result_vectors.append(all_vectors[i][3])
        
    return np.array(result_vectors, dtype=np.int64)
