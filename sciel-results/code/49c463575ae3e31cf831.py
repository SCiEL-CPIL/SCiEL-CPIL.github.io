import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.monotonic() + 620.0  # Safety margin for 630s limit

    # 1. Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        f = (x * x) % m
        if f != 0:
            forbidden.add(f)
    forbidden.discard(0)

    # 2. Precompute conflict masks
    # conflict_mask[i] has bit j set if i and j conflict
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Also store the "non-conflict" mask (including self) for easier intersection
    # non_conflict_mask[i] has bit j set if i and j do NOT conflict (and i!=j)
    # Actually, for MIS, we need the set of vertices that can be added to a set S.
    # If we pick v, the next candidates must be in (Candidates \ {v}) \ Conflicts(v)
    # Let's stick to conflict_mask.

    def popcount(x):
        return x.bit_count()

    # 3. Branch and Bound for Maximum Independent Set
    # We represent the set of available candidates as a bitmask.
    # We want to find the largest independent set in the graph defined by conflict_mask.
    
    best_size = 0
    best_mask = 0

    # Greedy coloring for upper bound
    # This is a simplified bound: we greedily color the candidate graph.
    # The number of colors used is an upper bound on the size of the independent set?
    # No, the chromatic number is a lower bound on the clique number, but for Independent Set,
    # the size of the maximum independent set is at most the number of colors in a valid coloring?
    # Wait. If we color the graph with k colors, the largest independent set can be as large as the largest color class.
    # But actually, a simpler bound is:
    # In the subgraph induced by candidates, if we greedily pick vertices and remove their neighbors,
    # the number of vertices we pick is a lower bound on the independence number? No, it's a constructive heuristic.
    # The standard upper bound for MIS in branch and bound is often derived from coloring.
    # If the graph can be colored with c colors, then the maximum independent set size is <= c? No.
    # The maximum independent set size is <= the number of vertices, obviously.
    # A better bound: If we partition the candidates into cliques, the MIS size is at most the number of cliques? No.
    # Let's use the "greedy coloring" bound where we color the *complement* graph?
    # Actually, the standard technique is:
    # Upper bound = size of maximum independent set in the candidate set.
    # A common heuristic upper bound is the result of a greedy algorithm that selects vertices and removes neighbors,
    # but that gives a *lower* bound (a valid solution).
    
    # Let's use a different bound:
    # If we have a set of candidates C, we can compute an upper bound by greedily coloring the graph induced by C.
    # Wait, if we color G with k colors, the size of the maximum independent set is NOT bounded by k.
    # Example: Path of 3 vertices. Colors=2. Max Ind Set=2.
    # Example: Empty graph. Colors=1. Max Ind Set=N.
    # So coloring G gives a lower bound on the number of colors, which is a lower bound on the clique cover?
    
    # Correct Bound:
    # The size of the maximum independent set in G is equal to the size of the maximum clique in G_complement.
    # So we can find an upper bound on the Max Clique in G_complement.
    # Or, more simply, use the "greedy independent set" size as a *lower* bound for pruning? No, that's for finding a good solution.
    # For pruning, we need an *upper* bound.
    
    # A simple upper bound:
    # If we sort vertices by degree, we can use a bound based on the fact that each vertex in the independent set
    # "consumes" itself and its neighbors.
    # A standard bound is:
    # UB = sum(1 / (1 + deg(v))) ? No.
    
    # Let's use the "Coloring" bound on the *Candidate Graph* (which is the conflict graph).
    # If we color the conflict graph with k colors, it doesn't directly bound the independent set size from above.
    # However, if we color the *Non-Conflict* graph (the graph where edges mean "can be together"),
    # then the graph is a union of cliques? No.
    
    # Let's stick to a simpler, effective strategy for N=205:
    # 1. Use a greedy algorithm to find a good initial lower bound (best_size).
    # 2. Use Branch and Bound with a simple upper bound:
    #    UB = current_size + (number of candidates remaining)
    #    This is too weak.
    
    # Better Bound:
    # Compute the maximum degree in the candidate subgraph.
    # Actually, let's use the "Greedy Coloring" of the *Conflict Graph* to get a lower bound on the number of colors?
    # No.
    
    # Let's try this bound:
    # For the remaining candidates, compute a greedy independent set size. This is a lower bound.
    # We need an upper bound.
    # Upper Bound = Current Size + Greedy Coloring of the *Complement* of the Conflict Graph?
    # If the Complement Graph (where edges = compatible) is colored with k colors,
    # then the vertices can be partitioned into k independent sets in the Conflict Graph?
    # No, an independent set in the Conflict Graph is a Clique in the Complement Graph.
    # So Max Independent Set in Conflict Graph = Max Clique in Complement Graph.
    # The size of the Max Clique is at most the number of colors in a coloring of the Complement Graph?
    # No, the size of the Max Clique is at most the number of colors in a coloring of the *Conflict* Graph? No.
    # The size of the Max Clique in G is at most the number of colors in a coloring of G? No.
    # The size of the Max Clique in G is at most the number of colors in a coloring of G_complement?
    # Yes! If G_complement is colored with k colors, then G is a union of k independent sets? No.
    # If G_complement is k-colorable, then G_complement has no clique of size k+1.
    # So G has no independent set of size k+1.
    # Therefore, Max Independent Set in G <= Chromatic Number of G_complement.
    
    # So:
    # 1. Build Complement Conflict Graph (edges where NOT conflicting).
    # 2. In the branch and bound, for the remaining candidates, compute a greedy coloring of the Complement Conflict Graph.
    # 3. The number of colors is an upper bound on the additional elements we can add.
    
    # Let's implement this.
    
    # Precompute complement masks
    # comp_mask[i] has bit j set if i and j are NOT in conflict (and i != j)
    comp_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j: continue
            if not (conflict_mask[i] >> j) & 1:
                mask |= (1 << j)
        comp_mask[i] = mask

    def greedy_color_bound(candidates_mask):
        """
        Returns an upper bound on the size of the maximum independent set in the subgraph induced by candidates_mask.
        This is done by greedily coloring the *Complement* of the conflict graph (i.e., the compatibility graph) restricted to candidates.
        Wait, if we color the compatibility graph, the number of colors is an upper bound on the size of the maximum clique in the compatibility graph,
        which is the size of the maximum independent set in the conflict graph.
        
        Greedy Coloring Algorithm for Upper Bound:
        We want to color the graph H (Compatibility Graph) induced by candidates.
        We process vertices in some order. We assign the smallest available color.
        The total number of colors used is an upper bound on the clique number of H?
        No, the chromatic number is a lower bound on the clique number.
        So Chromatic Number(H) >= Clique Number(H).
        This means the number of colors is a LOWER bound on the Max Clique size?
        No, Chromatic Number >= Clique Number.
        So if we use k colors, the clique number could be anything <= k? No, it must be <= k? No.
        If a graph is 3-colorable, it can have a clique of size 3, or 2, or 1.
        It cannot have a clique of size 4.
        So Max Clique <= Chromatic Number.
        So if we find a coloring with k colors, Max Clique <= k.
        So Max Independent Set in Conflict Graph <= k.
        
        So we need to compute a valid coloring of the Compatibility Graph induced by candidates.
        The number of colors used is the upper bound.
        """
        if candidates_mask == 0:
            return 0
        
        colors = {}
        num_colors = 0
        remaining = candidates_mask
        
        # Order vertices by degree in the Compatibility Graph?
        # Or just arbitrary. To get a tight bound, we should try to minimize colors.
        # A simple greedy approach:
        # While remaining:
        #   Pick a vertex v.
        #   Check which colors are used by its neighbors in the Compatibility Graph.
        #   Assign the smallest available color.
        #   Remove v from remaining.
        
        # To make it fast, we can just iterate through set bits.
        # However, doing this at every node of the B&B tree might be slow.
        # For N=205, it's probably okay if the tree is pruned well.
        
        # Optimization: We can just return popcount(candidates_mask) if we don't want to spend time?
        # That's too weak.
        
        # Let's implement a fast greedy coloring.
        # We maintain a list of colors for each vertex? No, we just need the count.
        
        # Actually, a simpler bound is often used:
        # UB = sum_{v in candidates} 1 / (1 + deg_comp(v)) ? No.
        
        # Let's stick to the coloring bound but optimize it.
        # We'll pick vertices in order of decreasing degree in the Compatibility Graph to try to reuse colors.
        
        # Precompute degrees in Compatibility Graph for all vertices?
        # deg_comp[i] = popcount(comp_mask[i])
        
        # In the subgraph, the degree might change.
        # Let's just do a simple greedy coloring on the induced subgraph.
        
        # To speed up, we can limit the depth or use a heuristic.
        # Given N=205, let's try it.
        
        # List of vertices in candidates
        verts = []
        temp = candidates_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp ^= lsb
            
        # Sort by degree in the induced compatibility graph?
        # This requires computing degrees.
        # For simplicity, let's just iterate.
        
        color_assign = {}
        max_color = 0
        
        # We need to know neighbors in the Compatibility Graph.
        # Neighbors of v in H are those u in candidates such that (comp_mask[v] >> u) & 1 is 1.
        
        # Greedy:
        for v in verts:
            # Find colors used by neighbors
            used_colors = set()
            neighbors_mask = comp_mask[v] & candidates_mask
            # Iterate neighbors
            temp_n = neighbors_mask
            while temp_n:
                lsb_n = temp_n & -temp_n
                idx_n = lsb_n.bit_length() - 1
                if idx_n in color_assign:
                    used_colors.add(color_assign[idx_n])
                temp_n ^= lsb_n
            
            # Assign smallest available color
            c = 0
            while c in used_colors:
                c += 1
            
            color_assign[v] = c
            if c > max_color:
                max_color = c
            
        return max_color + 1

    # 4. Main B&B Loop
    # State: (current_mask, candidates_mask)
    # current_mask: vertices already selected
    # candidates_mask: vertices that can still be selected
    
    # Initial candidates: all vertices
    initial_candidates = (1 << m) - 1
    
    # Get a good initial lower bound using greedy
    # Greedy: Pick vertex with highest degree in Conflict Graph? Or lowest?
    # For MIS, picking high degree vertices in Conflict Graph (low degree in Comp Graph) might be good?
    # Let's just use a random greedy or sorted by index.
    
    # Greedy Lower Bound
    def greedy_mis(candidates_mask):
        selected = 0
        rem = candidates_mask
        while rem:
            # Pick a vertex, e.g., the one with the smallest index
            lsb = rem & -rem
            v = lsb.bit_length() - 1
            selected |= lsb
            # Remove v and its conflicts from rem
            rem &= ~conflict_mask[v]
            rem &= ~lsb
        return selected, popcount(selected)

    init_mask, init_size = greedy_mis(initial_candidates)
    best_mask = init_mask
    best_size = init_size
    
    # Stack for DFS
    # Each item: (current_mask, candidates_mask, current_size)
    stack = [(0, initial_candidates, 0)]
    
    while stack:
        if time.monotonic() > deadline:
            break
            
        cur_mask, cands, cur_size = stack.pop()
        
        # Prune: if even adding all candidates doesn't beat best
        if cur_size + popcount(cands) <= best_size:
            continue
            
        # Prune: Upper Bound via Coloring
        # Only compute if it's close to the limit?
        # For N=205, computing the bound might be expensive.
        # Let's compute it only if cur_size + popcount(cands) > best_size
        # And maybe limit the frequency or use a cheaper bound first.
        
        # Cheaper bound: 
        # If we just use popcount, it's weak.
        # Let's use the coloring bound.
        
        ub = greedy_color_bound(cands)
        if cur_size + ub <= best_size:
            continue
            
        if cands == 0:
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask
            continue
            
        # Branch
        # Pick a vertex to branch on.
        # Heuristic: Pick vertex with highest degree in the Conflict Graph restricted to cands?
        # Or just the first one.
        
        # Let's pick the vertex with the highest degree in the *Conflict* graph within cands.
        # This tends to prune more.
        
        # Find best vertex
        best_v = -1
        max_deg = -1
        temp_c = cands
        while temp_c:
            lsb = temp_c & -temp_c
            v = lsb.bit_length() - 1
            # Degree in conflict graph within cands
            deg = popcount(conflict_mask[v] & cands)
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp_c ^= lsb
            
        if best_v == -1:
            continue
            
        v_mask = 1 << best_v
        
        # Branch 1: Include best_v
        # New candidates: cands without best_v and without its conflicts
        new_cands_include = cands & ~v_mask & ~conflict_mask[best_v]
        new_mask_include = cur_mask | v_mask
        new_size_include = cur_size + 1
        
        # Branch 2: Exclude best_v
        # New candidates: cands without best_v
        new_cands_exclude = cands & ~v_mask
        new_mask_exclude = cur_mask
        new_size_exclude = cur_size
        
        # Push to stack. 
        # To find larger sets faster, we might want to explore Include first?
        # Yes, usually exploring the branch that adds a vertex first helps find a good solution quickly.
        # But LIFO stack means the last pushed is processed first.
        # So push Exclude first, then Include.
        
        stack.append((new_mask_exclude, new_cands_exclude, new_size_exclude))
        stack.append((new_mask_include, new_cands_include, new_size_include))
        
        # Optional: Update best if we found a complete solution?
        # No, we update when cands becomes 0.

    # If B&B didn't find anything better (unlikely), fall back to init
    if best_mask == 0:
        best_mask = init_mask
        best_size = init_size

    result_list = [i for i in range(m) if (best_mask >> i) & 1]
    result = np.array(sorted(result_list), dtype=np.int64)
    return result
