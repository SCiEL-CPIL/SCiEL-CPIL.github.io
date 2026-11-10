# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large square-difference-free set modulo 205.
    
    Strategy:
    1. Compute the conflict graph (edges connect pairs with forbidden difference).
    2. Use a Branch and Bound (B&B) algorithm to find the Maximum Independent Set (MIS).
       - Uses a greedy coloring bound (Tomita's algorithm style) for pruning.
       - Uses bitmasks for fast neighborhood calculations.
    3. If B&B completes or finds a good solution, use it.
    4. Fallback/Refinement: Local search if B&B doesn't finish in time or to polish.
    """
    m = 205
    
    # Step 1: Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = pow(x, 2, m)
        if val != 0:
            forbidden.add(val)
    
    # Build adjacency as bitmasks for fast operations
    # adj_mask[i] is an integer where bit j is set if i and j are adjacent
    adj_mask = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j = (i - d) % m
            if i != j:
                mask |= (1 << j)
        adj_mask[i] = mask

    # Precompute non-neighbors for potential use? Or just use complement of adj_mask within the candidate set.
    
    # Greedy coloring for upper bound
    # colors[i] = color assigned to i
    # color_class[c] = bitmask of nodes with color c
    def greedy_color(subset_mask, adj_mask):
        """
        Greedy coloring of the subgraph induced by subset_mask.
        Returns number of colors used (which is a lower bound on clique size, upper bound on independent set size in complement? 
        Wait. We are finding Independent Set in G.
        In G, an independent set is a set of non-adjacent nodes.
        The bound used in B&B for MIS is usually the clique number of the *complement* graph?
        No, standard MIS B&B (Tomita) works on the graph G.
        It uses the fact that if we color the *neighborhood* or the *remaining candidates*...
        
        Let's clarify the bound.
        We want max independent set in G.
        This is equivalent to max clique in G_complement.
        
        However, the standard "Tomita" algorithm for Maximum Clique uses coloring on the candidate set.
        If we are finding MIS in G, we can find Max Clique in G_comp.
        Let's stick to MIS in G directly or convert?
        Converting to G_comp: adj_comp[i] = ~adj_mask[i] & full_mask.
        
        Let's use the Max Clique formulation on the complement graph, as the coloring bound is standard for Clique.
        """
        pass # We will implement a specific bound for MIS directly or use clique logic.
        
    # Let's implement the standard Maximum Clique algorithm on the complement graph.
    # G_comp edge (i,j) exists if i,j are NOT adjacent in G (i.e., difference is NOT in forbidden).
    # We want Max Clique in G_comp.
    
    full_mask = (1 << m) - 1
    adj_comp = [0] * m
    for i in range(m):
        # Nodes not adjacent to i in G (excluding i itself)
        # adj_mask[i] has neighbors in G.
        # We want non-neighbors in G.
        # non_neighbors = full_mask ^ adj_mask[i] ^ (1 << i)
        adj_comp[i] = full_mask ^ adj_mask[i] ^ (1 << i)

    # Greedy coloring for Max Clique bound
    # We color the candidate set. The number of colors is an upper bound on the clique size.
    def greedy_color_bound(candidates_mask):
        """
        Greedy coloring of the subgraph induced by candidates_mask in G_comp.
        Returns the number of colors used.
        """
        if candidates_mask == 0:
            return 0
        
        # We will assign colors greedily.
        # colors list: index is node, value is color.
        # We process nodes in some order (e.g., descending degree or just index).
        # For the bound to be tight, the order matters. Tomita uses a specific ordering.
        # Simple greedy: iterate nodes in candidates_mask, assign smallest available color.
        
        # To make it faster, we can use a bitset approach or just simple lists.
        # m=205 is small enough for O(N^2) coloring if done carefully, but we call it many times.
        # We need a fast coloring.
        
        # Let's use a simple greedy:
        # 1. Get list of nodes in candidates_mask.
        # 2. Sort them by degree in the induced subgraph? (Expensive)
        # 3. Just iterate 0..m-1.
        
        # Optimization: We only need the count, not the assignment.
        # We can use a "color class" approach.
        # color_classes is a list of bitmasks.
        # For each node u in candidates:
        #   Find the first color c such that u is not adjacent to any node in color_classes[c].
        #   i.e., adj_comp[u] & color_classes[c] == 0.
        #   If found, add u to color_classes[c].
        #   Else, create new color class.
        
        color_classes = []
        # To speed up, we should process nodes in an order that makes coloring easier?
        # Usually, high degree nodes first? Or low degree?
        # For clique bound, we want few colors.
        # Let's just iterate through the bits.
        
        # Extract nodes
        # Instead of extracting list, we can iterate through set bits.
        # But checking "is u in color_classes[c]" requires bit operations.
        
        # Let's try a slightly different approach:
        # We maintain an array 'colors' for the nodes in candidates.
        # But mapping node->index in candidates is tricky.
        
        # Let's stick to the bitmask color classes.
        # Iterate u from 0 to m-1.
        # If u is in candidates_mask:
        #    for c in range(len(color_classes)):
        #        if (adj_comp[u] & color_classes[c]) == 0:
        #            color_classes[c] |= (1 << u)
        #            break
        #    else:
        #        color_classes.append(1 << u)
        
        # This is O(N * K) where K is number of colors. K <= N. So O(N^2).
        # N=205. 205^2 = 42k ops.
        # If B&B nodes are ~100k, 4e9 ops. Too slow in Python.
        
        # We need a faster bound or fewer B&B nodes.
        # Maybe use a simpler bound? Like |candidates|? That's weak.
        # Maybe use a precomputed bound?
        
        # Alternative: Use the fact that the graph is dense/sparse?
        # Let's check density.
        # Forbidden squares: 102 values.
        # Each node has ~102 neighbors in G.
        # So in G_comp, each node has ~205 - 102 - 1 = 102 non-neighbors.
        # It's roughly 50% dense.
        
        # Let's try to optimize the coloring.
        # We can order the nodes by their degree in the induced subgraph?
        # Or just use a simple heuristic.
        
        # Actually, for m=205, a pure Python B&B with O(N^2) bound might be too slow if the tree is large.
        # Let's try a different approach:
        # 1. Get a good initial solution using Randomized Greedy + Local Search (fast).
        # 2. Use B&B to prove optimality or improve, but with a time limit.
        # 3. If B&B is too slow, return the best found.
        
        # Let's implement a faster bound:
        # Use a precomputed "color order" for the whole graph? No, induced subgraph changes.
        
        # Let's try to reduce the constant factor.
        # Use numpy? No, bitmasks are better for small N.
        
        # Let's try a different B&B strategy:
        # Branch on a node.
        # Include it: recurse on neighbors in G_comp (i.e., non-neighbors in G).
        # Exclude it: recurse on candidates without it.
        
        # The bound is the key.
        # Let's try a simpler bound first: just the size of the candidate set.
        # This will explore more nodes but each node is fast.
        # If the tree is not too big, it might work.
        
        # Or, use a "1-extendable" check?
        
        # Let's go with the bitmask coloring but optimize the loop.
        # We can skip nodes that are already colored? No, we color the whole candidate set.
        
        # Let's try to implement the coloring with lists of nodes instead of bitmasks for color classes?
        # color_classes is a list of lists of nodes.
        # Check if u is adjacent to any node in color_classes[c].
        # This requires checking adj_comp[u] & (1<<v) for each v in class.
        # Same complexity.
        
        # Let's try to limit the B&B depth or use a time limit.
        
        pass

    # Let's refine the plan:
    # 1. Fast Heuristic: Random Greedy + Local Search to get a good lower bound (LB).
    # 2. B&B with Greedy Coloring Bound to try to find the exact MIS or improve LB.
    # 3. Time limit guard.

    def get_initial_solution():
        best_set = []
        best_size = 0
        
        # Multiple random starts
        for _ in range(50):
            order = list(range(m))
            random.shuffle(order)
            current = []
            current_mask = 0
            for candidate in order:
                # Check if candidate is compatible with current set in G
                # i.e., no edges in G between candidate and current nodes
                # i.e., adj_mask[candidate] & current_mask == 0
                if (adj_mask[candidate] & current_mask) == 0:
                    current.append(candidate)
                    current_mask |= (1 << candidate)
            
            if len(current) > best_size:
                best_size = len(current)
                best_set = list(current)
            
            # Local search: try to swap nodes to improve
            # Simple 1-swap local search
            improved = True
            while improved:
                improved = False
                current_set = set(current)
                current_mask = 0
                for v in current:
                    current_mask |= (1 << v)
                
                # Try to remove u and add v
                for u in list(current):
                    mask_without_u = current_mask ^ (1 << u)
                    # Find v not in current such that adj_mask[v] & mask_without_u == 0
                    # And v != u
                    for v in range(m):
                        if v in current_set:
                            continue
                        if (adj_mask[v] & mask_without_u) == 0:
                            # Swap u with v
                            current.remove(u)
                            current.append(v)
                            current_mask ^= (1 << u)
                            current_mask |= (1 << v)
                            current_set.remove(u)
                            current_set.add(v)
                            improved = True
                            break
                    if improved:
                        break
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = list(current)
        
        return best_set, best_size

    # Run initial heuristic
    best_set, best_size = get_initial_solution()
    
    # B&B to improve
    # State: candidates_mask, current_clique_size
    # We are finding Max Clique in G_comp.
    # Initial candidates: all nodes.
    # Initial clique: empty.
    
    # To make B&B faster, we can order the initial candidates.
    # Sort by degree in G_comp (descending)?
    # Or just use 0..m-1.
    
    # Let's use a recursive B&B with a time limit.
    
    deadline = time.time() + 550.0
    global_best_size = best_size
    global_best_set = list(best_set)
    
    # Convert best_set to a mask for initialization?
    # No, B&B starts from scratch but uses the bound to prune.
    # We can also use the best_set to guide the search (branch on nodes in best_set first?).
    
    # Greedy Coloring Bound Function
    def greedy_color_bound(candidates_mask):
        """
        Returns an upper bound on the clique size in the subgraph induced by candidates_mask.
        """
        if candidates_mask == 0:
            return 0
        
        # We use a simple greedy coloring.
        # To make it faster, we can stop early if the number of colors exceeds the current best?
        # No, we need the exact count of colors for the bound.
        
        # Optimization: Use a precomputed order?
        # Let's just do the simple bitmask coloring.
        
        color_classes = []
        # Iterate through nodes in candidates_mask
        # We can iterate from 0 to m-1
        
        # To speed up, we can check if the number of remaining nodes + current_colors < global_best_size?
        # No, the bound is for the *subproblem*.
        
        # Let's try to optimize the inner loop.
        # Instead of checking all color classes, we can maintain a "available colors" mask?
        # No, colors are not simple bits.
        
        # Let's just do it.
        for u in range(m):
            if not (candidates_mask & (1 << u)):
                continue
            
            # Find a color for u
            placed = False
            for c in range(len(color_classes)):
                if (adj_comp[u] & color_classes[c]) == 0:
                    color_classes[c] |= (1 << u)
                    placed = True
                    break
            
            if not placed:
                color_classes.append(1 << u)
                
            # Early termination: if len(color_classes) + (number of uncolored nodes) <= global_best_size?
            # No, that's not right.
            # The bound is len(color_classes) at the end.
            # But if len(color_classes) is already >= global_best_size + 1, we can't prune based on < global_best_size.
            # We prune if bound <= global_best_size.
            # So if we have already used K colors, and K > global_best_size, we can stop?
            # No, we need the final count.
            # But if the final count will be > global_best_size, we don't prune.
            # We only prune if final count <= global_best_size.
            # So we must compute the full count.
            
        return len(color_classes)

    # B&B Function
    # We use a stack or recursion. Recursion is simpler but might hit depth limits? m=205, depth <= 205. Safe.
    # But Python recursion overhead is high. Let's use an explicit stack.
    
    # Stack items: (candidates_mask, current_clique_mask)
    # Actually, we just need candidates_mask and the size of the current clique.
    # But we need to reconstruct the set? No, just the size for now.
    # We want to update global_best_set.
    # So we need to track the current clique.
    # Let's store (candidates_mask, current_clique_list)
    
    # To reduce memory, we can store the clique as a bitmask too.
    # But updating a bitmask in a stack is easy.
    
    # Let's try a different approach:
    # Since m is small, we can use a recursive function with nonlocal variables.
    
    def branch_and_bound(candidates_mask, current_clique_mask):
        nonlocal global_best_size, global_best_set, deadline
        
        if time.time() > deadline:
            return False # Timeout
        
        if candidates_mask == 0:
            if bin(current_clique_mask).count('1') > global_best_size:
                global_best_size = bin(current_clique_mask).count('1')
                # Reconstruct set
                s = []
                for i in range(m):
                    if current_clique_mask & (1 << i):
                        s.append(i)
                global_best_set = s
            return True
        
        # Bound
        # If the bound is <= global_best_size, prune
        # Bound is greedy_color_bound(candidates_mask)
        # But we also have the current clique size.
        # Total size = current_size + bound
        # If current_size + bound <= global_best_size, prune.
        
        current_size = bin(current_clique_mask).count('1')
        
        # Quick bound: if current_size + popcount(candidates_mask) <= global_best_size, prune
        # popcount(candidates_mask) is a weak bound but fast.
        if current_size + bin(candidates_mask).count('1') <= global_best_size:
            return True
        
        # Stronger bound: greedy coloring
        # This is expensive. Let's do it.
        bound = greedy_color_bound(candidates_mask)
        if current_size + bound <= global_best_size:
            return True
        
        # Branch
        # Select a node to branch on.
        # Heuristic: pick a node with high degree in the induced subgraph?
        # Or just pick the lowest index?
        # Let's pick the node with the highest degree in the induced subgraph to reduce branching factor?
        # Or lowest?
        # Tomita suggests ordering by degree.
        
        # Let's just pick the lowest index for simplicity.
        # Or, to improve, pick a node that is in the best known solution?
        
        # Let's pick the lowest index u in candidates_mask.
        u_bit = candidates_mask & -candidates_mask
        u = u_bit.bit_length() - 1
        
        # Option 1: Include u
        # New candidates: neighbors of u in G_comp that are in candidates_mask and have index > u?
        # To avoid duplicates, we usually restrict to nodes with index > u.
        # But our candidates_mask might not be ordered.
        # Let's enforce an ordering: we only consider nodes with index > u.
        # This requires us to maintain the candidate set as a subset of nodes with index > last_branched_node.
        
        # This is a standard technique.
        # Let's modify the state to include 'last_index'.
        # Or, simpler: when we include u, the new candidates are (candidates_mask & adj_comp[u] & (~((1 << (u+1)) - 1)))?
        # i.e., neighbors of u that are greater than u.
        
        # When we exclude u, the new candidates are (candidates_mask & ~(1 << u)).
        
        # Let's try this.
        
        # Include u
        # new_candidates = candidates_mask & adj_comp[u] & (~((1 << (u + 1)) - 1))
        # Wait, adj_comp[u] includes all non-neighbors in G.
        # We want nodes v > u such that (u,v) is an edge in G_comp.
        # i.e., v is in candidates_mask, v > u, and v is not adjacent to u in G.
        
        mask_greater_u = full_mask ^ ((1 << (u + 1)) - 1)
        new_candidates_inc = candidates_mask & adj_comp[u] & mask_greater_u
        
        # Recurse
        branch_and_bound(new_candidates_inc, current_clique_mask | (1 << u))
        
        # Check time
        if time.time() > deadline:
            return False
        
        # Exclude u
        new_candidates_exc = candidates_mask & ~(1 << u)
        
        branch_and_bound(new_candidates_exc, current_clique_mask)
        
        return True

    # Start B&B
    # Initial candidates: all nodes.
    # To use the ordering trick, we should start with all nodes.
    # But the "greater than u" trick requires that we only branch on nodes in increasing order?
    # No, the trick is: when we include u, we only consider neighbors v > u.
    # This ensures each clique is generated exactly once.
    
    # However, the greedy coloring bound on `candidates_mask` assumes the graph induced by candidates_mask.
    # If we restrict to v > u, the induced subgraph is correct.
    
    # Let's run it.
    # But wait, the initial call should be with all nodes.
    # The first branch will pick the lowest node u=0.
    # Include 0: candidates become neighbors of 0 with index > 0.
    # Exclude 0: candidates become all nodes except 0.
    
    # This should work.
    
    # One issue: `greedy_color_bound` iterates 0..m-1.
    # If candidates_mask only has high indices, it still iterates 0..m-1 and skips.
    # This is fine.
    
    # Let's add a time check in the main loop of B&B?
    # The recursive function checks at the start.
    
    # Let's run B&B.
    # But if the tree is huge, it might take too long.
    # The time limit is 550s.
    # The initial heuristic takes some time.
    # Let's set the deadline to 580s to be safe.
    
    start_bnb = time.time()
    # We might want to sort the initial candidates?
    # The algorithm picks the lowest index.
    # We can permute the nodes to improve performance?
    # E.g., sort by degree in G_comp descending.
    # If we permute, we need to remap adj_comp.
    # Let's try that.
    
    # Compute degrees in G_comp
    degrees = [bin(adj_comp[i]).count('1') for i in range(m)]
    # Sort indices by degree descending
    sorted_indices = sorted(range(m), key=lambda i: degrees[i], reverse=True)
    
    # Create a mapping: new_index -> old_index
    # And remap adj_comp
    # new_adj_comp[new_i] has bits set for new_j if old_i and old_j are connected in G_comp.
    
    # This adds complexity. Let's try without permutation first.
    # If it's too slow, we can add it.
    
    # Actually, let's just run the B&B on the original graph.
    
    # To prevent stack overflow, Python's default limit is 1000. Depth 205 is fine.
    
    # Let's run it.
    # But we need to be careful: if B&B doesn't finish, we return the best found so far.
    # The recursive function returns False on timeout.
    # We can check the return value.
    
    # Let's wrap it in a try-except just in case.
    
    try:
        branch_and_bound(full_mask, 0)
    except RecursionError:
        pass
    except Exception:
        pass

    # If B&B found a better solution, use it.
    # Otherwise, use the heuristic solution.
    
    if global_best_set:
        return np.array(global_best_set, dtype=np.int64)
    
    # Fallback
    return np.array([0], dtype=np.int64)

# EVOLVE-BLOCK-END
