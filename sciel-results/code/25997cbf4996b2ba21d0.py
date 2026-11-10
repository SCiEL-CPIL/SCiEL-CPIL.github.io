# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 85.0
    n = 512
    
    # 1. Precompute conflict graph
    # conflicts[i] is a set of j such that i and j cannot be in the same set
    # This happens if there exists a 4-AP containing both i and j.
    # A 4-AP is {a, a+d, a+2d, a+3d}.
    # Any pair of elements in this set conflicts.
    
    conflicts = [set() for _ in range(n + 1)]
    
    # Generate all 4-APs
    for a in range(1, n + 1):
        # d such that a+3d <= n
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            terms = [a, a + d, a + 2 * d, a + 3 * d]
            # Add conflicts between all pairs in this AP
            # (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = terms[i], terms[j]
                    conflicts[u].add(v)
                    conflicts[v].add(u)
    
    # 2. Convert to adjacency matrix for the complement graph (compatibility graph)
    # We want to find a Maximum Independent Set in the conflict graph,
    # which is equivalent to a Maximum Clique in the complement graph.
    # Let G_comp be the graph where an edge exists between i and j if they are NOT in conflict.
    # We want Max Clique in G_comp.
    
    # Using numpy for fast operations
    # adj[i][j] is True if i and j are compatible (not in conflict)
    # Note: i and i are compatible (trivially), but clique vertices are distinct.
    
    adj = np.zeros((n + 1, n + 1), dtype=bool)
    np.fill_diagonal(adj, True)
    
    for i in range(1, n + 1):
        # Initialize all as True (compatible), then set False for conflicts
        adj[i, 1:n+1] = True
        for j in conflicts[i]:
            adj[i, j] = False
            adj[j, i] = False
            
    # 3. Greedy Coloring for Bound
    # We will implement a greedy coloring algorithm to bound the size of the maximum clique.
    # The bound is the number of colors used.
    
    def color_sort(P, adj, n):
        """
        P is a list of candidate nodes (1-indexed).
        Returns (P_sorted, colors) where P_sorted is ordered by color (descending),
        and colors[i] is the color of P_sorted[i].
        """
        if not P:
            return [], []
        
        # Unassigned nodes
        unassigned = P[:]
        node_colors = {}
        
        # We want to order nodes such that nodes with higher "potential" are colored first?
        # Standard Tomita colors in reverse order of the input list, or just processes in order.
        # To get a tight bound, we usually process nodes in the order they appear in P.
        # However, the coloring itself assigns colors 1, 2, ...
        
        # A simple greedy coloring:
        # For each node in P, assign the smallest available color not used by its neighbors in P.
        
        # To speed this up, we can just assign colors sequentially?
        # No, that's not a valid coloring.
        
        # Let's implement a standard greedy coloring.
        # Complexity O(|P|^2) which is fine for small P, but P can be large initially.
        # However, we only call this on the initial set and on subproblems.
        
        # Optimization: We can use a bitset for neighbors?
        # For now, simple list based.
        
        # We process nodes in the order given in P.
        # But wait, the standard algorithm for Max Clique (Tomita) colors the vertices in P
        # in the order they are processed.
        
        # Let's try a slightly smarter coloring:
        # Sort P by degree in the induced subgraph?
        # Or just process in the given order.
        
        # Let's stick to processing in the given order for simplicity and consistency with Tomita.
        
        colors_list = []
        
        # We need to know which colors are forbidden for a node.
        # Forbidden if a neighbor in P has that color.
        
        # Since P is small enough in deep recursion, this is fast.
        # For large P, it might be slow.
        # N=512. |P| <= 512. 512^2 = 262k ops. Fast in Python? Maybe.
        
        # Let's try to optimize the inner loop.
        
        # Actually, we can just assign color = 1 + number of neighbors already colored?
        # No.
        
        # Let's use a dictionary for node -> color.
        
        for node in unassigned:
            # Find forbidden colors
            # Neighbors of node in P
            # We can check adj[node, neighbor]
            
            # To make this faster, we can precompute the neighbors of node in P?
            # Or just iterate.
            
            forbidden = set()
            for neighbor in unassigned:
                if neighbor == node:
                    continue
                if adj[node, neighbor]:
                    if neighbor in node_colors:
                        forbidden.add(node_colors[neighbor])
                        
            # Find smallest available color
            c = 1
            while c in forbidden:
                c += 1
                
            node_colors[node] = c
            colors_list.append(c)
            
        # Create list of (node, color)
        colored_nodes = [(node, node_colors[node]) for node in unassigned]
        
        # Sort by color descending, then by node id?
        # Tomita sorts by color.
        colored_nodes.sort(key=lambda x: (-x[1], x[0]))
        
        P_sorted = [x[0] for x in colored_nodes]
        colors = [x[1] for x in colored_nodes]
        
        return P_sorted, colors, max(colors) if colors else 0

    best_clique = []
    best_size = 0
    
    # 4. Initial Greedy Clique to get a good lower bound
    # Greedy Max Clique heuristic
    
    # Order nodes by degree in G_comp descending
    deg = np.sum(adj, axis=1) - 1 # Subtract self
    # Sort 1..n by degree descending
    nodes = list(range(1, n + 1))
    nodes.sort(key=lambda x: -deg[x])
    
    # Greedy construction
    # We will pick nodes one by one.
    # A node is valid if it is connected to all previously picked nodes.
    
    current_clique = []
    P_curr = nodes[:]
    
    while P_curr:
        # Pick the first valid node
        valid_node = None
        for node in P_curr:
            if all(adj[node, c] for c in current_clique):
                valid_node = node
                break
        
        if valid_node is None:
            break
        
        current_clique.append(valid_node)
        
        # Remove invalid nodes from P_curr
        # A node is invalid if it is not connected to valid_node
        P_curr = [x for x in P_curr if x != valid_node and adj[x, valid_node]]
        
    if len(current_clique) > best_size:
        best_size = len(current_clique)
        best_clique = current_clique[:]
        
    # 5. Branch and Bound (Tomita Algorithm)
    
    # We need to be careful with recursion depth and time.
    # N=512. Depth can be 512.
    
    def expand(C, P, colors, max_color):
        nonlocal best_size, best_clique, deadline
        
        if time.time() > deadline:
            return False
            
        if not P:
            if len(C) > best_size:
                best_size = len(C)
                best_clique = C[:]
            return True
            
        # P is sorted by color descending.
        # colors[i] is the color of P[i].
        # The max clique size in P is at most max_color.
        # But we can prune if len(C) + colors[0] <= best_size.
        
        if len(C) + colors[0] <= best_size:
            return True
            
        # Iterate through P in order (highest color first)
        # Standard Tomita:
        # for i in range(len(P)):
        #   if len(C) + colors[i] <= best_size: return
        #   expand(C + [P[i]], P[:i] filtered by neighbors of P[i], ...)
        
        for i in range(len(P)):
            if time.time() > deadline:
                return False
                
            if len(C) + colors[i] <= best_size:
                return True # Since colors are sorted descending, all subsequent i' > i have colors[i'] <= colors[i]
            
            # Add P[i] to clique
            new_C = C + [P[i]]
            
            # New P is P[0..i-1] filtered by neighbors of P[i]
            # P[:i] contains nodes with index < i.
            # We need to keep only those connected to P[i].
            
            # Optimization: Use numpy for filtering?
            # P is a list. adj is numpy.
            
            new_P = []
            new_colors = []
            
            for j in range(i):
                if adj[P[i], P[j]]:
                    new_P.append(P[j])
                    new_colors.append(colors[j])
                    
            if new_P:
                # Recolor new_P
                # The colors of new_P must be recomputed relative to the subgraph?
                # Actually, the existing colors are an upper bound.
                # We can just use the existing colors?
                # Yes, the color of a node in a subgraph is <= color in the supergraph.
                # So colors[j] for j < i are valid upper bounds.
                # We just need to filter the colors list to match new_P.
                # We already did that in the loop above.
                
                # Note: new_P and new_colors are aligned.
                # We don't need to sort again if we preserve relative order?
                # The order in P was by color descending.
                # P[:i] is also by color descending.
                # Filtering preserves order.
                # So new_P is sorted by color descending.
                
                expand(new_C, new_P, new_colors, colors[0])
            else:
                # Leaf
                if len(new_C) > best_size:
                    best_size = len(new_C)
                    best_clique = new_C[:]
                    
    # Initial call
    # P is all nodes 1..n
    # Sort P by degree descending?
    # We already sorted nodes by degree descending.
    # Let's use that order for P0.
    
    P0 = nodes[:]
    
    # Color P0
    P0_sorted, colors0, max_color0 = color_sort(P0, adj, n)
    
    expand([], P0_sorted, colors0, max_color0)
    
    # 6. Local Search / Hill Climbing to improve the result
    # If B&B didn't find the optimum, we can try to improve the best_clique found.
    # Strategy: Try to add nodes to best_clique? No, it's maximal.
    # Try to swap nodes?
    # Or just run a simple greedy again with the best_clique as a starting point?
    
    # Actually, a common improvement is to use the result of B&B as a seed for a local search.
    # But given the time constraint, the B&B result is likely the best.
    
    # Let's try one more simple heuristic:
    # Randomized restarts of the greedy algorithm?
    # We have time left?
    
    if time.time() < deadline - 5.0:
        # Try a few randomized greedy runs
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
                
            # Randomize the order of nodes
            rand_nodes = nodes[:]
            np.random.shuffle(rand_nodes)
            
            rand_clique = []
            P_rand = rand_nodes[:]
            
            while P_rand:
                valid_node = None
                for node in P_rand:
                    if all(adj[node, c] for c in rand_clique):
                        valid_node = node
                        break
                
                if valid_node is None:
                    break
                
                rand_clique.append(valid_node)
                P_rand = [x for x in P_rand if x != valid_node and adj[x, valid_node]]
                
            if len(rand_clique) > best_size:
                best_size = len(rand_clique)
                best_clique = rand_clique[:]
                
    # Final result
    result = sorted(best_clique)
    
    # Sanity check: ensure non-empty
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
