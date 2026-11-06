"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 50.0  # safety margin before 60s limit (killed at 90s)
    
    n = 256
    max_sum = 2 * n  # 512
    
    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)
    
    # Filter out numbers where 2*a is a square (diagonal constraint)
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    
    # Build conflict graph (adjacency): a and b conflict if a+b is a perfect square
    # We need MIS on this conflict graph
    # Conflict[i] = set of indices j where valid_numbers[i] + valid_numbers[j] is a square
    conflict = [set() for _ in range(num_valid)]
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                conflict[i].add(val_to_idx[b])
    
    # Build adjacency matrix as bitmasks for the conflict graph
    # adj[i] = bitmask of nodes connected to i in conflict graph
    adj = [0] * num_valid
    for i in range(num_valid):
        mask = 0
        for j in conflict[i]:
            mask |= (1 << j)
        adj[i] = mask
    
    # Tomita's maximum clique algorithm on the COMPLEMENT of the conflict graph
    # MIS in G = MaxClique in complement(G)
    # Complement adjacency: nodes NOT in conflict[i] (excluding self)
    # comp_adj[i] = bitmask of nodes j where j != i and j not in conflict[i]
    
    # For Tomita's algorithm, we work with a candidate set P (bitmask)
    # and try to find a clique in the complement graph
    
    # Precompute complement adjacency
    all_mask = (1 << num_valid) - 1
    comp_adj = [0] * num_valid
    for i in range(num_valid):
        # complement: all nodes except self and those in conflict
        comp_adj[i] = all_mask & ~(adj[i] | (1 << i))
    
    # Tomita's maximum clique with greedy coloring
    # We use a recursive branch-and-bound
    
    best_clique_size = 0
    best_clique_nodes = []
    
    # Greedy coloring: assign colors to nodes in P such that adjacent nodes get different colors
    # The max color is an upper bound on clique size in P
    # For the complement graph, two nodes are "adjacent" if they are NOT in conflict
    # So in the complement graph, a clique = independent set in original = our answer
    
    def greedy_color(P):
        """
        P is a bitmask of candidate nodes.
        Returns (color_order, max_color) where color_order is a list of (node, color)
        ordered by color (for branching), and max_color is the upper bound.
        """
        # Greedy coloring: process nodes in some order, assign smallest available color
        # Two nodes can share a color if they are NOT adjacent in the complement graph
        # (i.e., they ARE in conflict in the original graph)
        
        # For efficient coloring, we use the approach from Tomita et al.:
        # Order nodes, then greedily assign colors
        
        P_list = []
        temp = P
        while temp:
            # Take lowest set bit
            lsb = temp & (-temp)
            node = lsb.bit_length() - 1
            P_list.append(node)
            temp ^= lsb
        
        # Greedy coloring
        color = [0] * num_valid  # color assignment for nodes in P
        # We need: nodes with same color must NOT be adjacent in complement
        # i.e., they MUST be in conflict in original graph
        # So: node u and v can share color iff (adj[u] has bit v set) or (adj[v] has bit u set)
        # i.e., they are connected in the conflict graph
        
        # Actually for clique finding in complement graph:
        # A clique in complement means all pairs are adjacent in complement
        # Adjacent in complement = NOT in conflict (and not self)
        # So for coloring upper bound: nodes with same color must NOT be adjacent in complement
        # = they MUST be in conflict in original
        # So u, v same color iff conflict edge exists between them
        
        # Greedy: assign colors
        colors_used = []  # list of lists of nodes per color
        node_color = {}
        
        for node in P_list:
            # Find smallest color where node is compatible (in conflict with all nodes of that color)
            placed = False
            for ci, color_nodes in enumerate(colors_used):
                # Check if node is in conflict with ALL nodes in this color class
                compatible = True
                for other in color_nodes:
                    # node and other must be in conflict (adjacent in original)
                    if not (adj[node] >> other) & 1:
                        compatible = False
                        break
                if compatible:
                    colors_used[ci].append(node)
                    node_color[node] = ci
                    placed = True
                    break
            if not placed:
                colors_used.append([node])
                node_color[node] = len(colors_used) - 1
        
        max_color = len(colors_used)
        
        # Order nodes by color (highest color first for branching)
        # Return list of (node, color) sorted by color descending
        ordered = []
        for ci in range(len(colors_used) - 1, -1, -1):
            for node in colors_used[ci]:
                ordered.append((node, ci + 1))
        
        return ordered, max_color
    
    def expand(P, current_clique, current_size):
        """
        Tomita's expand: P is bitmask of candidates, current_clique is list of nodes in current clique.
        """
        nonlocal best_clique_size, best_clique_nodes
        
        if time.time() > deadline:
            return
        
        if P == 0:
            if current_size > best_clique_size:
                best_clique_size = current_size
                best_clique_nodes = list(current_clique)
            return
        
        # Greedy coloring for upper bound
        ordered, max_color = greedy_color(P)
        
        # If current_size + max_color <= best_clique_size, prune
        if current_size + max_color <= best_clique_size:
            return
        
        # Process nodes in reverse color order (highest color first)
        # We iterate through ordered in reverse (which is color ascending in our construction)
        # Actually ordered is from high color to low color
        # We process from the end (low color) to the beginning (high color)
        # Wait, let me reconsider the ordering
        
        # In Tomita's algorithm, we process vertices in the order they appear in the coloring
        # from the last color to the first. Let me restructure.
        
        # Let's redo: ordered should be processed from index 0 to end
        # where index 0 has the highest color
        # We maintain a shrinking P
        
        # Process each node in order
        P_remaining = P
        for idx in range(len(ordered)):
            if time.time() > deadline:
                return
            
            node, col = ordered[idx]
            
            # If node is no longer in P_remaining, skip
            if not (P_remaining >> node) & 1:
                continue
            
            # Upper bound: current_size + col (since col is the color of this node,
            # and all remaining nodes have color <= col)
            if current_size + col <= best_clique_size:
                return
            
            # Branch: include node in clique
            # New candidates: intersection of P_remaining and neighbors of node in complement
            # Neighbors in complement = comp_adj[node]
            new_P = P_remaining & comp_adj[node]
            
            current_clique.append(node)
            expand(new_P, current_clique, current_size + 1)
            current_clique.pop()
            
            # Remove node from P_remaining
            P_remaining &= ~(1 << node)
            
            # If P_remaining is empty, done
            if P_remaining == 0:
                break
    
    # Initial candidate set: all valid nodes
    initial_P = all_mask
    expand(initial_P, [], 0)
    
    # Convert best clique back to valid numbers
    if best_clique_nodes:
        result = sorted([valid_numbers[i] for i in best_clique_nodes])
    else:
        # Fallback
        result = valid_numbers[:1]
    
    # Verify and return
    return result
# EVOLVE-BLOCK-END
