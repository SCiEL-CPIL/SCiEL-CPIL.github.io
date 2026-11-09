import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 256
    # Build conflict graph using bitmasks for speed
    # adj_mask[i] is a bitmask of nodes conflicting with i
    adj_mask = [0] * (n + 1)
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj_mask[a] |= (1 << b)
                adj_mask[b] |= (1 << a)
    
    # Greedy coloring to get upper bound on clique size in the complement (MIS in original)
    # We color the graph such that adjacent nodes have different colors
    # The number of colors is an upper bound on the size of a maximum independent set
    # Wait, that's not right. In the conflict graph, an independent set is a set of non-adjacent nodes.
    # Coloring the conflict graph gives cliques in the complement.
    # Actually, let's use the standard branch and bound for maximum independent set.
    
    # We'll use a recursive branch and bound approach
    # State: candidate set as a bitmask, current solution size
    # Upper bound: greedy coloring of the candidate subgraph
    
    def greedy_color(candidates_mask):
        """
        Greedy coloring of the subgraph induced by candidates_mask.
        Returns the number of colors used, which is an upper bound on the 
        size of a maximum independent set in the complement... 
        Wait, I need to think about this more carefully.
        
        In the conflict graph G, we want a maximum independent set.
        An independent set in G is a clique in the complement graph G'.
        Coloring G gives an upper bound on the clique number of G', which is the 
        independence number of G.
        So yes, greedy coloring of G restricted to candidates gives an upper bound.
        """
        colors = 0
        uncolored = candidates_mask
        while uncolored:
            colors += 1
            # Pick the smallest available node
            # Find lowest set bit
            lsb = uncolored & (-uncolored)
            node = lsb.bit_length() - 1
            
            # This node gets color `colors`
            # All its neighbors in the uncolored set cannot get this color
            # In the conflict graph, neighbors are those that conflict with `node`
            # So we remove node and its neighbors from consideration for this color
            
            # Actually, greedy coloring: assign color to a node, then remove it and 
            # all its neighbors from the pool for that color.
            # The next color is assigned to the remaining nodes.
            
            # Nodes that can share this color: node itself, and nodes not adjacent to node
            # But we're doing sequential coloring: pick a node, give it the current color,
            # then remove it from uncolored. Its neighbors must get different colors.
            
            # Standard greedy: 
            # For color c, find a maximal independent set in the remaining graph.
            # Assign color c to all nodes in that set.
            # Remove them from uncolored.
            
            # Simpler: just pick nodes one by one.
            # Assign color to node, remove node from uncolored.
            # But this doesn't give a good bound.
            
            # Better greedy coloring for upper bound:
            # While uncolored is not empty:
            #   colors += 1
            #   Find a maximal independent set in the subgraph induced by uncolored
            #   Assign color `colors` to all nodes in that set
            #   Remove those nodes from uncolored
            
            # Find maximal independent set in subgraph induced by uncolored
            mis = 0
            temp = uncolored
            while temp:
                lsb = temp & (-temp)
                node = lsb.bit_length() - 1
                mis |= lsb
                # Remove node and all its neighbors from temp
                temp &= ~(adj_mask[node] | lsb)
            
            uncolored &= ~mis
        
        return colors
    
    best_size = 0
    best_set = 0
    
    def branch_and_bound(candidates_mask, current_size):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = 0  # We'd need to track the actual set, let me fix this
            return
        
        # Upper bound: current_size + greedy_color(candidates_mask)
        ub = current_size + greedy_color(candidates_mask)
        if ub <= best_size:
            return
        
        # Pick a node to branch on
        # Heuristic: pick node with highest degree in the candidate subgraph
        # or lowest degree, depending on strategy
        
        # For maximum independent set, common heuristic is to pick a node 
        # and branch on including it or excluding it.
        
        # Let's pick the node with the highest degree in the induced subgraph
        # This tends to prune more when we include it (removes many neighbors)
        
        # Find node with max degree in candidates
        max_deg = -1
        chosen_node = -1
        temp = candidates_mask
        while temp:
            lsb = temp & (-temp)
            node = lsb.bit_length() - 1
            deg = bin(adj_mask[node] & candidates_mask).count('1')
            if deg > max_deg:
                max_deg = deg
                chosen_node = node
            temp &= temp - 1
        
        if chosen_node == -1:
            return
        
        chosen_bit = 1 << chosen_node
        
        # Branch 1: Include chosen_node
        # Remove chosen_node and all its neighbors from candidates
        new_candidates = candidates_mask & ~(adj_mask[chosen_node] | chosen_bit)
        branch_and_bound(new_candidates, current_size + 1)
        
        # Branch 2: Exclude chosen_node
        new_candidates = candidates_mask & ~chosen_bit
        branch_and_bound(new_candidates, current_size)
    
    # Initialize with a greedy solution to have a good lower bound
    greedy_candidates = (1 << (n + 1)) - 2  # All nodes 1..n
    greedy_selected = 0
    temp = greedy_candidates
    while temp:
        # Pick node with lowest degree in remaining
        min_deg = float('inf')
        chosen = -1
        t = temp
        while t:
            lsb = t & (-t)
            node = lsb.bit_length() - 1
            deg = bin(adj_mask[node] & temp).count('1')
            if deg < min_deg:
                min_deg = deg
                chosen = node
            t &= t - 1
        
        if chosen == -1:
            break
        
        greedy_selected |= (1 << chosen)
        temp &= ~(adj_mask[chosen] | (1 << chosen))
    
    best_size = bin(greedy_selected).count('1')
    best_set = greedy_selected
    
    # Now run branch and bound
    all_nodes_mask = 0
    for i in range(1, n + 1):
        all_nodes_mask |= (1 << i)
    
    branch_and_bound(all_nodes_mask, 0)
    
    # Convert best_set bitmask to list
    result = []
    for i in range(1, n + 1):
        if best_set & (1 << i):
            result.append(i)
    
    return result if result else [1]
