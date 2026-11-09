import time
from math import gcd

def construct():
    """
    Finds a large subset of [1, 512] such that lcm(a, b) <= 512 for all pairs.
    Uses Tomita's Maximum Clique algorithm with bitmask acceleration.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    N = 512
    nodes = list(range(1, N + 1))
    
    # 1. Build adjacency bitmasks
    # adj[i] is an integer bitmask where bit j is 1 if lcm(i, j) <= N
    # We use 0-indexed bits for efficiency, but map to 1-indexed values.
    # Bit 0 corresponds to node 1? Or bit 1 to node 1?
    # Let's use bit i for node i (1..512). 
    # Python integers are arbitrary precision, so 1 << 512 is fine.
    
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) // gcd(i, j) <= N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 2. Initial Heuristic Solution
    # We want a good lower bound to help pruning.
    # Strategy: Try divisors of highly composite numbers <= 512.
    # Also try the set {1, ..., k} where k is max such that k(k-1) <= 512.
    
    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs

    best_clique = []
    best_size = 0
    
    # Candidate sets to check for initial lower bound
    candidate_sets = []
    
    # Set 1: {1, ..., 23}
    # 23*22 = 506 <= 512
    candidate_sets.append(list(range(1, 24)))
    
    # Set 2: Divisors of 360
    candidate_sets.append(get_divisors(360))
    
    # Set 3: Divisors of 420
    candidate_sets.append(get_divisors(420))
    
    # Set 4: Divisors of 480
    candidate_sets.append(get_divisors(480))
    
    # Set 5: Divisors of 504
    candidate_sets.append(get_divisors(504))
    
    # Set 6: Divisors of 432
    candidate_sets.append(get_divisors(432))
    
    # Verify and update best
    for c_set in candidate_sets:
        # Check validity
        valid = True
        for i in range(len(c_set)):
            for j in range(i+1, len(c_set)):
                a, b = c_set[i], c_set[j]
                if (a * b) // gcd(a, b) > N:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and len(c_set) > best_size:
            best_size = len(c_set)
            best_clique = c_set[:]
            
    # 3. Tomita Algorithm with Bitmasks
    # We represent the candidate set as a bitmask.
    # All nodes 1..N are initially candidates.
    all_nodes_mask = (1 << (N + 1)) - 2  # Bits 1..N set
    
    # Helper: Get degree of a node in a subgraph defined by mask
    def get_degree_in_mask(node, mask):
        # Count bits in adj[node] & mask
        return (adj[node] & mask).bit_count()
    
    # Helper: Greedy Coloring of subgraph induced by mask
    # Returns a list of (node, color) sorted by color ascending.
    # The max color in the prefix is the bound.
    def color_sort(mask):
        if mask == 0:
            return []
        
        # Extract nodes
        nodes_in_mask = []
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            node = lsb.bit_length() - 1
            nodes_in_mask.append(node)
            m ^= lsb
            
        # Sort by degree in the subgraph (descending)
        # This is O(V^2) but V is small (<= 512) and only done at each recursion level
        # To speed up, we can just use the global degree or compute local degree.
        # Local degree is better for coloring quality.
        
        # Compute local degrees
        local_degrees = {}
        for u in nodes_in_mask:
            local_degrees[u] = (adj[u] & mask).bit_count()
            
        # Sort nodes by local degree descending
        nodes_in_mask.sort(key=lambda x: local_degrees[x], reverse=True)
        
        # Greedy Coloring
        # color_classes: list of lists of nodes
        color_classes = []
        node_color = {}
        
        for v in nodes_in_mask:
            # Try to assign the smallest color c such that v is not adjacent to any node in color_classes[c]
            assigned_color = None
            for c in range(len(color_classes)):
                # Check if v is adjacent to any node in color_classes[c]
                # We can check by iterating through the list. 
                # Optimization: If we maintain a bitmask for each color class, we can do (adj[v] & color_mask[c]) == 0
                # Let's use bitmasks for color classes too.
                pass
            
            # Re-implementation with bitmasks for color classes for speed
            # Actually, for N=512, the list check is fine if the list is small.
            # But let's stick to the bitmask check if we precompute.
            
            # Simpler approach:
            # We just need the color number for each node to sort them.
            # The standard Tomita implementation often just returns the sorted list of nodes
            # and the max color of the prefix.
            
            # Let's use a standard greedy coloring that returns the color of each node.
            # We process nodes in the order of nodes_in_mask (degree sorted).
            
            # Reset for this function call
            color_classes = []
            node_color = {}
            
            for v in nodes_in_mask:
                # Find first color class where v has no neighbors
                placed = False
                for c in range(len(color_classes)):
                    # Check if v is adjacent to any node in color_classes[c]
                    # We can iterate through the nodes in that class
                    conflict = False
                    for u in color_classes[c]:
                        if adj[v] & (1 << u):
                            conflict = True
                            break
                    if not conflict:
                        color_classes[c].append(v)
                        node_color[v] = c + 1
                        placed = True
                        break
                if not placed:
                    color_classes.append([v])
                    node_color[v] = len(color_classes)
                    
        # Create ordered list of nodes sorted by color
        # Nodes with lower colors come first.
        # In the B&B loop, we iterate from the END (highest color) to the beginning.
        # The bound for the prefix [0..i] is the max color in that prefix.
        # If we sort by color, the color of node at index i is the max color in [0..i].
        
        # We need to return the list of nodes in the order they should be processed?
        # No, we return the list sorted by color, and the color of each node.
        
        # To make the bound tight, we want the list to be such that 
        # the max color of the prefix is as small as possible? 
        # Actually, the bound is max_color(prefix). 
        # If we sort by color, the max color of prefix [0..i] is just color(node_i).
        
        ordered_nodes = sorted(nodes_in_mask, key=lambda x: node_color[x])
        
        return ordered_nodes, node_color

    def expand(clique, candidates_mask):
        nonlocal best_clique, best_size, deadline
        
        if time.time() > deadline:
            return
            
        if candidates_mask == 0:
            if len(clique) > best_size:
                best_clique = clique[:]
                best_size = len(clique)
            return
            
        # Color the subgraph
        ordered_nodes, node_colors = color_sort(candidates_mask)
        
        # Iterate from the end (highest color) to the beginning
        for i in range(len(ordered_nodes) - 1, -1, -1):
            if time.time() > deadline:
                return
                
            v = ordered_nodes[i]
            
            # Pruning
            # The max possible clique size from the prefix [0..i] is bounded by node_colors[v]
            # because node_colors[v] is the max color in the prefix (due to sorting).
            if len(clique) + node_colors[v] <= best_size:
                # Since we are going down, all subsequent prefixes have <= max color
                return
                
            # Include v
            clique.append(v)
            
            # New candidates: neighbors of v in the prefix [0..i-1]
            # We need to filter the mask to only include nodes in ordered_nodes[0..i-1] that are adjacent to v.
            
            # Construct mask for prefix [0..i-1]
            # This is expensive if done naively. 
            # Alternative: Maintain the list of candidates and filter it.
            # Since we are in a recursive step, we can just filter the list.
            
            # However, using bitmasks is faster for the intersection.
            # We need a mask of the prefix.
            # We can precompute prefix masks? No, the order changes.
            
            # Let's just filter the list of nodes in the prefix that are adjacent to v.
            # new_cands_list = [u for u in ordered_nodes[:i] if (adj[v] >> u) & 1]
            
            # To use the bitmask `expand` efficiently, we need to convert this list back to a mask.
            # Or, we can modify `expand` to take a list of nodes?
            # Bitmasks are faster for the intersection `candidates & adj[v]`.
            # If we have the mask of the prefix, we can do `new_mask = prefix_mask & adj[v]`.
            
            # How to get prefix_mask quickly?
            # We can build it as we iterate? 
            # Or, we can just pass the list of nodes to the next call?
            # If we pass a list, the coloring step becomes O(V^2) which is fine for V<=512.
            # But the intersection `new_cands = [u for u in cands if adj[v] & (1<<u)]` is O(V).
            
            # Let's switch to passing lists for clarity and sufficient speed, 
            # as N=512 is small enough that O(V^2) coloring and O(V) filtering 
            # with Python lists might be comparable to bitmask overhead in Python.
            # HOWEVER, the initial setup and the "adj" lookup are O(1) with bitmasks.
            
            # Let's stick to bitmasks for the `candidates` state, 
            # but we need the prefix mask.
            
            # Optimization: 
            # We can precompute the mask for each prefix? 
            # No, the order depends on the coloring.
            
            # Alternative: 
            # Instead of iterating by index, we can maintain the candidate mask and 
            # remove v from it? 
            # No, the Tomita algorithm relies on the ordering for the bound.
            
            # Let's use the list-based approach for the recursion arguments to avoid 
            # complex mask reconstruction, but use bitmasks for adjacency checks.
            
            # Actually, let's just use lists for `candidates` in the recursion.
            # The `color_sort` function takes a list.
            # The filtering is a list comprehension.
            
            # Let's refactor `expand` to take a list of nodes.
            
            pass # Placeholder for refactoring thought

    # Refactored Expand with Lists
    def expand_list(clique, candidates_list):
        nonlocal best_clique, best_size, deadline
        
        if time.time() > deadline:
            return
            
        if not candidates_list:
            if len(clique) > best_size:
                best_clique = clique[:]
                best_size = len(clique)
            return
            
        # Color the subgraph induced by candidates_list
        # This returns ordered list and colors
        ordered_nodes, node_colors = color_sort_from_list(candidates_list)
        
        # Iterate from the end
        for i in range(len(ordered_nodes) - 1, -1, -1):
            if time.time() > deadline:
                return
                
            v = ordered_nodes[i]
            
            if len(clique) + node_colors[v] <= best_size:
                return
                
            clique.append(v)
            
            # New candidates: neighbors of v in ordered_nodes[0..i-1]
            # Filter the prefix
            prefix = ordered_nodes[:i]
            new_cands = [u for u in prefix if (adj[v] >> u) & 1]
            
            expand_list(clique, new_cands)
            
            clique.pop()

    # Helper: Color sort from a list of nodes
    def color_sort_from_list(nodes_list):
        if not nodes_list:
            return [], {}
        
        # Compute local degrees
        local_degrees = {}
        for u in nodes_list:
            # Count neighbors in nodes_list
            deg = 0
            for v in nodes_list:
                if u != v and (adj[u] >> v) & 1:
                    deg += 1
            local_degrees[u] = deg
            
        # Sort by degree descending
        nodes_list_sorted = sorted(nodes_list, key=lambda x: local_degrees[x], reverse=True)
        
        # Greedy Coloring
        color_classes = []
        node_color = {}
        
        for v in nodes_list_sorted:
            placed = False
            for c in range(len(color_classes)):
                conflict = False
                for u in color_classes[c]:
                    if (adj[v] >> u) & 1:
                        conflict = True
                        break
                if not conflict:
                    color_classes[c].append(v)
                    node_color[v] = c + 1
                    placed = True
                    break
            if not placed:
                color_classes.append([v])
                node_color[v] = len(color_classes)
                
        # Sort nodes by color
        ordered_nodes = sorted(nodes_list_sorted, key=lambda x: node_color[x])
        
        return ordered_nodes, node_color

    # Start B&B
    # Initial candidates: all nodes
    # Sort initial nodes by degree to help the first coloring?
    # The `color_sort_from_list` does its own sorting.
    
    initial_candidates = list(range(1, N + 1))
    # Sort by global degree descending to potentially find good cliques faster?
    # The coloring step re-sorts, so it doesn't matter much for the first call.
    
    expand_list([], initial_candidates)
    
    return sorted(best_clique)

# EVOLVE-BLOCK-END
