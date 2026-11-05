import math
import time

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin before 90s kill
    
    N = 255
    
    # Step 1: Identify candidates
    # A number x must have gcd(x, 255) > 1
    # 255 = 3 * 5 * 17
    candidates = [x for x in range(1, N + 1) if math.gcd(x, 255) > 1]
    
    # Ensure 255 is in candidates (it is, gcd(255,255)=255)
    # We will solve for the max clique in the subgraph excluding 255,
    # then add 255 back, since 255 is compatible with all candidates.
    
    if 255 not in candidates:
        candidates.append(255)
        
    # Separate 255
    idx_255 = candidates.index(255)
    other_candidates = [x for i, x in enumerate(candidates) if i != idx_255]
    
    n = len(other_candidates)
    
    # Step 2: Build adjacency matrix for the graph where edge means gcd > 1
    # adj[i][j] is True if gcd(other_candidates[i], other_candidates[j]) > 1
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(other_candidates[i], other_candidates[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True
    
    # Step 3: Find Maximum Clique
    # We use a branch and bound algorithm with greedy coloring (Tomita's algorithm)
    # to efficiently find the maximum clique.
    
    best_clique = []
    best_size = 0
    
    # Precompute degrees for initial sorting or just use the algorithm
    # Tomita's algorithm requires coloring the candidates at each step.
    
    def color_sort(candidates_list):
        """
        Greedily color the candidates to provide an upper bound on clique size.
        Returns a list of (node, color) sorted by color.
        The color of a node is an upper bound on the size of the clique
        that can be formed from that node and subsequent nodes in the sorted list.
        """
        if not candidates_list:
            return []
        
        # We want to sort by color to prune.
        # Standard Tomita:
        # 1. Color the graph induced by candidates_list greedily.
        # 2. Sort nodes by color in non-decreasing order.
        
        # Greedy coloring
        colors = [0] * len(candidates_list)
        color_map = {} # node_index -> color
        
        # We need to map indices in candidates_list to actual node indices in 0..n-1
        # candidates_list contains actual node indices (0..n-1)
        
        # Simple greedy coloring:
        # Iterate through nodes, assign smallest available color
        # This is O(N^2) per call, which is fine for N=126.
        
        # To make it faster and provide a tighter bound, we can use a specific order.
        # But standard greedy is okay.
        
        # Let's use a simple greedy coloring where we process nodes in the order they appear
        # in candidates_list.
        
        used_colors = set()
        # This simple approach doesn't guarantee the *best* coloring (min colors),
        # but it provides a valid upper bound for each node if we process carefully?
        # Actually, Tomita's algorithm relies on the property that if we sort by color,
        # the color of the last node is the max clique size bound for the whole set.
        # The color of a node v is the max color assigned to its neighbors + 1?
        # No, it's just a valid coloring. The color class size is not the bound.
        # The *color number* assigned to a node is the bound for the subgraph induced by that node and subsequent nodes.
        
        # Let's implement the standard greedy coloring used in Tomita's MCS algorithm.
        # We sort the input candidates by degree (descending) often helps, but the coloring
        # itself depends on the order.
        
        # For simplicity and correctness with N=126, we can just use a recursive backtracking
        # with a simple bound: the size of the candidate list.
        # With N=126, the search space might be large if not pruned well.
        # However, the graph is dense (many gcds > 1).
        # Let's try a simpler Branch and Bound first.
        
        pass

    # Given the small N, a simple recursive search with pruning based on candidate count
    # might be sufficient, but let's implement a slightly more optimized version.
    
    # We represent the set of candidates as a bitset (integer) for fast operations.
    # n <= 126, so we need more than 64 bits. Python integers handle arbitrary precision.
    
    # Precompute bitmasks for neighbors
    # mask[i] is an integer with bit j set if i and j are connected.
    masks = [0] * n
    for i in range(n):
        m = 0
        for j in range(n):
            if adj[i][j]:
                m |= (1 << j)
        masks[i] = m
        
    # Recursive function
    # clique: list of nodes in current clique
    # cand_mask: bitmask of candidates that are adjacent to all nodes in clique
    
    def expand(clique, cand_mask):
        nonlocal best_clique, best_size
        
        if cand_mask == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = list(clique)
            return
            
        # Pruning: if current clique size + number of candidates <= best_size, prune
        num_cands = bin(cand_mask).count("1")
        if len(clique) + num_cands <= best_size:
            return
            
        # Optional: Greedy coloring bound
        # For N=126, let's try to extract a list of candidates
        cand_list = []
        temp_mask = cand_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            cand_list.append(idx)
            temp_mask ^= lsb
            
        # Simple heuristic: pick a candidate, branch on include/exclude
        # To improve, pick the candidate with the highest degree in the subgraph induced by cand_mask?
        # Or just pick the first one.
        
        # Let's use a greedy coloring bound to be safe.
        # We will color the subgraph induced by cand_list.
        # This requires building the subgraph adjacency.
        
        # Given time constraints and N=126, let's use a simpler but effective strategy:
        # Try to add candidates in a specific order.
        
        # Heuristic: Sort candidates by degree in the remaining graph (descending)
        # This often finds large cliques early, improving the bound.
        
        # Calculate degrees within cand_mask
        degrees = [0] * len(cand_list)
        for i, u in enumerate(cand_list):
            deg = 0
            # Count neighbors in cand_mask
            neigh_mask = masks[u] & cand_mask
            deg = bin(neigh_mask).count("1")
            degrees[i] = deg
            
        # Sort indices by degree descending
        sorted_indices = sorted(range(len(cand_list)), key=lambda i: degrees[i], reverse=True)
        
        # Process in this order
        # We need to maintain the cand_mask correctly.
        # This "sort and then branch" approach is tricky with bitmasks if we don't re-sort at every level.
        # Standard Tomita re-colors at every level.
        
        # Let's stick to a robust, simpler Branch and Bound without complex coloring first,
        # relying on the density of the graph.
        # If it's too slow, we'll add coloring.
        
        # Pick a node to branch on.
        # Let's pick the first one in the sorted list.
        if not sorted_indices:
            return
            
        # We will iterate through candidates in the order of sorted_indices
        # But we need to update cand_mask as we go.
        
        # A better approach for small N:
        # Pick a pivot node p.
        # Partition candidates into N(p) and Non-N(p).
        # Any maximal clique must contain p OR be contained in Non-N(p).
        
        # Let's implement Bron-Kerbosch with Pivot.
        
        def bron_kerbosch(R, P, X):
            nonlocal best_clique, best_size
            
            if not P and not X:
                if len(R) > best_size:
                    best_size = len(R)
                    best_clique = list(R)
                return
            
            # Pivot selection
            # Choose u in P U X that maximizes |N(u) intersection P|
            # If P is empty, we are at a leaf.
            
            # Convert sets to lists for iteration
            # Using Python sets for P and X is efficient for small N.
            
            if P or X:
                # Find pivot u
                pivot = None
                max_count = -1
                for u in list(P) + list(X):
                    # Count neighbors of u in P
                    # We need to check intersection of neighbors of u with P
                    # Using bitmasks is faster here.
                    pass
            
            # Given the complexity of implementing full Bron-Kerbosch with sets and bitmasks
            # and the small N, let's use a simpler recursive search that picks a node,
            # includes it (intersect P with neighbors), or excludes it (remove from P, add to X).
            # This is essentially the same as above but explicit.
            
            # To ensure speed, we will use the bitmask approach.
            # P is a bitmask. X is a bitmask.
            # R is a list.
            
            pass

    # Let's use a standard, efficient implementation for Max Clique on small graphs.
    # We'll use the "color bound" pruning which is very effective.
    
    def max_clique_bitmask(cand_mask):
        """
        Finds max clique in the subgraph induced by cand_mask.
        Returns the size and the set of nodes (indices).
        """
        nonlocal best_size, best_clique
        
        if cand_mask == 0:
            return 0, set()
            
        # Get list of nodes
        nodes = []
        temp = cand_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            nodes.append(idx)
            temp ^= lsb
            
        # Greedy Coloring to get an upper bound
        # We color the graph induced by 'nodes'.
        # We sort nodes by degree in the induced subgraph?
        # Tomita's algorithm sorts by color after coloring.
        
        # Simple Greedy Coloring:
        # Assign colors to nodes such that adjacent nodes have different colors.
        # The number of colors used is an upper bound on the clique size.
        # We can also use the color of each node as a bound for the subproblem starting from that node.
        
        # Let's implement a simple greedy coloring that returns a list of (node, color)
        # sorted by color.
        
        color_of_node = {}
        colors_used = 0
        # Sort nodes by degree descending to improve coloring quality
        degs = []
        for u in nodes:
            deg = bin(masks[u] & cand_mask).count("1")
            degs.append((deg, u))
        degs.sort(reverse=True)
        
        # Assign colors
        # We maintain a list of color classes (sets of nodes)
        # Or simply assign the smallest available color.
        
        # For the bound, we just need a valid coloring.
        # Let's use a simple sequential coloring.
        
        node_colors = []
        available_colors = [True] * (len(nodes) + 1) # Max colors is N
        
        # This simple coloring might not be optimal, but it's fast.
        # Actually, for pruning, we want the colors to be as small as possible.
        # Greedy coloring in order of descending degree is a good heuristic.
        
        used_colors_for_node = {}
        current_max_color = 0
        
        # We will store the color assigned to each node
        # and the order of nodes sorted by color.
        
        # To keep it simple and fast for N=126:
        # We will just use the size of the candidate set as the bound initially,
        # and rely on the recursive structure.
        # However, without coloring, the tree can be large.
        
        # Let's try a different approach:
        # Since N is small, we can use `itertools.combinations` to check cliques of decreasing size?
        # No, that's too slow.
        
        # Let's use a standard recursive backtracking with the "first node" branching.
        
        def solve(clique, mask):
            nonlocal best_size, best_clique
            
            if mask == 0:
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = list(clique)
                return
                
            # Bound: len(clique) + popcount(mask) <= best_size
            if len(clique) + bin(mask).count("1") <= best_size:
                return
                
            # Pick a node u from mask
            # Heuristic: pick node with max degree in mask
            lsb = mask & -mask
            u = lsb.bit_length() - 1
            
            # Option 1: Include u
            # New clique: clique + [u]
            # New mask: mask & masks[u] (neighbors of u in mask)
            solve(clique + [u], mask & masks[u])
            
            # Option 2: Exclude u
            # New clique: clique
            # New mask: mask & ~lsb (remove u from mask)
            solve(clique, mask ^ lsb)
            
        solve([], cand_mask)
        return best_size, set(best_clique)

    # Initialize best
    best_size = 0
    best_clique = []
    
    # Run solver
    # The solver above updates global best_size and best_clique.
    # We need to pass the initial mask for all other candidates.
    initial_mask = (1 << n) - 1
    
    max_clique_bitmask(initial_mask)
    
    # Reconstruct the set
    result_set = set(other_candidates[i] for i in best_clique)
    result_set.add(255)
    
    # Verify
    valid = True
    vals = list(result_set)
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if math.gcd(vals[i], vals[j]) == 1:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback
        return sorted(x for x in range(1, N + 1) if x % 3 == 0)
        
    return sorted(result_set)

construct()
