import math
import time
from typing import List, Set, Tuple

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, with gcd(a, b) > 1 
    for all distinct pairs. Maximizes cardinality.
    
    Strategy:
    1. Identify the candidate set: numbers in [1, 255] sharing a factor with 255.
    2. Build the graph where edges represent gcd > 1.
    3. Use Tomita's algorithm (Branch and Bound with greedy coloring) to find 
       the maximum clique containing 255.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # 1. Generate candidates
    # 255 = 3 * 5 * 17
    # Candidates must be divisible by 3, 5, or 17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    if n == 0:
        return [255] # Should not happen
        
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    idx_255 = val_to_idx[255]
    
    # 2. Build adjacency list
    # adj[i] is a set of indices j such that gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # 3. Tomita's Algorithm for Maximum Clique
    # We want the max clique containing idx_255.
    # We can fix idx_255 in the solution and search for the max clique in its neighborhood, then add 1.
    # Or just run max clique on the whole graph and ensure 255 is in it?
    # Since 255 is connected to all candidates (by definition of candidates), 
    # any max clique in the candidate graph might not contain 255.
    # However, 255 is a "hub". 
    # Actually, if we find the max clique of the whole graph, does it contain 255?
    # Not necessarily. But we require 255 to be in the set.
    # So we restrict the search to the neighborhood of 255, plus 255 itself.
    
    # P0 = neighbors of 255
    P0 = adj[idx_255].copy()
    
    best_clique_size = 0
    best_clique_indices = set()
    
    # Global best tracking
    def update_best(clique_indices, size):
        nonlocal best_clique_size, best_clique_indices
        if size > best_clique_size:
            best_clique_size = size
            best_clique_indices = set(clique_indices)
    
    # Greedy coloring function
    # Returns a list of colors for nodes in P, and the max color
    # Nodes are indexed 0..len(P)-1 in the list P
    def color_sort(P_list):
        # P_list is a list of node indices
        # We want to color the subgraph induced by P_list
        # such that adjacent nodes have different colors.
        # We use a simple greedy sequential coloring.
        # To improve bound, we sort nodes by degree in P_list descending?
        # Tomita's original paper suggests coloring in reverse order of a specific heuristic.
        
        # Simple greedy:
        # colors[i] is the color of P_list[i]
        colors = [0] * len(P_list)
        # map node -> color used
        # We iterate through nodes, assign smallest available color not used by neighbors in P_list
        
        # For efficiency, we can just use a standard greedy coloring.
        # Since P is small (<= 127), this is fast.
        
        # To get a tighter bound, we want the max color to be small.
        # Greedy coloring usually gives a decent upper bound.
        
        # Let's sort P_list by degree in the subgraph induced by P_list, descending.
        # This often leads to better coloring (fewer colors).
        
        deg_in_p = {v: 0 for v in P_list}
        p_set = set(P_list)
        for v in P_list:
            for u in adj[v]:
                if u in p_set:
                    deg_in_p[v] += 1
        
        # Sort by degree descending
        sorted_P = sorted(P_list, key=lambda x: deg_in_p[x], reverse=True)
        
        # Now color sorted_P
        # We need to map back to original indices or just return the colored list
        # The algorithm needs to iterate over P in a specific order.
        # Usually, we iterate over P in the order of increasing color.
        
        # Let's implement a coloring that returns (sorted_nodes, colors)
        # where sorted_nodes[k] has color colors[k].
        
        node_to_color = {}
        current_max_color = 0
        
        # We process nodes one by one.
        # To make the bound tight, we want to use as few colors as possible.
        # Standard greedy: for each node, pick smallest color not used by already colored neighbors.
        
        # However, Tomita's algorithm typically uses a coloring of the entire P set 
        # and then branches on nodes with the highest color first (or lowest, depending on implementation).
        # Common implementation:
        # 1. Color P.
        # 2. Let max_color be the max color.
        # 3. If |R| + max_color <= best, prune.
        # 4. Iterate v in P in order of decreasing color.
        
        # Let's do a simple sequential coloring on the sorted list.
        # This is O(N^2) which is fine for N=127.
        
        colored_nodes = []
        node_colors = []
        
        # We need to be careful: the coloring must be valid for the subgraph.
        # If we color node v with c, no neighbor u in P_list can have c.
        
        # Let's just use a standard greedy coloring on the set P_list.
        # We'll return a list of (node, color) sorted by color.
        
        # To ensure we get a good bound, we can use the "reverse order of appearance" 
        # or similar heuristics, but simple greedy is robust.
        
        # Let's use a dictionary to track colors used by neighbors.
        # This is getting complex to optimize perfectly. 
        # Given N=127, even a simple O(N^2) coloring is fast.
        
        # Simplified approach:
        # Assign colors 1, 2, ... to nodes in P_list.
        # We want the max color to be an upper bound on clique size.
        
        # Let's sort P_list by degree descending to help coloring.
        # Then color sequentially.
        
        # Actually, a very effective bound is provided by coloring the graph 
        # such that we use the minimum number of colors (chromatic number approximation).
        
        # Let's implement a simple greedy coloring.
        # Input: list of nodes P_list
        # Output: list of (node, color) sorted by color
        
        # We'll map node to its assigned color
        assigned_colors = {}
        max_c = 0
        
        # Iterate through nodes in P_list (sorted by degree desc)
        for v in sorted_P:
            # Find colors used by neighbors of v in P_list
            neighbor_colors = set()
            for u in adj[v]:
                if u in assigned_colors:
                    neighbor_colors.add(assigned_colors[u])
            
            # Find smallest available color
            c = 1
            while c in neighbor_colors:
                c += 1
            
            assigned_colors[v] = c
            if c > max_c:
                max_c = c
        
        # Create list of (node, color)
        colored = [(v, assigned_colors[v]) for v in P_list]
        
        # Sort by color
        colored.sort(key=lambda x: x[1])
        
        return colored, max_c

    def bron_kerbosch_tomita(R, P):
        nonlocal best_clique_size, best_clique_indices
        
        if time.time() > deadline:
            return
            
        if len(P) == 0:
            if len(R) > best_clique_size:
                best_clique_size = len(R)
                best_clique_indices = set(R)
            return
        
        # Coloring
        P_list = list(P)
        colored, max_color = color_sort(P_list)
        
        # Pruning: if current clique size + max possible extension <= best, return
        if len(R) + max_color <= best_clique_size:
            return
            
        # Iterate over nodes in P in order of decreasing color
        # This is a key part of Tomita's algorithm: branch on nodes with highest color first
        # to find large cliques early.
        
        # We have 'colored' sorted by increasing color.
        # We want to process highest color first.
        
        # We need to maintain P as a set for fast intersection.
        # But we are iterating over a specific order.
        
        # Let's iterate from the end of the colored list (highest color) to the start.
        # However, we must remove the node from P after processing it.
        
        # Note: The standard Tomita algorithm uses a specific data structure.
        # Here, we use sets for P.
        
        # To handle the ordering correctly with sets:
        # We create a list of nodes to visit: [node with max color, ..., node with min color]
        # We iterate through this list.
        
        visit_order = [node for node, c in reversed(colored)]
        
        for v in visit_order:
            if time.time() > deadline:
                return
            
            # Check if v is still in P (it should be, unless we removed it in a previous iteration? 
            # No, we remove it *after* the recursive call returns).
            # But wait, if we iterate over a static list, and we remove v from P,
            # subsequent nodes in the list are still in P (unless they were removed? No, only v is removed).
            
            # However, the coloring was done on the *original* P.
            # After removing v, the coloring bound for the remaining P might be lower, 
            # but we don't re-color in this simple implementation. 
            # This is acceptable for small graphs.
            
            if v not in P:
                continue
                
            new_R = R | {v}
            new_P = P & adj[v]
            
            bron_kerbosch_tomita(new_R, new_P)
            
            P.remove(v)
            
            # Optional: Re-coloring P after each removal to tighten bound further
            # This is expensive. For N=127, the initial bound is likely tight enough.
            
    # Initialize
    R = {idx_255}
    P = adj[idx_255].copy()
    
    # Run Tomita
    bron_kerbosch_tomita(R, P)
    
    # Convert result
    if best_clique_indices:
        result_indices = best_clique_indices | {idx_255}
        result = sorted([candidates[i] for i in result_indices])
        # Verify 255 is present
        if 255 in result:
            return result
    
    # Fallback: Multiples of 3
    return [i for i in range(3, 256, 3)]

# EVOLVE-BLOCK-END
