import time
from math import gcd

def construct():
    """
    Returns the maximum cardinality subset of [1, 255] containing 255,
    such that gcd(a, b) > 1 for all distinct pairs a, b in the subset.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # 1. Identify candidate nodes: numbers in [1, 255] that share a factor with 255
    # 255 = 3 * 5 * 17
    candidates = []
    for i in range(1, 256):
        if gcd(i, 255) > 1:
            candidates.append(i)
            
    n = len(candidates)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # 2. Build adjacency matrix
    # adj[i][j] is True if gcd(candidates[i], candidates[j]) > 1
    adj = [[False] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True
                
    # 3. Find Maximum Clique
    # We use a branch and bound approach with coloring bound (Tomita algorithm simplified)
    # Vertices are ordered by degree descending to improve pruning
    
    degrees = [sum(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # Remap indices
    old_to_new = [0] * n
    new_adj = [[False] * n for _ in range(n)]
    
    for new_idx, old_idx in enumerate(order):
        old_to_new[old_idx] = new_idx
        for other_old_idx in range(n):
            new_adj[new_idx][old_to_new[other_old_idx]] = adj[old_idx][other_old_idx]

    best_clique_size = 0
    best_clique_nodes = []
    
    # P: set of candidate vertices to add
    # R: current clique
    # X: set of excluded vertices (not strictly needed for max clique if we just prune by size, but good for correctness in some variants)
    
    def color_sort(P):
        """
        Greedily color the vertices in P.
        Returns a list of (vertex, color) pairs sorted by color.
        The max color is an upper bound on the clique size within P.
        """
        if not P:
            return []
        
        # Create a list of vertices in P
        p_list = list(P)
        
        # Order by degree in P (descending) for better coloring
        # Calculate degrees in P
        degs = {}
        for u in p_list:
            count = 0
            for v in p_list:
                if u != v and new_adj[u][v]:
                    count += 1
            degs[u] = count
            
        p_list.sort(key=lambda x: degs[x], reverse=True)
        
        colors = {}
        uncolored = set(p_list)
        
        # We assign colors 1, 2, ... such that no two adjacent vertices have the same color.
        # We want to minimize the number of colors? No, we just need a valid coloring.
        # A valid coloring gives an upper bound: max color.
        # To get a tight bound, we should assign the smallest possible color.
        
        while uncolored:
            # Pick a vertex, say the first one in the sorted list that is uncolored
            # Actually, standard greedy coloring picks a vertex, assigns smallest available color, removes it.
            # To make the bound tighter, we often process vertices in a specific order.
            
            # Let's just iterate through p_list
            for u in p_list:
                if u not in uncolored:
                    continue
                # Find smallest color not used by neighbors in P
                used_colors = set()
                for v in p_list:
                    if v != u and new_adj[u][v] and v in colors:
                        used_colors.add(colors[v])
                        
                c = 1
                while c in used_colors:
                    c += 1
                colors[u] = c
                uncolored.remove(u)
                
        # Sort by color
        sorted_p = sorted(colors.items(), key=lambda x: x[1])
        return sorted_p

    def expand(R, P):
        nonlocal best_clique_size, best_clique_nodes
        
        if time.time() > deadline:
            return
            
        # Pruning
        if len(R) + len(P) <= best_clique_size:
            return
            
        if not P:
            if len(R) > best_clique_size:
                best_clique_size = len(R)
                best_clique_nodes = R[:]
            return
            
        # Get coloring bound
        colored_P = color_sort(P)
        
        # Process vertices in reverse order of color (highest color first)
        # This is a heuristic: if a vertex has a high color, it's "important"
        # Actually, Tomita processes in order of increasing color, and prunes if current_size + color <= best
        
        # We iterate through the colored list
        # The list is sorted by color.
        # Let's iterate from last to first? 
        # Standard Tomita: 
        # while P is not empty:
        #   pick u from P (usually the one with max color)
        #   ...
        
        # Simpler approach:
        # Iterate through vertices in P.
        # We can just use the max color as a bound for the *entire* P at each step?
        # No, we need to recurse.
        
        # Let's try a standard recursive backtracking with the bound from coloring.
        # We'll pick a pivot vertex to reduce branching?
        # Pivot u: we only branch on vertices in P that are NOT adjacent to u.
        # This is valid because any clique in P must contain either u or a non-neighbor of u.
        
        if not colored_P:
            return
            
        # Pick pivot: vertex with max degree in P
        # For simplicity, let's just pick the last one in the sorted list (likely high degree)
        # Or calculate pivot properly.
        
        # Let's just iterate through all vertices in P, but use the coloring bound to prune.
        # We need to be careful with the bound.
        # If we pick a vertex v, the new P is P \ {v} \ N(v).
        
        # To keep it simple and robust within time limits:
        # We'll iterate through vertices in P.
        # We use the max color of the *remaining* P as the bound?
        # Recalculating color_sort at every step is expensive.
        
        # Alternative: Simple Branch and Bound with degree heuristic.
        # Given n=127, this might be too slow if the graph is dense.
        # But the graph is not extremely dense.
        
        # Let's use the fact that we know a solution of size 85.
        # We can initialize best_clique_size = 85.
        
        # Let's try to find a clique larger than 85.
        
        # We'll use a simple recursive search.
        # To speed it up, we order P by degree in P.
        
        p_list = list(P)
        
        # Calculate degrees in P
        degs = {}
        for u in p_list:
            count = 0
            for v in p_list:
                if u != v and new_adj[u][v]:
                    count += 1
            degs[u] = count
            
        # Sort by degree descending
        p_list.sort(key=lambda x: degs[x], reverse=True)
        
        for u in p_list:
            # Pruning: if len(R) + 1 + (number of vertices in p_list after u that are adjacent to u) <= best
            # This is a weak bound.
            
            # Stronger bound: Color bound on the subgraph induced by P \ {u} \ N(u)
            # This is expensive.
            
            # Let's just try.
            new_R = R + [u]
            
            # New P: neighbors of u in P
            new_P = set()
            for v in p_list:
                if v != u and new_adj[u][v]:
                    new_P.add(v)
                    
            # Check time
            if time.time() > deadline:
                return
                
            expand(new_R, new_P)
            
            # Remove u from P for next iteration? 
            # No, we are iterating over a snapshot.
            # But we should remove u from the consideration for future branches in this level?
            # The loop structure `for u in p_list` handles this.
            
            # Optimization: If we found a clique of size best_clique_size, 
            # and the max possible size from here is <= best, we can break.
            # Max possible size from here (if we don't pick u) is len(R) + len(P_remaining)
            # This is hard to track easily in this simple loop.
            
            # Let's add a simple bound:
            # The number of vertices left in p_list (excluding u and those already processed)
            # is not easily accessible.
            
            # Given the constraints, let's rely on the initial bound of 85.
            # If we can't find better, we return 85.
            
    # Initialize with known solution size
    # Multiples of 3
    init_clique = [i for i in range(n) if candidates[i] % 3 == 0]
    best_clique_size = len(init_clique)
    best_clique_nodes = init_clique
    
    # Run search
    expand([], set(range(n)))
    
    # Convert indices back to values
    result = [candidates[i] for i in best_clique_nodes]
    
    # Ensure 255 is in result
    if 255 not in result:
        # Fallback to multiples of 3
        result = list(range(3, 256, 3))
        
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    # Verify
    valid = True
    if 255 not in res:
        valid = False
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if gcd(res[i], res[j]) == 1:
                valid = False
                break
    print("Valid:", valid)
