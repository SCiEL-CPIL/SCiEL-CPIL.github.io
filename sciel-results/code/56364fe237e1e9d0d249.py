import time
import sys

def construct():
    n = 16
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)
    
    # Map edge to index for bitsets
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    best_count = 0
    best_edges = []
    start_time = time.time()
    timeout = 115
    
    # State for backtracking
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    # current_edges is a set of edge indices
    current_edges = set()
    
    # Order edges to find good cores early.
    # Groetzsch graph edges are a good starting point.
    # Groetzsch: 11 vertices, 20 edges.
    # Let's identify Groetzsch edges and put them first.
    # Cycle 0-1-2-3-4-0
    # Copies 5-9
    # Root 10
    # Edges:
    # (0,1), (1,2), (2,3), (3,4), (4,0)
    # (0,6), (1,5), (2,7), (3,8), (4,9) -- wait, mycielski construction:
    # v_i -> v'_i. Edges (v_i, v'_{i+1}) and (v_{i+1}, v'_i).
    # i=0: (0, 5+1=6), (1, 5+0=5)
    # i=1: (1, 5+2=7), (2, 5+1=6)
    # i=2: (2, 5+3=8), (3, 5+2=7)
    # i=3: (3, 5+4=9), (4, 5+3=8)
    # i=4: (4, 5+0=5), (0, 5+4=9)
    # Root 10 connected to all copies 5..9.
    
    groetzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 6), (1, 5),
        (1, 7), (2, 6),
        (2, 8), (3, 7),
        (3, 9), (4, 8),
        (4, 5), (0, 9),
        (5, 10), (6, 10), (7, 10), (8, 10), (9, 10)
    ]
    
    # Create a priority list for edges
    # 1. Groetzsch edges
    # 2. Other edges involving vertices 0-10
    # 3. Edges involving 11-15
    
    priority = [0] * num_edges
    for i, e in enumerate(all_edges):
        if e in groetzsch_edges:
            priority[i] = 0
        elif e[0] <= 10 and e[1] <= 10:
            priority[i] = 1
        else:
            priority[i] = 2
            
    # Sort edges by priority, then by index
    sorted_edges = sorted(range(num_edges), key=lambda i: (priority[i], i))
    
    def is_triangle_free_add(u, v):
        """Check if adding (u,v) creates a triangle."""
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        if common:
            return False
        return True

    def is_3_colorable():
        """
        Check if the current graph is 3-colorable.
        Uses backtracking with vertex ordering by degree (DSATUR-like).
        """
        # Get current degrees
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        
        # Order vertices by degree descending
        order = sorted(range(n), key=lambda i: -degrees[i])
        
        color = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Check available colors
            used_colors = 0
            for u in range(n):
                if adj[v] & (1 << u):
                    if color[u] != -1:
                        used_colors |= (1 << color[u])
            
            for c in range(3):
                if not (used_colors & (1 << c)):
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)

    def compute_max_extension():
        """
        Given the current core (which is not 3-colorable),
        find the maximum number of edges we can add while staying triangle-free.
        This is an MIS problem on the conflict graph of compatible edges.
        """
        # 1. Find all edges that can be added to the current core
        compatible = []
        for i in range(num_edges):
            if i in current_edges:
                continue
            u, v = all_edges[i]
            if is_triangle_free_add(u, v):
                compatible.append(i)
        
        m = len(compatible)
        if m == 0:
            return len(current_edges)
        
        # 2. Build conflict graph for these compatible edges
        # Two compatible edges conflict if adding both creates a triangle.
        # Triangle can be formed by:
        # a) They share a vertex and the third vertex is in the core?
        #    If e1=(u,v), e2=(v,w). Core has (u,w)? Then triangle.
        #    But we checked is_triangle_free_add for each individually.
        #    So (u,w) is NOT in core.
        # b) They share a vertex and form a triangle with each other?
        #    e1=(u,v), e2=(v,w). Do they form a triangle? Only if (u,w) is present.
        #    (u,w) is not in core. Is (u,w) one of the compatible edges?
        #    If yes, then e1, e2, and e3=(u,w) form a triangle.
        #    So e1 and e2 conflict if there exists e3 in compatible such that e1, e2, e3 form a triangle.
        #    OR if e1 and e2 themselves form a triangle with a core edge?
        #    Wait. If e1=(u,v) and e2=(v,w).
        #    If (u,w) is in core, then e1 is not compatible.
        #    So (u,w) is not in core.
        #    If we add e1 and e2, do we create a triangle?
        #    Only if (u,w) is also added.
        #    So the conflict is 3-way?
        #    No, a triangle is 3 edges.
        #    If we select a set S of edges, it is triangle-free if no 3 edges in S form a triangle.
        #    This is not a standard MIS on a pairwise conflict graph unless the conflict is pairwise.
        #    Pairwise conflict: e1 and e2 cannot be in the same set.
        #    When do e1 and e2 conflict?
        #    If e1=(u,v) and e2=(v,w).
        #    They don't form a triangle by themselves.
        #    They form a triangle with e3=(u,w).
        #    So e1 and e2 are "compatible" pairwise, but incompatible with e3.
        #    This means the "Triangle-Free" constraint is a 3-uniform hypergraph independent set, not a graph independent set.
        #    
        #    However, for small m, we can just use a greedy or small backtracking on the compatible edges.
        #    Since m is likely small (< 30), we can do a simple backtracking to maximize edges.
        
        # Let's use a simple backtracking on the compatible edges to find max subset.
        # State: index in compatible list, count, adj_mask (copy of current adj)
        # To avoid copying adj, we can pass a list of added edges and check triangles on the fly?
        # Or just copy the 16-int adj array. It's cheap.
        
        best_local = 0
        
        def backtrack_ext(idx, count, adj_copy):
            nonlocal best_local
            if idx == m:
                if count > best_local:
                    best_local = count
                return
            
            # Pruning
            if count + (m - idx) <= best_local:
                return
            
            edge_idx = compatible[idx]
            u, v = all_edges[edge_idx]
            
            # Option 1: Add edge
            if not (adj_copy[u] & adj_copy[v]): # No triangle
                # Add
                adj_copy[u] |= (1 << v)
                adj_copy[v] |= (1 << u)
                backtrack_ext(idx + 1, count + 1, adj_copy)
                # Undo
                adj_copy[u] &= ~(1 << v)
                adj_copy[v] &= ~(1 << u)
            
            # Option 2: Skip edge
            backtrack_ext(idx + 1, count, adj_copy)

        adj_copy = adj[:]
        backtrack_ext(0, 0, adj_copy)
        
        return len(current_edges) + best_local

    def search(idx):
        nonlocal best_count, best_edges
        
        if time.time() - start_time > timeout:
            return
        
        if idx == num_edges:
            return
        
        edge_idx = sorted_edges[idx]
        u, v = all_edges[edge_idx]
        
        # Option 1: Include edge
        if is_triangle_free_add(u, v):
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_edges.add(edge_idx)
            
            # Check if core is found
            if not is_3_colorable():
                # We have a non-3-colorable core.
                # Calculate max extension
                ext_count = compute_max_extension()
                if ext_count > best_count:
                    best_count = ext_count
                    # Reconstruct edges for best_edges
                    # We need to actually run the extension to get the edges.
                    # For now, we just record the count.
                    # To get the edges, we need to modify compute_max_extension to return them or run it again.
                    # Let's assume we can reconstruct or we store the core and re-extend at the end.
                    # For simplicity, we'll store the core.
                    best_edges = list(current_edges) # Placeholder, needs proper reconstruction
                    # Actually, let's just store the count for now and reconstruct at the end if needed.
                    # Or better, have compute_max_extension return the edge list.
                
                # Pruning:
                # Upper bound for any extension of this core is ext_count.
                # If ext_count <= best_count, we can prune this branch?
                # Yes, because any other path from this core will have <= ext_count edges.
                # And any path that doesn't include this edge is handled by "Option 2".
                # So if ext_count <= best_count, we don't need to explore deeper?
                # Wait. If we prune here, we assume that adding MORE edges to this core
                # won't yield a better result than the current best.
                # But we already calculated the MAX extension.
                # So yes, if max_extension <= best, prune.
                
                if ext_count <= best_count:
                    # Undo and return
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    current_edges.remove(edge_idx)
                    return
            
            search(idx + 1)
            
            # Undo
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            current_edges.remove(edge_idx)
        
        # Option 2: Exclude edge
        search(idx + 1)

    # Run search
    search(0)
    
    # If best_edges is still empty or we didn't reconstruct it properly, 
    # we might have a count but no edges.
    # The task requires returning the list of edges.
    # The current logic stores `list(current_edges)` which are indices, not [u,v] pairs.
    # And it doesn't include the extended edges.
    # We need to fix this.
    
    # Let's refine the "store best" logic.
    # We'll keep `best_core_edges` (indices) and `best_ext_edges` (indices).
    
    # Since I can't easily edit the whole logic in one go without seeing the output,
    # I will implement a simpler version that just returns the Groetzsch extension if the search fails or is slow,
    # but the search is the main part.
    
    # To ensure valid output, I will reconstruct the best solution at the end.
    # For now, the search updates best_count.
    
    # If best_count is 0, return empty or Groetzsch?
    # Groetzsch extension should be found quickly.
    
    # Let's add a fallback: if best_count == 0, run the original Groetzsch logic.
    
    if best_count == 0:
        # Fallback to Groetzsch
        groetzsch_adj = [0] * n
        for u, v in groetzsch_edges:
            groetzsch_adj[u] |= (1 << v)
            groetzsch_adj[v] |= (1 << u)
        
        # Extend
        for i in range(num_edges):
            u, v = all_edges[i]
            if not (groetzsch_adj[u] & (1 << v)):
                if not (groetzsch_adj[u] & groetzsch_adj[v]):
                    groetzsch_adj[u] |= (1 << v)
                    groetzsch_adj[v] |= (1 << u)
        
        res = []
        for u in range(n):
            for v in range(u + 1, n):
                if groetzsch_adj[u] & (1 << v):
                    res.append([u, v])
        return res
    
    # Reconstruct best_edges from best_count?
    # We didn't store the actual edges in the search loop properly.
    # I need to modify the search to store the actual edge list when best_count is updated.
    
    # Due to the constraints of the edit, I will provide the full corrected program below.
    # The logic above is the core. I will wrap it in the final function.
    
    # Since I must output SEARCH/REPLACE, I will replace the whole file content.
    pass

# The above is a draft. I will now write the actual SEARCH/REPLACE blocks.
# I will replace the entire file content with the new implementation.

