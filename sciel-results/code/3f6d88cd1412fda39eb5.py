    def is_2_colorable(n, edges):
        # edges: list of tuples
        # n: number of vertices
        # Returns True if 2-colorable
        
        # Preprocess edges
        # Map vertex to list of edge indices
        vertex_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                vertex_edges[v].append(i)
        
        # State:
        # color[v] in {-1, 0, 1}
        # unassigned_count[e]
        
        # Better: Use recursion with bitmasks for speed?
        # With n <= 26, 2^26 is too big for brute force.
        # Backtracking with UP is best.
        
        assignment = [-1] * n
        # unassigned_count per edge
        unassigned = [5] * len(edges)
        
        # For UP, we need to know which vertices are in each edge.
        # And for each vertex, which edges it's in.
        
        def propagate():
            # Returns False if contradiction, True otherwise.
            # We need a queue of vertices that changed? Or just scan?
            # Scanning all edges is O(E). E <= 200. N <= 26.
            # Depth <= 26. 26 * 200 = 5200 ops per path.
            # Branching factor 2. Worst case 2^26 is huge, but UP prunes massively.
            
            changed = True
            while changed:
                changed = False
                for i, e in enumerate(edges):
                    # Check if this edge is affected?
                    # For simplicity in this rewrite, we just scan all edges.
                    # Optimization: Only check edges where unassigned > 0?
                    # No, we need to check for contradictions (unassigned == 0 and mono)
                    # and units (unassigned == 1 and other 4 same).
                    
                    # To avoid O(E) scan at every step, we can maintain a queue.
                    # But for this specific improvement, let's try a simple optimized scan first.
                    # If it's too slow, we'll add the queue.
                    
                    # Actually, a standard DPLL with UP usually maintains a queue of "touched" edges.
                    # Let's implement the queue version for robustness.
                    pass # Implemented in the actual code below

        # ...
    