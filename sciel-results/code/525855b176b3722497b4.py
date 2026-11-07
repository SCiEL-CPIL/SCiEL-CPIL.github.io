import time

P = {'n': 18, 'edge_order': [[0, 1, 6], [4, 13, 14], [2, 3, 16], [9, 10, 17], [6, 14, 17], [11, 12, 16], [3, 10, 17], [10, 12, 14], [3, 9, 13], [0, 7, 13], [3, 14, 16], [1, 5, 8], [0, 9, 12], [9, 13, 17], [4, 11, 17], [4, 8, 13], [3, 14, 17], [3, 11, 15], [11, 13, 17], [8, 11, 12], [4, 9, 13], [2, 7, 9], [3, 6, 14], [0, 3, 5], [5, 7, 16], [2, 9, 16], [6, 7, 9], [1, 7, 8], [3, 10, 13], [3, 10, 14], [1, 10, 15], [2, 4, 14], [4, 11, 14], [6, 7, 10], [5, 6, 12], [3, 13, 17], [0, 5, 7], [5, 7, 11], [0, 2, 11], [9, 15, 16], [0, 2, 14], [5, 15, 17], [0, 3, 7], [10, 12, 15], [7, 9, 11], [4, 7, 17], [4, 11, 16], [3, 6, 17], [7, 11, 14], [1, 3, 8], [1, 10, 16], [3, 9, 16], [7, 8, 11], [7, 10, 14], [4, 10, 14], [3, 14, 15], [4, 14, 17], [4, 16, 17], [9, 12, 15], [0, 2, 4], [7, 9, 15], [10, 12, 17], [7, 8, 13], [8, 11, 15], [8, 16, 17], [5, 11, 13], [1, 11, 14], [0, 6, 10], [0, 8, 14], [5, 6, 16], [0, 6, 17], [1, 16, 17], [7, 11, 16], [2, 11, 16], [0, 1, 13], [6, 7, 14], [4, 7, 11], [6, 11, 14], [1, 7, 14], [1, 4, 8], [5, 9, 15], [8, 9, 15], [3, 11, 13], [2, 11, 14], [3, 12, 13], [2, 4, 8], [12, 14, 15], [1, 12, 17], [1, 14, 17], [6, 15, 16], [0, 11, 13], [2, 5, 15], [11, 12, 17], [5, 15, 16], [2, 6, 15], [5, 6, 15], [5, 8, 15], [0, 7, 9], [1, 4, 14], [9, 14, 15], [2, 4, 10], [3, 7, 15], [5, 7, 17], [1, 10, 14], [6, 8, 17], [1, 2, 15], [1, 14, 15], [2, 5, 17], [7, 9, 16], [10, 13, 17], [5, 9, 14], [0, 12, 13], [0, 13, 14], [7, 10, 12], [2, 11, 15], [4, 5, 10], [10, 16, 17], [4, 14, 16], [1, 8, 15], [5, 6, 14], [7, 11, 15], [3, 11, 12], [7, 9, 17], [4, 7, 14], [1, 5, 15], [3, 7, 12], [4, 5, 12], [3, 4, 15], [1, 6, 10], [1, 8, 10], [6, 10, 13], [9, 11, 17], [10, 14, 15], [2, 3, 5], [0, 5, 6], [4, 11, 12], [2, 6, 16], [2, 8, 10], [2, 3, 11], [9, 10, 11], [2, 9, 17], [0, 12, 16], [0, 6, 13], [1, 9, 13], [5, 6, 9], [4, 5, 17], [2, 7, 10], [2, 3, 7], [1, 7, 11], [3, 7, 17], [0, 2, 7], [3, 5, 10], [4, 7, 15], [1, 2, 11], [6, 8, 10], [2, 4, 6], [0, 2, 13], [2, 3, 17], [3, 7, 16], [3, 4, 10], [1, 6, 7], [6, 10, 12], [4, 6, 7], [7, 10, 17], [8, 13, 17], [1, 12, 15], [0, 8, 15], [2, 5, 7], [1, 3, 15], [2, 4, 11], [4, 6, 8], [9, 14, 16], [0, 13, 16], [0, 9, 16], [8, 10, 14], [3, 5, 12], [2, 9, 13], [2, 7, 12], [0, 1, 11], [1, 7, 13], [1, 4, 7], [0, 9, 17], [0, 1, 14], [8, 13, 14], [0, 2, 5], [1, 10, 12], [8, 12, 13], [1, 14, 16], [3, 8, 12], [5, 7, 14], [1, 5, 9], [5, 13, 16], [0, 1, 15], [0, 16, 17], [3, 5, 6], [5, 8, 11], [0, 3, 13], [0, 12, 14], [5, 6, 8], [5, 7, 12], [4, 5, 8], [0, 10, 11], [1, 8, 11], [1, 13, 16], [3, 12, 16], [3, 6, 8], [0, 2, 17], [9, 10, 12], [2, 5, 8], [1, 12, 14], [8, 9, 13], [0, 7, 8], [0, 14, 17], [4, 10, 13], [9, 12, 13], [5, 11, 15], [13, 14, 15], [1, 9, 10], [2, 7, 15], [6, 9, 10], [0, 12, 15], [12, 13, 16], [0, 9, 14], [6, 13, 15], [1, 4, 10], [4, 7, 16], [12, 15, 17], [2, 7, 14], [0, 4, 5], [3, 16, 17], [13, 15, 16], [5, 7, 10], [6, 7, 16], [6, 15, 17], [6, 12, 13], [1, 13, 15], [0, 4, 8], [7, 13, 15], [4, 9, 11], [6, 7, 13], [0, 8, 16], [5, 6, 11], [2, 4, 17], [3, 13, 14], [10, 11, 13], [2, 11, 12], [6, 7, 17], [4, 13, 15], [3, 5, 13], [0, 9, 15], [3, 13, 16], [4, 12, 15], [12, 16, 17], [2, 6, 12], [7, 8, 12], [0, 7, 17], [2, 10, 12], [8, 9, 16], [8, 10, 13], [6, 9, 16], [6, 13, 17], [1, 3, 13], [6, 11, 12], [6, 13, 14], [2, 4, 9], [0, 10, 15], [0, 14, 16], [1, 9, 17], [4, 8, 10], [3, 6, 13], [4, 6, 17], [1, 2, 8], [2, 10, 17], [2, 8, 16], [0, 7, 16], [0, 10, 17], [2, 14, 15], [6, 7, 11], [10, 11, 16], [6, 9, 11], [1, 8, 17], [1, 7, 9], [0, 6, 9], [10, 14, 17], [2, 5, 10], [4, 8, 9], [7, 8, 15], [2, 12, 17], [9, 10, 14], [8, 12, 17], [6, 7, 12], [5, 8, 14], [6, 9, 14], [5, 12, 14], [3, 15, 17], [10, 13, 16], [8, 14, 15], [7, 11, 17], [5, 10, 13], [5, 11, 14], [6, 12, 15], [2, 10, 15], [0, 5, 15], [12, 14, 17], [11, 13, 16], [2, 13, 16], [2, 9, 15], [0, 15, 17], [0, 6, 11], [2, 8, 13], [0, 1, 4], [4, 12, 13], [5, 9, 12], [0, 9, 10], [1, 9, 11], [7, 15, 16], [0, 1, 8], [5, 7, 9], [0, 1, 9], [11, 12, 14], [1, 8, 14], [6, 8, 12], [5, 8, 17], [3, 8, 15], [3, 5, 17], [2, 5, 11], [0, 3, 12], [5, 14, 16], [7, 14, 17], [6, 9, 12], [9, 10, 15], [7, 9, 13], [9, 11, 16], [3, 7, 11], [3, 9, 12], [3, 4, 7], [1, 5, 17], [0, 1, 3], [5, 10, 12], [2, 3, 8], [7, 13, 14], [0, 7, 12], [9, 13, 15], [3, 11, 14], [0, 4, 16], [12, 13, 15], [7, 9, 14], [2, 10, 16], [7, 9, 10], [3, 10, 12], [14, 15, 16], [5, 12, 16], [1, 3, 10], [4, 10, 15], [5, 6, 13], [1, 7, 10], [6, 9, 13], [2, 9, 11], [1, 7, 15], [1, 7, 16], [3, 10, 16], [1, 6, 13], [0, 9, 13], [5, 8, 13], [11, 13, 15], [2, 4, 15], [10, 11, 14], [13, 16, 17], [3, 5, 7], [0, 8, 13], [3, 8, 17], [2, 5, 6], [3, 4, 6], [0, 1, 10], [3, 8, 10], [0, 3, 8], [2, 13, 17], [0, 4, 15], [2, 7, 13], [3, 12, 17], [8, 10, 11], [4, 8, 12], [10, 12, 13], [11, 12, 15], [0, 4, 10], [8, 10, 17], [4, 5, 11], [0, 10, 16], [0, 6, 7], [3, 10, 11], [1, 15, 17], [3, 8, 13], [3, 6, 9], [4, 15, 17], [2, 4, 13], [4, 8, 11], [1, 10, 17], [0, 2, 12], [0, 7, 10], [0, 1, 16], [0, 7, 14], [4, 5, 6], [4, 5, 9], [2, 6, 10], [2, 6, 17], [0, 7, 15], [4, 5, 14], [3, 7, 10], [2, 16, 17], [5, 8, 12], [6, 11, 16], [6, 10, 15], [2, 15, 17], [4, 6, 14], [0, 10, 12], [10, 15, 17], [1, 6, 16], [0, 12, 17], [6, 11, 13], [1, 3, 9], [7, 12, 16], [3, 5, 14], [6, 10, 11], [3, 11, 16], [1, 4, 12], [7, 15, 17], [4, 9, 17], [7, 8, 17], [1, 8, 13], [3, 6, 12], [1, 13, 17], [2, 9, 10], [7, 9, 12], [1, 8, 9], [2, 8, 17], [0, 11, 14], [3, 6, 11], [2, 5, 12], [11, 15, 16], [1, 9, 12], [2, 7, 16], [5, 8, 10], [2, 6, 14], [9, 13, 14], [5, 9, 10], [2, 12, 14], [0, 8, 11], [8, 12, 16], [3, 4, 8], [7, 11, 13], [5, 14, 15], [2, 4, 16], [1, 2, 10], [11, 14, 15], [6, 8, 15], [10, 11, 15], [6, 8, 16], [11, 12, 13], [8, 11, 17], [11, 15, 17], [2, 3, 13], [1, 5, 11], [0, 5, 16], [4, 6, 15], [14, 16, 17], [0, 9, 11], [0, 3, 10], [0, 2, 3], [3, 10, 15], [10, 12, 16], [2, 11, 13], [4, 5, 7], [10, 15, 16], [8, 14, 17], [1, 13, 14], [2, 6, 8], [1, 2, 17], [2, 4, 5], [9, 13, 16], [4, 10, 12], [7, 14, 16], [5, 7, 13], [3, 4, 16], [2, 12, 15], [0, 3, 4], [6, 7, 15], [8, 11, 14], [5, 9, 13], [1, 2, 9], [0, 8, 12], [1, 5, 16], [0, 5, 14], [8, 10, 16], [1, 10, 13], [10, 11, 12], [4, 15, 16], [5, 13, 15], [1, 10, 11], [13, 14, 16], [2, 10, 14], [1, 5, 7], [6, 14, 16], [4, 6, 11], [2, 3, 12], [7, 14, 15], [7, 10, 16], [4, 12, 17], [0, 8, 10], [0, 7, 11], [1, 9, 14], [8, 13, 16], [4, 13, 16], [1, 8, 12], [1, 6, 14], [9, 11, 14], [8, 10, 12], [1, 3, 5], [0, 3, 14], [0, 4, 7], [2, 13, 14], [5, 11, 17], [1, 2, 16], [3, 7, 8], [2, 11, 17], [5, 13, 14], [3, 5, 11], [4, 8, 14], [1, 3, 17], [0, 14, 15], [5, 8, 9], [0, 15, 16], [0, 11, 12], [0, 2, 16], [2, 3, 9], [2, 5, 13], [8, 12, 15], [9, 10, 13], [7, 10, 13], [0, 6, 12], [3, 11, 17], [1, 5, 13], [3, 7, 9], [0, 1, 2], [1, 12, 13], [4, 9, 10], [1, 4, 6], [8, 9, 11], [3, 5, 15], [2, 14, 17], [2, 9, 14], [2, 8, 11], [5, 12, 13], [7, 10, 11], [9, 12, 16], [5, 12, 17], [0, 5, 10], [10, 14, 16], [5, 10, 15], [8, 9, 17], [1, 6, 15], [1, 8, 16], [6, 13, 16], [4, 5, 15], [8, 13, 15], [6, 10, 14], [3, 7, 13], [3, 4, 5], [0, 2, 9], [5, 6, 7], [8, 15, 17], [0, 11, 16], [5, 7, 15], [7, 12, 15], [6, 8, 9], [0, 1, 12], [6, 8, 11], [0, 1, 7], [0, 5, 8], [0, 13, 15], [0, 3, 11], [0, 4, 17], [9, 15, 17], [5, 11, 16], [9, 16, 17], [4, 11, 15], [8, 12, 14], [3, 5, 16], [3, 5, 8], [0, 6, 16], [2, 3, 14], [4, 6, 12], [4, 12, 16], [6, 8, 13], [5, 9, 17], [5, 9, 11], [3, 7, 14], [8, 9, 12], [0, 5, 17], [4, 7, 10], [3, 9, 14], [3, 9, 17], [1, 2, 4], [11, 14, 16], [0, 2, 10], [1, 3, 6], [2, 8, 15], [5, 8, 16], [3, 4, 9], [2, 8, 14], [2, 5, 9], [1, 4, 5], [4, 6, 13], [9, 12, 17], [7, 12, 17], [3, 6, 16], [10, 13, 14], [5, 10, 17], [6, 11, 17], [5, 12, 15], [4, 12, 14], [2, 7, 17], [3, 12, 14], [0, 3, 15], [1, 4, 16], [6, 11, 15], [0, 4, 9], [0, 10, 14], [0, 10, 13], [4, 9, 16], [0, 6, 8], [0, 4, 14], [4, 9, 14], [2, 14, 16], [13, 14, 17], [0, 3, 9], [2, 8, 12], [7, 16, 17], [6, 14, 15], [3, 13, 15], [4, 10, 16], [9, 11, 15], [4, 7, 9], [4, 10, 11], [1, 2, 5], [6, 7, 8], [0, 3, 17], [3, 9, 15], [0, 13, 17], [3, 12, 15], [5, 6, 17], [4, 5, 13], [6, 10, 17], [1, 5, 10], [1, 2, 6], [6, 12, 14], [11, 13, 14], [1, 11, 17], [0, 6, 14], [0, 2, 8], [1, 2, 7], [1, 6, 8], [1, 3, 12], [3, 4, 11], [7, 12, 13], [12, 13, 14], [4, 7, 13], [2, 5, 14], [1, 2, 3], [7, 8, 10], [0, 4, 12], [0, 4, 6], [2, 3, 4], [8, 14, 16], [1, 3, 16], [1, 15, 16], [3, 15, 16], [0, 5, 13], [1, 9, 16], [6, 9, 15], [5, 9, 16], [1, 5, 14], [7, 13, 16], [1, 4, 15], [10, 11, 17], [1, 3, 4], [2, 12, 16], [0, 11, 17], [4, 9, 12], [5, 13, 17], [2, 6, 13], [4, 9, 15], [5, 7, 8], [8, 11, 13], [1, 9, 15], [9, 10, 16], [0, 11, 15], [0, 2, 6], [0, 1, 17], [1, 6, 17], [9, 11, 13], [2, 9, 12], [4, 7, 8], [1, 2, 12], [1, 11, 16], [4, 11, 13], [1, 11, 12], [4, 8, 15], [2, 3, 15], [2, 15, 16], [2, 10, 13], [0, 5, 12], [13, 15, 17], [3, 4, 13], [5, 16, 17], [9, 11, 12], [2, 10, 11], [3, 4, 14], [6, 10, 16], [8, 10, 15], [4, 5, 16], [4, 6, 10], [2, 3, 6], [8, 9, 14], [9, 12, 14], [5, 10, 11], [12, 14, 16], [0, 8, 9], [6, 16, 17], [4, 6, 9], [15, 16, 17], [7, 12, 14], [3, 5, 9], [1, 7, 12], [0, 8, 17], [0, 5, 11], [12, 13, 17], [6, 12, 16], [5, 10, 14], [1, 3, 11], [3, 9, 11], [7, 8, 16], [1, 4, 13], [1, 6, 9], [6, 9, 17], [8, 15, 16], [1, 3, 7], [0, 1, 5], [6, 12, 17], [3, 4, 17], [8, 11, 16], [11, 16, 17], [3, 8, 9], [7, 13, 17], [1, 6, 12], [1, 2, 13], [4, 13, 17], [0, 4, 11], [2, 6, 9], [2, 6, 11], [5, 6, 10], [7, 8, 9], [2, 3, 10], [0, 4, 13], [3, 6, 10], [6, 8, 14], [2, 7, 11], [4, 7, 12], [10, 13, 15], [1, 5, 12], [5, 14, 17], [2, 4, 7], [1, 4, 17], [4, 14, 15], [3, 8, 16], [12, 15, 16], [7, 8, 14], [2, 4, 12], [9, 14, 17], [3, 8, 14], [1, 4, 11], [0, 5, 9], [4, 8, 17], [1, 11, 15], [1, 4, 9], [0, 3, 16], [4, 8, 16], [0, 3, 6], [3, 8, 11], [4, 10, 17], [1, 7, 17], [8, 9, 10], [3, 4, 12], [2, 8, 9], [5, 11, 12], [1, 12, 16], [3, 9, 10], [2, 7, 8], [1, 3, 14], [4, 6, 16], [2, 5, 16], [1, 6, 11], [2, 13, 15], [5, 10, 16], [14, 15, 17], [0, 2, 15], [3, 6, 15], [2, 12, 13], [11, 14, 17], [2, 6, 7], [1, 2, 14], [1, 5, 6], [7, 10, 15], [7, 11, 12], [0, 6, 15], [3, 6, 7], [1, 11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    edge_order = P['edge_order']
    n = P['n']
    full_mask = (1 << n) - 1
    
    # 1. Preprocess: Get unique triples with their minimum 1-based index
    triple_to_index = {}
    for i, tri in enumerate(edge_order):
        key = frozenset(tri)
        if key not in triple_to_index:
            triple_to_index[key] = i + 1
            
    # List of (index, mask, tuple_of_vertices)
    unique_triples = []
    for tri, idx in triple_to_index.items():
        mask = 0
        for v in tri:
            mask |= (1 << v)
        unique_triples.append((idx, mask, tuple(tri)))
        
    # Sort by index
    unique_triples.sort(key=lambda x: x[0])
    
    # Build adjacency: for each vertex, which triples (by index in unique_triples) contain it?
    # We need to check triples by index limit, so we will filter on the fly or precompute.
    # Precomputing a list of triple indices for each vertex is useful for backtracking.
    # vertex_to_triple_indices[v] = list of indices i such that v is in unique_triples[i]
    vertex_to_triple_indices = {v: [] for v in range(n)}
    for i, (idx, mask, tri) in enumerate(unique_triples):
        for v in tri:
            vertex_to_triple_indices[v].append(i)
            
    # 2. Feasibility Check Function
    def can_match(limit_index, record_path=False):
        """
        Checks if a perfect matching exists using only triples with index <= limit_index.
        If record_path is True, it returns the list of triple indices used.
        Otherwise, it returns True/False.
        """
        # Filter triples to those with index <= limit_index
        # To speed up, we can just iterate all and check index, or pre-filter.
        # Since limit_index changes in binary search, filtering on the fly in the search
        # or passing a pre-filtered list is better.
        
        # Let's pre-filter the list of available triple indices for this limit
        # available_trip_indices = [i for i, (idx, mask, tri) in enumerate(unique_triples) if idx <= limit_index]
        # This is O(M) per check, which is fine.
        
        # However, for backtracking, we need to pick a vertex and find available triples.
        # Let's use a recursive backtracker.
        
        available_indices = set()
        for i, (idx, mask, tri) in enumerate(unique_triples):
            if idx <= limit_index:
                available_indices.add(i)
        
        if not available_indices:
            return [] if record_path else False
            
        used_mask = 0
        path = []
        
        # Heuristic: Pick the vertex with the fewest available options?
        # Or just pick the smallest uncovered vertex.
        # Given n=18, picking smallest uncovered vertex is usually sufficient.
        
        def backtrack(used_mask, path, remaining_count):
            if time.time() > deadline:
                return False # Timeout, treat as failure for now (should be handled by caller)
                
            if remaining_count == 0:
                if used_mask == full_mask:
                    return True
                else:
                    return False
                    
            if used_mask == full_mask:
                return remaining_count == 0
                
            # Find the first uncovered vertex
            # This is O(n)
            target_v = -1
            for v in range(n):
                if not (used_mask & (1 << v)):
                    target_v = v
                    break
            
            if target_v == -1:
                return remaining_count == 0
                
            # Find all available triples containing target_v
            candidates = []
            for tri_idx in vertex_to_triple_indices[target_v]:
                if tri_idx in available_indices:
                    idx, mask, tri = unique_triples[tri_idx]
                    # Check if disjoint
                    if (mask & used_mask) == 0:
                        candidates.append(tri_idx)
                        
            if not candidates:
                return False
                
            # Optional: Sort candidates by something? E.g., index?
            # Not strictly necessary for correctness, but might affect speed.
            
            for tri_idx in candidates:
                idx, mask, tri = unique_triples[tri_idx]
                new_mask = used_mask | mask
                path.append(tri_idx)
                if backtrack(new_mask, path, remaining_count - 1):
                    return True
                path.pop()
                
            return False
            
        if record_path:
            # We need to return the path if successful
            # The above function returns bool. We need to modify it or use a wrapper.
            # Let's use a mutable container for the path.
            final_path = []
            
            def backtrack_rec(used_mask, remaining_count):
                if time.time() > deadline:
                    return False
                if remaining_count == 0:
                    return used_mask == full_mask
                if used_mask == full_mask:
                    return remaining_count == 0
                
                target_v = -1
                for v in range(n):
                    if not (used_mask & (1 << v)):
                        target_v = v
                        break
                if target_v == -1:
                    return remaining_count == 0
                
                candidates = []
                for tri_idx in vertex_to_triple_indices[target_v]:
                    if tri_idx in available_indices:
                        idx, mask, tri = unique_triples[tri_idx]
                        if (mask & used_mask) == 0:
                            candidates.append(tri_idx)
                            
                if not candidates:
                    return False
                    
                for tri_idx in candidates:
                    idx, mask, tri = unique_triples[tri_idx]
                    new_mask = used_mask | mask
                    final_path.append(tri_idx)
                    if backtrack_rec(new_mask, remaining_count - 1):
                        return True
                    final_path.pop()
                return False
                
            if backtrack_rec(0, 6):
                return final_path
            else:
                return []
        else:
            return backtrack(0, [], 6)

    # 3. Binary Search for the minimum index
    # The indices in unique_triples are not contiguous 1..M, they are the original indices.
    # But we sorted unique_triples by index.
    # So unique_triples[i][0] is the index.
    # We want to find the smallest K such that can_match(K) is True.
    
    min_possible = unique_triples[0][0]
    max_possible = unique_triples[-1][0]
    
    best_K = -1
    best_path = []
    
    # Binary search over the sorted list of unique indices
    # The set of possible K values is {unique_triples[i][0] for i in range(len(unique_triples))}
    # We can binary search on the range of indices directly.
    
    lo = min_possible
    hi = max_possible
    found = False
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        # Check feasibility
        # Note: can_match uses "index <= mid".
        # This is correct.
        
        # Optimization: If we just want to find the min index, we don't need the path in the binary search steps.
        if can_match(mid, record_path=False):
            best_K = mid
            found = True
            hi = mid - 1
        else:
            lo = mid + 1
            
    if found:
        # Retrieve the path for the best_K
        # We need to run the search again to get the actual triples
        path_indices = can_match(best_K, record_path=True)
        
        if path_indices:
            result = []
            for tri_idx in path_indices:
                _, _, tri = unique_triples[tri_idx]
                result.append([int(v) for v in tri])
            return result
        else:
            # Should not happen if can_match returned True
            return [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14],[15,16,17]]
    else:
        # Fallback if no solution found (should not happen for valid input)
        return [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14],[15,16,17]]
