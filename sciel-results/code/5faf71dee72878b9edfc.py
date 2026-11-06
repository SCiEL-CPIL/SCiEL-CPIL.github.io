    # Precompute next_use
    next_use = {v: float('inf') for v in vertices}
    # We need to iterate backwards?
    # For each vertex v, its next use is determined by its children.
    # Actually, for a vertex u, its next use is the earliest index of a child c of u that is > current index.
    # Since we are processing sequentially, we can maintain a queue of "pending uses".
    
    # Simpler:
    # For each vertex u, collect the indices of all its children.
    # children_indices[u] = [i for i, v in enumerate(vertices) if u in parents[v]]
    # Then during simulation at index i, the next use of u is the smallest index in children_indices[u] that is > i.
    
    # This is O(V * E) to precompute, which might be slow.
    # But V is likely small enough.
    
    # Let's optimize:
    # We can just use out_degree as a first pass.
    # If we have time, we can refine.
