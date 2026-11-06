    # Normalize parents and children to dicts
    parents = {}
    children = {}
    
    if parents_raw is not None:
        if isinstance(parents_raw, dict):
            parents = parents_raw
        elif isinstance(parents_raw, list):
            # Could be list of (child, parent) or list of lists
            # Heuristic: if elements are lists, treat as adjacency list (index = child)
            # If elements are tuples of 2, treat as edges
            if all(isinstance(item, (list, tuple)) for item in parents_raw):
                if all(len(item) == 2 for item in parents_raw):
                    # List of edges (u, v) where u->v? Or v is child?
                    # Usually (parent, child) or (child, parent).
                    # Let's assume (parent, child) for now, or check context.
                    # In pebbling, we need parents of each node.
                    for u, v in parents_raw:
                        parents.setdefault(v, []).append(u)
                else:
                    # List of lists: parents_raw[i] is parents of i
                    for i, p_list in enumerate(parents_raw):
                        parents[i] = p_list
            else:
                # Maybe just a list of parents for a single node? Unlikely for whole DAG.
                pass
    
    if children_raw is not None:
        if isinstance(children_raw, dict):
            children = children_raw
        elif isinstance(children_raw, list):
             # Similar logic
             if all(isinstance(item, (list, tuple)) for item in children_raw):
                if all(len(item) == 2 for item in children_raw):
                    for u, v in children_raw:
                        children.setdefault(u, []).append(v)
                else:
                    for i, c_list in enumerate(children_raw):
                        children[i] = c_list

    # If parents is empty but edges exist, build from edges
    if not parents and edges:
        for edge in edges:
            if len(edge) == 2:
                p, c = edge
                parents.setdefault(c, []).append(p)
                children.setdefault(p, []).append(c)
    
    # If children is empty, build from parents
    if not children:
        for p, c_list in parents.items():
            for c in c_list:
                children.setdefault(p, []).append(c)
    
    # Ensure all vertices are in parents/children dicts
    if vertices:
        for v in vertices:
            if v not in parents: parents[v] = []
            if v not in children: children[v] = []
