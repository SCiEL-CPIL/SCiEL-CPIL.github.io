<<<<<<< SEARCH
    # Determine structure of dag
    # Assuming dag is a list of lists, where dag[i] is the list of parents of vertex i
    # Or a dict. Let's handle both.
    if isinstance(dag, dict):
        vertices = list(dag.keys())
        parents = {v: dag[v] for v in vertices}
    elif isinstance(dag, list):
        vertices = range(len(dag))
        parents = {i: dag[i] for i in vertices}
    else:
        # Fallback
        return []
        
    num_vertices = len(vertices)
=======
    # Determine structure of dag
    # Handles:
    # 1. dict: {vertex: parents}
    # 2. list of lists: dag[i] = [parent1, parent2, ...]
    # 3. list of records: dag[i] = [id, [parent1, parent2, ...]]
    
    if isinstance(dag, dict):
        vertices = list(dag.keys())
        parents = {v: dag[v] for v in vertices}
    elif isinstance(dag, list):
        if not dag:
            return []
        
        first = dag[0]
        # Heuristic: Check if it's a record [id, parents]
        if isinstance(first, (list, tuple)) and len(first) >= 2 and isinstance(first[1], (list, tuple)):
            # Assume format [id, parents]
            vertices = range(len(dag))
            parents = {i: dag[i][1] for i in vertices}
        else:
            # Assume format [parents]
            vertices = range(len(dag))
            parents = {i: dag[i] for i in vertices}
    else:
        return []
        
    num_vertices = len(vertices)
>>>>>>> REPLACE

