<<<<<<< SEARCH
    # If inputs is still empty, derive from parents
    if not inputs:
        all_vertices = set(parents.keys())
        for v, ps in parents.items():
            for p in ps:
                all_vertices.add(p)
        # Inputs are vertices with no parents
        inputs = [v for v in all_vertices if v not in parents or not parents.get(v)]
        
    # If outputs is still empty, maybe all vertices are outputs? Or vertices with no children?
    # Let's assume if outputs is empty, we need to infer.
    # But without children map, we can't easily infer "no children" unless we build it.
    # Let's build children map to help infer outputs if needed.
    if not outputs:
        # Build children map
        children_map = {v: [] for v in parents}
        for v, ps in parents.items():
            for p in ps:
                if p not in children_map:
                    children_map[p] = []
                children_map[p].append(v)
        # Vertices with no children are candidates for outputs
        # But inputs might also have no children if graph is disconnected?
        # Usually outputs are the "sinks".
        candidates = [v for v in children_map if not children_map[v]]
        # Filter out inputs? No, an input can be an output if it's a trivial graph?
        # Let's assume all sinks are outputs.
        outputs = candidates
=======
    # If inputs is still empty, derive from parents
    if not inputs:
        all_vertices = set(parents.keys())
        for v, ps in parents.items():
            for p in ps:
                all_vertices.add(p)
        # Inputs are vertices with no parents
        inputs = [v for v in all_vertices if v not in parents or not parents.get(v)]
        
    # If outputs is still empty, infer as sinks (vertices with no children)
    if not outputs:
        all_vertices = set(parents.keys())
        for v, ps in parents.items():
            for p in ps:
                all_vertices.add(p)
        
        children_map = {v: [] for v in all_vertices}
        for v, ps in parents.items():
            for p in ps:
                if p in children_map:
                    children_map[p].append(v)
        
        # Sinks are vertices with no children
        outputs = [v for v in all_vertices if not children_map[v]]
        
        # If still empty (e.g. empty graph), leave as empty
        if not outputs and all_vertices:
            # Fallback: maybe all vertices are outputs?
            # Or maybe the last vertex in topological order?
            # For safety, let's assume all vertices that are not inputs are outputs?
            # No, that's too broad.
            # If the graph is a single node, it is both input and output.
            # If children_map is empty for all, all are sinks.
            # If we have a chain A->B, A is input, B is output.
            # children_map: A: [B], B: []. Sinks: [B]. Correct.
            # If we have A->B, A->C. Sinks: [B, C]. Correct.
            # If we have no edges, all are sinks. All are inputs too.
            # This seems reasonable.
            pass
>>>>>>> REPLACE

This ensures `outputs` is populated.

One final check: The `children` variable used later in the code is built from `parents`.
