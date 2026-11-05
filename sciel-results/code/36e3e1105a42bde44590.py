<<<<<<< SEARCH
    vertices = dag['vertices']
    parents = dag['parents']
    inputs = set(dag['inputs'])
    outputs = set(dag['outputs'])
=======
    # Robustly extract graph components
    if 'vertices' in dag:
        vertices = dag['vertices']
    elif 'nodes' in dag:
        vertices = dag['nodes']
    else:
        # Derive vertices from parents, inputs, and outputs
        vertices_set = set()
        parents_dict = dag.get('parents', {})
        for v, ps in parents_dict.items():
            vertices_set.add(v)
            for p in ps:
                vertices_set.add(p)
        inputs_list = dag.get('inputs', [])
        outputs_list = dag.get('outputs', [])
        vertices_set.update(inputs_list)
        vertices_set.update(outputs_list)
        vertices = list(vertices_set)
    
    parents = dag.get('parents', {})
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
>>>>>>> REPLACE
