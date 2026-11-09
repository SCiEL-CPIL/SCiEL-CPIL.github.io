if isinstance(dag, list):
    dag_dict = {}
    for item in dag:
        # Handle case where item is [node, parents] or (node, parents)
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            node = normalize_node(item[0])
            parents = [normalize_node(p) for p in item[1]]
            dag_dict[node] = parents
        else:
            # Fallback
            node = normalize_node(item)
            dag_dict[node] = []
