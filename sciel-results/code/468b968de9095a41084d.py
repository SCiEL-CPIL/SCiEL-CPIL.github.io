    if isinstance(dag, dict):
        # Convert keys and values to int if possible
        new_parents_dict = {}
        for k, v in dag.items():
            try:
                int_k = int(k)
            except (ValueError, TypeError):
                continue # or raise
            int_v = []
            for item in v:
                try:
                    int_v.append(int(item))
                except (ValueError, TypeError):
                    pass # or keep as is? The problem says "vertex", usually int.
            new_parents_dict[int_k] = int_v
        parents_dict = new_parents_dict
        n = max(parents_dict.keys()) + 1 if parents_dict else 0
    elif isinstance(dag, (list, tuple)):
        if len(dag) > 0 and isinstance(dag[0], (list, tuple)):
            parents_dict = {}
            for i, item in enumerate(dag):
                int_item = []
                for p in item:
                    try:
                        int_item.append(int(p))
                    except (ValueError, TypeError):
                        pass
                parents_dict[i] = int_item
            n = len(dag)
        else:
            # Assume list of edges
            parents_dict = {}
            for edge in dag:
                if len(edge) == 2:
                    try:
                        p = int(edge[0])
                        c = int(edge[1])
                    except (ValueError, TypeError):
                        continue
                    if c not in parents_dict:
                        parents_dict[c] = []
                    parents_dict[c].append(p)
            n = max(parents_dict.keys()) + 1 if parents_dict else 0
    else:
        return []
