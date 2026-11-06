    children = defaultdict(list)
    for v in vertices:
        for p in parents.get(v, []):
            children[p].append(v)
