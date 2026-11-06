def validate(pieces, edges):
    edge_set = set(tuple(sorted(e)) for e in edges)
    seen = set()
    for p in pieces:
        if len(p) == 2:
            e = tuple(sorted(p))
            if e in seen: return False
            seen.add(e)
        elif len(p) >= 3:
            # check distinct vertices
            if len(set(p)) != len(p): return False
            for i in range(len(p)):
                a, b = p[i], p[(i+1) % len(p)]
                e = tuple(sorted((a, b)))
                if e in seen: return False
                seen.add(e)
        else:
            return False
    return seen == edge_set
