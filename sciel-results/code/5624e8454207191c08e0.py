<<<<<<< SEARCH
    from itertools import combinations
    n = 9
    edges = [list(e) for e in combinations(range(n), 5)]
    
    return {'n': n, 'edges': edges}
=======
    n = 11
    d = [0, 1, 2, 4, 7]
    edges = []
    for i in range(n):
        edge = [(i + x) % n for x in d]
        edges.append(sorted(edge))
    
    return {'n': n, 'edges': edges}
>>>>>>> REPLACE
