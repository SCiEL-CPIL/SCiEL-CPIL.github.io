def construct():
    # Construct two disjoint K_{3,3} graphs
    # Vertices 0-5 for first K_{3,3}, 6-11 for second K_{3,3}
    
    edges = []
    
    # Helper to add edges for a K_{3,3} with parts (a, b, c) and (d, e, f)
    def add_k33(a, b, c, d, e, f):
        parts1 = [a, b, c]
        parts2 = [d, e, f]
        for u in parts1:
            for v in parts2:
                if u < v:
                    edges.append([u, v])
                else:
                    edges.append([v, u])

    add_k33(0, 1, 2, 3, 4, 5)
    add_k33(6, 7, 8, 9, 10, 11)
    
    # Sort edges for consistency
    edges.sort()
    
    return edges
