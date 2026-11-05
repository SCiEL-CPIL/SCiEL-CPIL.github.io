import itertools

def construct():
    n = 20
    A = list(range(5))
    B = list(range(5, 20))
    
    # Initialize neighbors
    neighbors = {a: set(B) for a in A}
    
    # Helper to find a K3,3
    def find_k33():
        for triple_a in itertools.combinations(A, 3):
            common = neighbors[triple_a[0]] & neighbors[triple_a[1]] & neighbors[triple_a[2]]
            if len(common) >= 3:
                triple_b = tuple(list(common)[:3])
                return (triple_a, triple_b)
        return None

    # Helper to count K3,3s containing a specific edge (a, b)
    def count_k33_for_edge(a, b):
        count = 0
        # We need 2 other vertices in A
        other_a = [x for x in A if x != a]
        for pair_a in itertools.combinations(other_a, 2):
            triple_a = (a, pair_a[0], pair_a[1])
            # Common neighbors of triple_a
            common = neighbors[triple_a[0]] & neighbors[triple_a[1]] & neighbors[triple_a[2]]
            if b in common and len(common) >= 3:
                # We need to choose 2 other neighbors from common excluding b
                others = common - {b}
                if len(others) >= 2:
                    count += len(list(itertools.combinations(others, 2)))
        return count

    # Greedy removal loop
    while True:
        k33 = find_k33()
        if k33 is None:
            break
        triple_a, triple_b = k33
        
        # Edges in this K3,3
        k33_edges = [(a, b) for a in triple_a for b in triple_b]
        
        max_count = -1
        edge_to_remove = None
        for a, b in k33_edges:
            c = count_k33_for_edge(a, b)
            if c > max_count:
                max_count = c
                edge_to_remove = (a, b)
        
        if edge_to_remove:
            a, b = edge_to_remove
            neighbors[a].remove(b)
        else:
            break

    # Construct edge list
    edges = []
    for a in A:
        for b in neighbors[a]:
            edges.append([a, b])
            
    return edges
