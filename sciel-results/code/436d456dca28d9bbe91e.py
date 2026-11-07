import time

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = P['n']
    adj = [set() for _ in range(n)]
    edges_set = set()
    for u, v in P['edges']:
        if u < v:
            edges_set.add((u, v))
        else:
            edges_set.add((v, u))
        adj[u].add(v)
        adj[v].add(u)

    remaining_edges = set(edges_set)
    bicliques = []

    while remaining_edges:
        if time.time() > deadline:
            for u, v in remaining_edges:
                bicliques.append([[u], [v]])
            break

        # Select top K candidate edges
        # Sort by degree sum (static)
        # To save time, we can just pick a few. 
        # Let's pick the 5 edges with highest degree sum.
        candidates = sorted(remaining_edges, key=lambda e: len(adj[e[0]]) + len(adj[e[1]]), reverse=True)
        top_candidates = candidates[:5]
        
        best_biclique = None
        best_covered = 0

        for u0, v0 in top_candidates:
            # Greedy expansion
            L = {u0}
            R = {v0}
            
            changed = True
            while changed:
                changed = False
                
                # Try to add to L
                if R:
                    first_r = next(iter(R))
                    common_R = set(adj[first_r])
                    for r in R:
                        if r != first_r:
                            common_R &= adj[r]
                else:
                    common_R = set(range(n))
                
                cand_L = common_R - L
                if cand_L:
                    # Pick the one with highest degree to L? Or just any?
                    # Just pick one
                    w = next(iter(cand_L))
                    L.add(w)
                    changed = True
                    continue

                # Try to add to R
                if L:
                    first_l = next(iter(L))
                    common_L = set(adj[first_l])
                    for l in L:
                        if l != first_l:
                            common_L &= adj[l]
                else:
                    common_L = set(range(n))
                
                cand_R = common_L - R
                if cand_R:
                    x = next(iter(cand_R))
                    R.add(x)
                    changed = True

            # Calculate coverage
            covered = 0
            for u in L:
                for v in R:
                    if u < v:
                        if (u, v) in remaining_edges:
                            covered += 1
                    else:
                        if (v, u) in remaining_edges:
                            covered += 1

            if covered > best_covered:
                best_covered = covered
                best_biclique = (L, R)

        if best_biclique:
            L, R = best_biclique
            bicliques.append([list(L), list(R)])
            for u in L:
                for v in R:
                    if u < v:
                        remaining_edges.discard((u, v))
                    else:
                        remaining_edges.discard((v, u))
        else:
            break

    return bicliques
# EVOLVE-BLOCK-END
