"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    n = P['n']
    # Build adjacency sets for quick lookup
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Remaining edges to cover
    remaining_edges = set()
    for u, v in P['edges']:
        remaining_edges.add((min(u,v), max(u,v)))
    
    bicliques = []
    
    while remaining_edges:
        if time.time() > deadline:
            break
        
        # Find a good biclique by trying pairs of vertices
        best_biclique = None
        best_count = 0
        
        # Get candidate vertices from remaining edges
        candidate_vertices = set()
        for e in remaining_edges:
            candidate_vertices.add(e[0])
            candidate_vertices.add(e[1])
        
        # Try all pairs of candidate vertices as a seed for the biclique
        cand_list = list(candidate_vertices)
        for i in range(len(cand_list)):
            if time.time() > deadline:
                break
            u = cand_list[i]
            for j in range(i+1, len(cand_list)):
                v = cand_list[j]
                # Check if u-v is a remaining edge
                key = (min(u,v), max(u,v))
                if key not in remaining_edges:
                    continue
                
                # Find common neighbors that form a valid biclique
                # Left side: vertices connected to all right-side vertices
                # Right side: vertices connected to all left-side vertices
                # Start with {u} on left, {v} on right, expand greedily
                
                left = {u}
                right = {v}
                
                # Try to expand: add vertices to left that connect to all in right
                # and vertices to right that connect to all in left
                changed = True
                while changed:
                    changed = False
                    # Try adding to left
                    for w in cand_list:
                        if w in left or w in right:
                            continue
                        if all(w in adj[x] for x in right):
                            # Check all edges from w to right are in remaining
                            if all((min(w,x), max(w,x)) in remaining_edges for x in right):
                                left.add(w)
                                changed = True
                    # Try adding to right
                    for w in cand_list:
                        if w in left or w in right:
                            continue
                        if all(w in adj[x] for x in left):
                            if all((min(w,x), max(w,x)) in remaining_edges for x in left):
                                right.add(w)
                                changed = True
                
                count = len(left) * len(right)
                if count > best_count:
                    best_count = count
                    best_biclique = (left, right)
        
        if best_biclique is None:
            # Fallback: take any remaining edge as a biclique
            e = next(iter(remaining_edges))
            best_biclique = ({e[0]}, {e[1]})
        
        left, right = best_biclique
        
        # Remove all edges between left and right from remaining
        for u in left:
            for v in right:
                key = (min(u,v), max(u,v))
                remaining_edges.discard(key)
        
        # Convert to lists of ints
        bicliques.append([sorted(int(x) for x in left), sorted(int(x) for x in right)])
    
    # If we timed out, fall back to covering remaining edges one by one
    if remaining_edges:
        for e in remaining_edges:
            bicliques.append([[int(e[0])], [int(e[1])]])
    
    return bicliques
# EVOLVE-BLOCK-END
