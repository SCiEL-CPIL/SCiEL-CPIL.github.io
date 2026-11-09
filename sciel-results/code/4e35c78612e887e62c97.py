<<<<<<< SEARCH
    # Third phase: Final polish - try to add any remaining safe edges
    if time.time() < deadline - 2:
        edge_matrix = [[False] * 16 for _ in range(16)]
        for (u, v) in best_edges:
            edge_matrix[u][v] = True
            edge_matrix[v][u] = True
        
        all_pairs = list(itertools.combinations(range(16), 2))
        non_edges = [e for e in all_pairs if e not in best_edges]
        
        for e in non_edges:
            if time.time() > deadline - 2:
                break
            u, v = e
            if edge_matrix[u][v]:
                continue
            
            remaining = [x for x in range(16) if x != u and x != v]
            safe = True
            
            # Temporarily add edge
            edge_matrix[u][v] = True
            edge_matrix[v][u] = True
            best_edges.add((u, v))
            
            # Check for Q3
            for combo in itertools.combinations(remaining, 6):
                verts = [u, v] + list(combo)
                if contains_q3_fast(verts, edge_matrix):
                    safe = False
                    break
            
            if not safe:
                edge_matrix[u][v] = False
                edge_matrix[v][u] = False
                best_edges.remove((u, v))
=======
    # Third phase: Final polish - try to add any remaining safe edges
    if time.time() < deadline - 2:
        edge_matrix = [[False] * 16 for _ in range(16)]
        for (u, v) in best_edges:
            edge_matrix[u][v] = True
            edge_matrix[v][u] = True
        
        all_pairs = list(itertools.combinations(range(16), 2))
        non_edges = [e for e in all_pairs if e not in best_edges]
        
        # Sort by degree sum to try high-degree edges first
        degrees = [0] * 16
        for v in range(16):
            degrees[v] = sum(1 for x in range(16) if x != v and edge_matrix[v][x])
        non_edges.sort(key=lambda e: degrees[e[0]] + degrees[e[1]], reverse=True)
        
        for e in non_edges:
            if time.time() > deadline - 2:
                break
            u, v = e
            if edge_matrix[u][v]:
                continue
            
            remaining = [x for x in range(16) if x != u and x != v]
            safe = True
            
            # Temporarily add edge
            edge_matrix[u][v] = True
            edge_matrix[v][u] = True
            best_edges.add((u, v))
            
            # Check for Q3
            for combo in itertools.combinations(remaining, 6):
                verts = [u, v] + list(combo)
                if contains_q3_fast(verts, edge_matrix):
                    safe = False
                    break
            
            if not safe:
                edge_matrix[u][v] = False
                edge_matrix[v][u] = False
                best_edges.remove((u, v))
            else:
                # Update degrees
                degrees[u] += 1
                degrees[v] += 1
>>>>>>> REPLACE
