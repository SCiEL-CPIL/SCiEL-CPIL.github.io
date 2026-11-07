# Inside the while rem_edges loop:
            # ... find best_clique ...
            
            cliques.append(best_clique)
            
            # Update rem_edges
            edges_to_remove = set()
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    a, b = best_clique[i], best_clique[j]
                    e = (a, b) if a < b else (b, a)
                    if e in rem_edges:
                        edges_to_remove.add(e)
            rem_edges -= edges_to_remove
