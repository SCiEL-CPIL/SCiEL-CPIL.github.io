        active_cliques = list(clique_edge_sets)
        remaining_edges = set(edge_set)
        partition = []
        
        while remaining_edges and time.time() < deadline:
            best_idx = -1
            max_count = 0
            best_c_edges = None
            
            for i, (c, c_edges) in enumerate(active_cliques):
                # If c_edges is subset of already covered? 
                # We can check if c_edges & remaining_edges is empty.
                # If yes, remove from active.
                inter = c_edges & remaining_edges
                if not inter:
                    # Mark for removal
                    active_cliques[i] = None 
                    continue
                
                count = len(inter)
                if count > max_count:
                    max_count = count
                    best_idx = i
                    best_c_edges = inter
            
            # Clean up active_cliques
            if any(x is None for x in active_cliques):
                active_cliques = [x for x in active_cliques if x is not None]
            
            if best_idx == -1:
                # No clique covers any remaining edge?
                # Fallback
                e = next(iter(remaining_edges))
                partition.append([e[0], e[1]])
                remaining_edges.discard(e)
            else:
                c, _ = active_cliques[best_idx]
                partition.append(list(c))
                remaining_edges -= best_c_edges
                # Remove the used clique from active? 
                # It might still cover other edges? No, we removed best_c_edges.
                # But c_edges might have other edges? No, best_c_edges is the intersection.
                # The clique is "used" in the sense that we picked it.
                # We should remove it from active_cliques to avoid re-picking it?
                # If we pick it again, its intersection with remaining will be smaller or 0.
                # So we can remove it.
                active_cliques.pop(best_idx)
                
        # Handle remaining
        while remaining_edges:
            e = next(iter(remaining_edges))
            partition.append([e[0], e[1]])
            remaining_edges.discard(e)
