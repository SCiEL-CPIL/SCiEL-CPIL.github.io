    def try_reduce_colors(edges, adj, color):
        """Try to reduce number of colors by reassigning edges."""
        n_colors = max(color.values()) + 1 if color else 0
        if n_colors == 0:
            return color, n_colors
        
        improved = True
        while improved:
            improved = False
            # Try to eliminate each color from highest to lowest
            for target_color in range(n_colors - 1, -1, -1):
                edges_with_color = [e for e, c in color.items() if c == target_color]
                if not edges_with_color:
                    continue
                
                new_color = dict(color)
                success = True
                for e in edges_with_color:
                    u, v = e
                    near = {u, v}
                    near.update(adj[u])
                    near.update(adj[v])
                    assigned = False
                    for c in range(target_color):
                        conflict = False
                        for f, fc in new_color.items():
                            if fc != c:
                                continue
                            fu, fv = f
                            if fu in near or fv in near:
                                conflict = True
                                break
                        if not conflict:
                            new_color[e] = c
                            assigned = True
                            break
                    if not assigned:
                        success = False
                        break
                
                if success:
                    color = new_color
                    n_colors -= 1
                    improved = True
                    break # Restart loop with new n_colors
        
        return color, n_colors
