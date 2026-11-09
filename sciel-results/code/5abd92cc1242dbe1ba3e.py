    def check_connectivity(signs):
        """Check if the 2-lift is connected."""
        visited = np.zeros((n, 2), dtype=bool)
        queue = [(0, 0)]
        visited[0, 0] = True
        count = 1
        
        while queue:
            u, side = queue.pop(0)
            for idx, (u_e, v_e) in enumerate(edges):
                if u_e == u:
                    v = v_e
                    s = signs[idx]
                elif v_e == u:
                    v = u_e
                    s = signs[idx]
                else:
                    continue
                
                if s == 1:
                    new_side = side
                else:
                    new_side = 1 - side
                
                if not visited[v, new_side]:
                    visited[v, new_side] = True
                    count += 1
                    queue.append((v, new_side))
        
        return count == 2 * n
