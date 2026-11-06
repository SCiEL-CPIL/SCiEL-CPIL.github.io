    def try_extend(adj, n):
        masks = [[0]*n for _ in range(4)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    c = adj[i][j]
                    masks[c][i] |= (1 << j)
        
        assigned = [-1] * n
        conflict_masks = [0, 0, 0, 0]
        unassigned = set(range(n))
        nodes = [0]
        limit = 200000
        
        def bt():
            if not unassigned:
                return True
            nodes[0] += 1
            if nodes[0] > limit:
                return False
            
            # MRV: find u with min available colors
            best_u = -1
            best_count = 5
            best_avail = []
            for u in unassigned:
                avail = []
                for c in range(4):
                    if (masks[c][u] & conflict_masks[c]) == 0:
                        avail.append(c)
                if len(avail) < best_count:
                    best_count = len(avail)
                    best_u = u
                    best_avail = avail
                    if best_count == 0:
                        return False
                    if best_count == 1:
                        break
                        
            unassigned.remove(best_u)
            
            # Shuffle avail for randomness?
            random.shuffle(best_avail)
            
            for c in best_avail:
                assigned[best_u] = c
                conflict_masks[c] |= (1 << best_u)
                
                if bt():
                    return True
                    
                conflict_masks[c] &= ~(1 << best_u)
                assigned[best_u] = -1
                
            unassigned.add(best_u)
            return False

        if bt():
            new_adj = [row[:] for row in adj]
            for _ in range(n+1):
                new_adj.append([-1]*(n+1))
            for i in range(n):
                for j in range(n):
                    new_adj[i][j] = adj[i][j]
            for i in range(n):
                new_adj[n][i] = new_adj[i][n] = assigned[i]
            new_adj[n][n] = -1
            return new_adj
        return None
