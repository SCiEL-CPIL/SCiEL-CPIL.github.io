    def greedy_coloring(P):
        if P == 0:
            return []
        
        # Get list of vertices in P
        p_list = []
        temp_P = P
        while temp_P:
            v = (temp_P & -temp_P).bit_length() - 1
            p_list.append(v)
            temp_P ^= (1 << v)
            
        # Sort by global order index (precomputed)
        # order_index[v] is the position of v in the 'order' list
        p_list.sort(key=lambda x: order_index[x])
        
        colors = {}
        color_count = 0
        
        for v in p_list:
            used_colors = set()
            n_mask = adj[v] & P
            while n_mask:
                u = (n_mask & -n_mask).bit_length() - 1
                n_mask ^= (1 << u)
                if u in colors:
                    used_colors.add(colors[u])
            
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            if c + 1 > color_count:
                color_count = c + 1
        
        colored_list = [(v, colors[v]) for v in colors]
        colored_list.sort(key=lambda x: x[1], reverse=True)
        return colored_list
