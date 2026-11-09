"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    n = 20
    
    def check_add_bitmask(adj_mask, u, v, c, adj_color):
        """
        Check if adding (u,v) with color c creates any non-rainbow C4.
        adj_mask: list of int bitmasks. adj_mask[i] has bit j set if (i,j) is an edge.
        adj_color: 2D list of colors.
        """
        # C4: u-v-w-x-u
        # w must be a neighbor of v (existing)
        mask_v = adj_mask[v]
        
        # We need to iterate w in N(v)
        # Optimization: if N(v) is small, this is fast.
        m_v = mask_v
        while m_v:
            lsb = m_v & -m_v
            w = lsb.bit_length() - 1
            m_v ^= lsb
            
            if w == u:
                continue
            
            # x must be in N(w) and N(u)
            mask_x = adj_mask[w] & adj_mask[u]
            
            # Exclude u, v, w from mask_x
            # u is in mask_x if (u,w) exists.
            # v is in mask_x if (v,w) exists. (It does, since w in N(v))
            # w is in mask_x if (w,w) exists? No. But w might be in N(u) and N(w) is not self-loop.
            # Wait, mask_x = N(w) \cap N(u).
            # w is not in N(w).
            # So w is not in mask_x.
            # u is in N(w) if (u,w) exists. u is in N(u)? No.
            # So u is in mask_x iff (u,w) exists.
            # v is in N(w) if (v,w) exists. v is in N(u) if (v,u) exists.
            # We are adding (u,v), so (v,u) is NOT in the graph yet.
            # So v is NOT in N(u).
            # So v is NOT in mask_x.
            # So we only need to exclude u.
            
            mask_x &= ~(1 << u)
            
            m_x = mask_x
            while m_x:
                lsb_x = m_x & -m_x
                x = lsb_x.bit_length() - 1
                m_x ^= lsb_x
                
                # Check colors of u-v (c), v-w, w-x, x-u
                c_vw = adj_color[v][w]
                c_wx = adj_color[w][x]
                c_xu = adj_color[x][u]
                
                if len({c, c_vw, c_wx, c_xu}) < 4:
                    return False
        return True

    start_time = time.time()
    time_limit = 115.0
    
    best_edges = 0
    best_adj_color = [[-1 for _ in range(n)] for _ in range(n)]
    best_adj_mask = [0] * n
    
    num_restarts = 2000 # Can do many more now
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        adj_color = [[-1 for _ in range(n)] for _ in range(n)]
        adj_mask = [0] * n
        
        edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() - start_time > time_limit:
                break
            
            # Try colors
            # Random order
            colors = list(range(20))
            random.shuffle(colors)
            
            added = False
            for c in colors:
                if check_add_bitmask(adj_mask, u, v, c, adj_color):
                    adj_color[u][v] = c
                    adj_color[v][u] = c
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    added = True
                    break
            
        # Count edges
        current_edges = sum(bin(mask).count("1") for mask in adj_mask) // 2
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj_color = [row[:] for row in adj_color]
            best_adj_mask = adj_mask[:]
            
    coloured_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj_color[u][v] != -1:
                coloured_edges.append([u, v, best_adj_color[u][v]])
    
    return coloured_edges
# EVOLVE-BLOCK-END
