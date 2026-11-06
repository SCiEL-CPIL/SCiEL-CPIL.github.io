import time
import random

def _is_not_3_colorable(adj, n):
    """Check if graph is NOT 3-colorable using bitmask backtracking."""
    neigh = [0] * n
    for i in range(n):
        mask = 0
        row = adj[i]
        for j in range(n):
            if row[j]:
                mask |= (1 << j)
        neigh[i] = mask
    
    deg = [bin(neigh[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    pos_of = [0] * n
    for idx, v in enumerate(order):
        pos_of[v] = idx
        
    earlier_neigh = [0] * n
    for idx, v in enumerate(order):
        mask = 0
        nm = neigh[v]
        while nm:
            lsb = nm & (-nm)
            u = lsb.bit_length() - 1
            if pos_of[u] < idx:
                mask |= (1 << pos_of[u])
            nm ^= lsb
        earlier_neigh[v] = mask
        
    color_masks = [0, 0, 0]
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        forbidden = 0
        em = earlier_neigh[v]
        for c in range(3):
            if em & color_masks[c]:
                forbidden |= (1 << c)
        
        available = 7 & ~forbidden
        if available == 0:
            return False
            
        while available:
            c_bit = available & (-available)
            c = c_bit.bit_length() - 1
            color_masks[c] |= (1 << idx)
            if backtrack(idx + 1):
                return True
            color_masks[c] ^= (1 << idx)
            available ^= c_bit
        return False
    
    return not backtrack(0)

def _count_edges(adj, n):
    return sum(sum(row) for row in adj) // 2

def _get_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges

def _groetzsch_adj(n=16):
    """Create adjacency matrix for Grötzsch graph on vertices 0-10."""
    adj = [[0]*n for _ in range(n)]
    # Cycle on 0-4
    for i in range(5):
        u, v = i, (i+1) % 5
        adj[u][v] = adj[v][u] = 1
    # Vertices 5-9
    for i in range(5):
        u, w = i, (i+1) % 5
        adj[u][5+w] = adj[5+w][u] = 1
        adj[w][5+u] = adj[5+u][w] = 1
    # Vertex 10
    for u in range(5):
        adj[5+u][10] = adj[10][5+u] = 1
    return adj

def _find_max_independent_set(adj, n):
    """Find a maximum independent set using branch and bound."""
    best_set = []
    
    # Order vertices by degree descending for better pruning
    deg = [sum(adj[v]) for v in range(n)]
    vertices = sorted(range(n), key=lambda x: -deg[x])
    
    def backtrack(idx, current_set, candidates):
        nonlocal best_set
        if len(current_set) + len(candidates) <= len(best_set):
            return
        if not candidates:
            if len(current_set) > len(best_set):
                best_set = current_set[:]
            return
        
        # Try including the first candidate
        v = candidates[0]
        remaining = candidates[1:]
        
        # Include v
        new_candidates = [u for u in remaining if not adj[v][u]]
        current_set.append(v)
        backtrack(idx + 1, current_set, new_candidates)
        current_set.pop()
        
        # Exclude v
        backtrack(idx + 1, current_set, remaining)
    
    backtrack(0, [], vertices)
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 16
    adj = _groetzsch_adj(n)
    
    # Find maximum independent set in Grötzsch part (vertices 0-10)
    groetzsch_adj = [adj[i][:11] for i in range(11)]
    max_iso = _find_max_independent_set(groetzsch_adj, 11)
    
    # Connect new vertices 11-15 to the independent set
    for i in range(5):
        v_new = 11 + i
        for u in max_iso:
            adj[v_new][u] = 1
            adj[u][v_new] = 1
    
    best_adj = [row[:] for row in adj]
    best_edges = _count_edges(adj, n)
    
    # Precompute common neighbor masks
    def compute_cn(adj, n):
        cn = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                mask = 0
                for k in range(n):
                    if k != i and k != j and adj[i][k] and adj[j][k]:
                        mask |= (1 << k)
                cn[i][j] = mask
                cn[j][i] = mask
        return cn
    
    cn = compute_cn(adj, n)
    
    def try_add_edge(i, j):
        if adj[i][j]:
            return False
        if cn[i][j] != 0:
            return False
        
        adj[i][j] = 1
        adj[j][i] = 1
        
        old_cn = {}
        for k in range(n):
            if k != i and k != j:
                old_cn[(i,k)] = cn[i][k]
                old_cn[(j,k)] = cn[j][k]
                if adj[i][k]:
                    cn[i][k] |= (1 << j)
                    cn[k][i] |= (1 << j)
                if adj[j][k]:
                    cn[j][k] |= (1 << i)
                    cn[k][j] |= (1 << i)
        cn[i][j] = 0
        cn[j][i] = 0
        
        if _is_not_3_colorable(adj, n):
            return True
        else:
            adj[i][j] = 0
            adj[j][i] = 0
            for (a,b), val in old_cn.items():
                cn[a][b] = val
            for k in range(n):
                if k != i and k != j:
                    cn[k][i] = old_cn.get((i,k), cn[k][i])
                    cn[k][j] = old_cn.get((j,k), cn[k][j])
            return False
    
    # Local search
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
        random.shuffle(non_edges)
        
        improved = False
        for (i, j) in non_edges:
            if time.time() > deadline:
                break
            if try_add_edge(i, j):
                new_edges = _count_edges(adj, n)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = [row[:] for row in adj]
                improved = True
        
        if not improved and time.time() < deadline:
            edge_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
            random.shuffle(edge_list)
            
            for (ri, rj) in edge_list[:15]:
                if time.time() > deadline:
                    break
                
                # Remove edge
                adj[ri][rj] = 0
                adj[rj][ri] = 0
                for k in range(n):
                    if k != ri and k != rj:
                        if adj[ri][k]:
                            cn[ri][k] &= ~(1 << rj)
                            cn[k][ri] &= ~(1 << rj)
                        if adj[rj][k]:
                            cn[rj][k] &= ~(1 << ri)
                            cn[k][rj] &= ~(1 << ri)
                
                # Try to add a different edge
                for (ai, aj) in non_edges[:50]:
                    if time.time() > deadline:
                        break
                    if adj[ai][aj] or cn[ai][aj] != 0:
                        continue
                    
                    adj[ai][aj] = 1
                    adj[aj][ai] = 1
                    old_cn2 = {}
                    for k in range(n):
                        if k != ai and k != aj:
                            old_cn2[(ai,k)] = cn[ai][k]
                            old_cn2[(aj,k)] = cn[aj][k]
                            if adj[ai][k]:
                                cn[ai][k] |= (1 << aj)
                                cn[k][ai] |= (1 << aj)
                            if adj[aj][k]:
                                cn[aj][k] |= (1 << ai)
                                cn[k][aj] |= (1 << ai)
                    cn[ai][aj] = 0
                    cn[aj][ai] = 0
                    
                    if _is_not_3_colorable(adj, n):
                        # Keep the swap
                        new_edges = _count_edges(adj, n)
                        if new_edges > best_edges:
                            best_edges = new_edges
                            best_adj = [row[:] for row in adj]
                        improved = True
                        break
                    else:
                        # Revert add
                        adj[ai][aj] = 0
                        adj[aj][ai] = 0
                        for (a,b), val in old_cn2.items():
                            cn[a][b] = val
                        for k in range(n):
                            if k != ai and k != aj:
                                cn[k][ai] = old_cn2.get((ai,k), cn[k][ai])
                                cn[k][aj] = old_cn2.get((aj,k), cn[k][aj])
                
                # Re-add removed edge
                adj[ri][rj] = 1
                adj[rj][ri] = 1
                for k in range(n):
                    if k != ri and k != rj:
                        if adj[ri][k]:
                            cn[ri][k] |= (1 << rj)
                            cn[k][ri] |= (1 << rj)
                        if adj[rj][k]:
                            cn[rj][k] |= (1 << ri)
                            cn[k][rj] |= (1 << ri)
                
                if improved:
                    break

    return _get_edges(best_adj, n)
