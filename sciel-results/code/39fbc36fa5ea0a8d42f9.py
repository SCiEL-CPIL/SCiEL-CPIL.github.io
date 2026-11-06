import time
import random
from itertools import combinations

def _is_not_3_colorable(adj, n):
    """
    Check if graph is NOT 3-colorable.
    Returns True if not 3-colorable (i.e., chromatic number >= 4).
    Uses bitmask backtracking with ordering by degree.
    """
    # Build neighbor masks
    neigh = [0] * n
    for i in range(n):
        mask = 0
        row = adj[i]
        for j in range(n):
            if row[j]:
                mask |= (1 << j)
        neigh[i] = mask
    
    # Order by degree descending to prune faster
    deg = [bin(neigh[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    # Map vertex to position in order
    pos_of = [0] * n
    for idx, v in enumerate(order):
        pos_of[v] = idx
        
    # Precompute earlier neighbors mask for each vertex in the order
    # earlier_neigh[v] is a bitmask of positions of neighbors of v that come before v in 'order'
    earlier_neigh = [0] * n
    for idx, v in enumerate(order):
        mask = 0
        # Neighbors of v that have position < idx
        # We can iterate through neighbors of v
        nm = neigh[v]
        while nm:
            lsb = nm & (-nm)
            u = lsb.bit_length() - 1
            if pos_of[u] < idx:
                mask |= (1 << pos_of[u])
            nm ^= lsb
        earlier_neigh[v] = mask
        
    # color_masks[c] has bit idx set if order[idx] has color c
    color_masks = [0, 0, 0]
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        # Determine forbidden colors based on earlier neighbors
        forbidden = 0
        em = earlier_neigh[v]
        for c in range(3):
            if em & color_masks[c]:
                forbidden |= (1 << c)
        
        available = 7 & ~forbidden
        if available == 0:
            return False
            
        # Try available colors
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
    """
    Create adjacency matrix for Grötzsch graph on vertices 0-10.
    Vertices 11-15 are initially isolated.
    """
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

def _find_independent_set(adj, n, size):
    """Find an independent set of given size. Returns list of vertices or None."""
    # Simple backtracking
    vertices = list(range(n))
    # Order by degree ascending (low degree vertices are more likely to be in independent set)
    deg = [sum(adj[v]) for v in range(n)]
    vertices.sort(key=lambda x: deg[x])
    
    def backtrack(idx, current_set):
        if len(current_set) == size:
            return current_set
        if idx == len(vertices):
            return None
        
        v = vertices[idx]
        # Option 1: Include v
        if all(not adj[v][u] for u in current_set):
            res = backtrack(idx + 1, current_set + [v])
            if res:
                return res
        # Option 2: Exclude v
        res = backtrack(idx + 1, current_set)
        if res:
            return res
        return None

    return backtrack(0, [])

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    n = 16
    adj = _groetzsch_adj(n)
    
    # Find an independent set of size 5 in the Grötzsch part (0-10)
    iso_set = _find_independent_set(adj, 11, 5)
    if not iso_set:
        iso_set = [0, 2, 5, 7, 10]
    
    # Connect new vertices 11-15 to the independent set
    for i in range(5):
        v_new = 11 + i
        for u in iso_set:
            adj[v_new][u] = 1
            adj[u][v_new] = 1
    
    best_adj = [row[:] for row in adj]
    best_edges = _count_edges(adj, n)
    
    # Precompute common neighbor masks for triangle checking
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
    
    # Local Search
    def try_add_edge(i, j):
        """Tries to add edge (i,j). Returns True if added and valid."""
        if adj[i][j]:
            return False
        if cn[i][j] != 0:
            return False
        
        # Tentatively add
        adj[i][j] = 1
        adj[j][i] = 1
        
        # Update CN
        old_cn_i = [cn[i][k] for k in range(n)]
        old_cn_j = [cn[j][k] for k in range(n)]
        for k in range(n):
            if k != i and k != j:
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
            # Revert
            adj[i][j] = 0
            adj[j][i] = 0
            for k in range(n):
                if k != i and k != j:
                    if adj[i][k]:
                        cn[i][k] = old_cn_i[k]
                        cn[k][i] = old_cn_i[k]
                    if adj[j][k]:
                        cn[j][k] = old_cn_j[k]
                        cn[k][j] = old_cn_j[k]
            return False
    
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    non_edges.append((i, j))
        random.shuffle(non_edges)
        
        improved = False
        
        # Phase 1: Add edges
        for (i, j) in non_edges:
            if time.time() > deadline:
                break
            if try_add_edge(i, j):
                new_edges = _count_edges(adj, n)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = [row[:] for row in adj]
                improved = True
        
        # Phase 2: Swaps (Remove one, Add one or more)
        if not improved and time.time() < deadline:
            edge_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edge_list.append((i, j))
            random.shuffle(edge_list)
            
            for (ri, rj) in edge_list[:20]:
                if time.time() > deadline:
                    break
                
                # Remove (ri, rj)
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
                cn[ri][rj] = 0
                cn[rj][ri] = 0
                
                added = False
                for (ai, aj) in non_edges:
                    if time.time() > deadline:
                        break
                    if adj[ai][aj]:
                        continue
                    if cn[ai][aj] != 0:
                        continue
                    
                    # Tentative add
                    adj[ai][aj] = 1
                    adj[aj][ai] = 1
                    old_cn_ai = [cn[ai][k] for k in range(n)]
                    old_cn_aj = [cn[aj][k] for k in range(n)]
                    for k in range(n):
                        if k != ai and k != aj:
                            if adj[ai][k]:
                                cn[ai][k] |= (1 << aj)
                                cn[k][ai] |= (1 << aj)
                            if adj[aj][k]:
                                cn[aj][k] |= (1 << ai)
                                cn[k][aj] |= (1 << ai)
                    cn[ai][aj] = 0
                    cn[aj][ai] = 0
                    
                    if _is_not_3_colorable(adj, n):
                        added = True
                        break
                    else:
                        # Revert add
                        adj[ai][aj] = 0
                        adj[aj][ai] = 0
                        for k in range(n):
                            if k != ai and k != aj:
                                if adj[ai][k]:
                                    cn[ai][k] = old_cn_ai[k]
                                    cn[k][ai] = old_cn_ai[k]
                                if adj[aj][k]:
                                    cn[aj][k] = old_cn_aj[k]
                                    cn[k][aj] = old_cn_aj[k]
                
                # Re-add removed edge (ri, rj)
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
                
                if added:
                    new_edges = _count_edges(adj, n)
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = [row[:] for row in adj]
                    improved = True

    return _get_edges(best_adj, n)
