import time
import random
from itertools import combinations

def _is_not_3_colorable(adj, n):
    """Check if graph is NOT 3-colorable using bitmask backtracking."""
    # Build neighbor masks
    neigh = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        neigh[i] = mask
    
    # Order by degree descending
    deg = [bin(neigh[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    # Precompute for each position in order, which earlier positions are neighbors
    pos_of = [0] * n
    for idx, v in enumerate(order):
        pos_of[v] = idx
    
    # For each vertex v in order, compute bitmask of neighbors that come before v in order
    earlier_neigh = [0] * n
    for idx, v in enumerate(order):
        mask = 0
        for u in range(n):
            if (neigh[v] >> u) & 1:
                if pos_of[u] < idx:
                    mask |= (1 << pos_of[u])
        earlier_neigh[v] = mask
    
    # Bitmask for colors: color_masks[c] has bit idx set if order[idx] has color c
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
    return sum(sum(adj[i]) for i in range(n)) // 2

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
    for i in range(5):
        u, v = i, (i+1) % 5
        adj[u][v] = adj[v][u] = 1
    for i in range(5):
        u, w = i, (i+1) % 5
        adj[u][5+w] = adj[5+w][u] = 1
        adj[w][5+u] = adj[5+u][w] = 1
    for u in range(5):
        adj[5+u][10] = adj[10][5+u] = 1
    return adj

def _find_max_independent_sets(adj, n, count, size):
    """Find 'count' independent sets of size 'size' in the graph."""
    sets = []
    vertices = list(range(n))
    
    def is_independent(S):
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if adj[S[i]][S[j]]:
                    return False
        return True

    for comb in combinations(vertices, size):
        if is_independent(comb):
            sets.append(list(comb))
            if len(sets) >= count:
                break
    return sets

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    n = 16
    adj = _groetzsch_adj(n)
    
    iso_sets = _find_max_independent_sets(adj, 11, 5, 5)
    if len(iso_sets) < 5:
        while len(iso_sets) < 5:
            iso_sets.append(iso_sets[0] if iso_sets else [])
    
    for i in range(5):
        v_new = 11 + i
        for u in iso_sets[i]:
            adj[v_new][u] = 1
            adj[u][v_new] = 1
    
    best_adj = [row[:] for row in adj]
    best_edges = _count_edges(adj, n)
    
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
    
    # Precompute list of all possible edges for faster iteration
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for iteration in range(8000):
        if time.time() > deadline:
            break
        
        # Build non-edges list with current cn check
        non_edges = []
        for (i, j) in all_pairs:
            if not adj[i][j] and cn[i][j] == 0:
                non_edges.append((i, j))
        random.shuffle(non_edges)
        
        improved = False
        
        # Phase 1: Try adding edges
        for (i, j) in non_edges:
            if time.time() > deadline:
                break
            
            # Tentatively add
            adj[i][j] = 1
            adj[j][i] = 1
            
            # Update common neighbor masks
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
                new_edges = _count_edges(adj, n)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = [row[:] for row in adj]
                improved = True
            else:
                # Revert
                adj[i][j] = 0
                adj[j][i] = 0
                for k in range(n):
                    if k != i and k != j:
                        if adj[i][k]:
                            cn[i][k] &= ~(1 << j)
                            cn[k][i] &= ~(1 << j)
                        if adj[j][k]:
                            cn[j][k] &= ~(1 << i)
                            cn[k][j] &= ~(1 << i)
        
        # Phase 2: Try remove+add moves
        if not improved and time.time() < deadline:
            edge_list = [(i, j) for (i, j) in all_pairs if adj[i][j]]
            random.shuffle(edge_list)
            
            for (ri, rj) in edge_list[:30]:
                if time.time() > deadline:
                    break
                
                # Remove edge (ri, rj)
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
                
                # Try adding some non-edges that were previously blocked
                added = False
                for (ai, aj) in non_edges:
                    if time.time() > deadline:
                        break
                    if adj[ai][aj]:
                        continue
                    if cn[ai][aj] != 0:
                        continue
                    
                    adj[ai][aj] = 1
                    adj[aj][ai] = 1
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
                        adj[ai][aj] = 0
                        adj[aj][ai] = 0
                        for k in range(n):
                            if k != ai and k != aj:
                                if adj[ai][k]:
                                    cn[ai][k] &= ~(1 << aj)
                                    cn[k][ai] &= ~(1 << aj)
                                if adj[aj][k]:
                                    cn[aj][k] &= ~(1 << ai)
                                    cn[k][aj] &= ~(1 << ai)
                
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
                
                if added:
                    new_edges = _count_edges(adj, n)
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = [row[:] for row in adj]
                    improved = True
                    break
        
        if not improved and (iteration % 50 == 0):
            # Occasionally restart with a fresh random perturbation
            if time.time() > deadline - 10:
                break
    
    return _get_edges(best_adj, n)
