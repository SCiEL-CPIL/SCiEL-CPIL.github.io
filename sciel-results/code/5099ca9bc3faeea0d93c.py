"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    n = 18
    
    # Build Paley graph on 17 vertices (Z_17, quadratic residues)
    residues = set((x * x) % 17 for x in range(1, 17))
    residues.discard(0)
    
    # Build adjacency as list of bitmasks
    adj = [0] * n
    for u in range(17):
        for v in range(u + 1, 17):
            if (v - u) % 17 in residues:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    # Vertex 17 starts isolated
    
    def has_k4(adj_bm, n_verts):
        """Check K4-free using bitmasks."""
        for a in range(n_verts):
            na = adj_bm[a]
            for b in range(a + 1, n_verts):
                if not (na >> b) & 1:
                    continue
                nb = adj_bm[b] & na & ~(1 << a) & ~(1 << b)
                c_mask = nb
                while c_mask:
                    c = (c_mask & (-c_mask)).bit_length() - 1
                    c_mask &= c_mask - 1
                    cd = adj_bm[c] & nb & ~(1 << c)
                    if cd:
                        return True
        return False
    
    def largest_tf(adj_bm, n_verts):
        """Largest triangle-free induced subgraph size."""
        tri_list = []
        for i in range(n_verts):
            for j in range(i + 1, n_verts):
                if not (adj_bm[i] >> j) & 1:
                    continue
                common = adj_bm[i] & adj_bm[j] & ~(1 << i) & ~(1 << j)
                k_mask = common
                while k_mask:
                    k = (k_mask & (-k_mask)).bit_length() - 1
                    k_mask &= k_mask - 1
                    if k > j:
                        tri_list.append((1 << i) | (1 << j) | (1 << k))
        
        for size in range(n_verts, 0, -1):
            for subset in combinations(range(n_verts), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                tf = True
                for tm in tri_list:
                    if (mask & tm) == tm:
                        tf = False
                        break
                if tf:
                    return size
        return 0
    
    current_score = largest_tf(adj, n)
    best_adj = adj[:]
    best_score = current_score
    
    random.seed(999)
    
    while time.time() < deadline:
        improved = False
        candidates = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(candidates)
        
        for (i, j) in candidates:
            if time.time() > deadline:
                break
            
            old = (adj[i] >> j) & 1
            if old:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
            else:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            
            if has_k4(adj, n):
                if old:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                else:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                continue
            
            new_score = largest_tf(adj, n)
            
            if new_score < best_score:
                best_score = new_score
                best_adj = adj[:]
                improved = True
            else:
                if old:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                else:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
        
        if not improved:
            break
    
    adj = best_adj
    
    edges_list = []
    for i in range(n):
        mask = adj[i] & ~((1 << (i + 1)) - 1)
        while mask:
            j = (mask & (-mask)).bit_length() - 1
            mask &= mask - 1
            edges_list.append([i, j])
    
    return edges_list
# EVOLVE-BLOCK-END
