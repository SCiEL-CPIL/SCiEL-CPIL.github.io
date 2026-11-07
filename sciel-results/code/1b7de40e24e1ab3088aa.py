"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    n = 15
    deadline = time.time() + 80.0
    
    def is_valid(adj_masks):
        """Check if graph with adjacency bitmasks is valid."""
        degs = [bin(adj_masks[i]).count('1') for i in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if not (adj_masks[u] & (1 << v)):
                    continue
                nu = adj_masks[u] & ~(1 << v)
                nv = adj_masks[v] & ~(1 << u)
                for s in range(n):
                    if not (nu & (1 << s)):
                        continue
                    ds = degs[s]
                    nv_excl_s = nv & ~(1 << s)
                    t = nv_excl_s
                    while t:
                        lsb = t & (-t)
                        ti = lsb.bit_length() - 1
                        if degs[ti] == ds:
                            return False
                        t &= t - 1
        return True
    
    def edges_from_masks(adj_masks):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj_masks[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    # Start with K_{7,8}
    adj = [0] * n
    for u in range(7):
        for v in range(7, 15):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    all_pairs = list(combinations(range(n), 2))
    
    # Greedy: try to add edges one by one
    for i, j in all_pairs:
        if time.time() > deadline - 5.0:
            break
        if adj[i] & (1 << j):
            continue
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
        if is_valid(adj):
            pass  # keep it
        else:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
    
    # Local search: try removing one edge and adding another
    for _ in range(100):
        if time.time() > deadline - 5.0:
            break
        improved = False
        # Try: remove edge (a,b), add edge (c,d)
        for a, b in all_pairs:
            if time.time() > deadline - 5.0:
                break
            if not (adj[a] & (1 << b)):
                continue
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            for c, d in all_pairs:
                if time.time() > deadline - 5.0:
                    break
                if c == a and d == b:
                    continue
                if adj[c] & (1 << d):
                    continue
                adj[c] |= (1 << d)
                adj[d] |= (1 << c)
                if is_valid(adj):
                    improved = True
                    break
                else:
                    adj[c] &= ~(1 << d)
                    adj[d] &= ~(1 << c)
            if improved:
                break
            else:
                # Restore edge (a,b)
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
    
    return edges_from_masks(adj)
# EVOLVE-BLOCK-END
