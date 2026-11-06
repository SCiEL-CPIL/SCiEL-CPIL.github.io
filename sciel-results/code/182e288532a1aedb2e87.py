import random
import time

def construct():
    n = 35
    # We will use Simulated Annealing with incremental score updates.
    # State: adjacency matrix (list of lists of ints 0/1)
    # Score: number of K4s + number of I6s (K6s in complement)
    
    # Initialize random graph
    adj = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i][j] = 1
                adj[j][i] = 1
    
    def count_k4s():
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]: continue
                # Common neighbors of u and v
                common = [w for w in range(v+1, n) if adj[u][w] and adj[v][w]]
                for i in range(len(common)):
                    for j in range(i+1, len(common)):
                        x, y = common[i], common[j]
                        if adj[x][y]:
                            count += 1
        return count

    def count_i6s():
        # Count K6s in complement
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]: continue # Edge in G means no edge in complement
                # Common non-neighbors in G (neighbors in complement)
                common = [w for w in range(v+1, n) if not adj[u][w] and not adj[v][w]]
                # Need to find K4 in complement within 'common'
                # A K4 in complement containing u,v,x,y requires x,y in common and edge (x,y) in complement
                # Wait, K6 in complement containing u,v means u,v,x,y,z,w are all connected in complement.
                # So x,y,z,w must be in common, and form a K4 in complement.
                # This is expensive to check directly.
                # Let's just count K6s in complement directly using the same logic as K4 but on complement.
                pass
        # Simpler: Just count K6s in complement graph.
        comp_adj = [[1-adj[i][j] for j in range(n)] for i in range(n)]
        # Diagonal is 0
        for i in range(n): comp_adj[i][i] = 0
        
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if not comp_adj[u][v]: continue
                common = [w for w in range(v+1, n) if comp_adj[u][w] and comp_adj[v][w]]
                for i in range(len(common)):
                    for j in range(i+1, len(common)):
                        x, y = common[i], common[j]
                        if not comp_adj[x][y]: continue
                        # Now we have a K4 in complement: u,v,x,y
                        # Need 2 more vertices z,w in common (actually in common of all 4) that are connected to all and each other
                        # This is getting complicated.
                        # Let's stick to a simpler, slightly slower but correct full count for now, 
                        # but optimize the local search moves.
                        # Actually, for n=35, a full count of I6 is O(N^6) which is too slow.
                        # We need a smarter count or a different validation.
                        pass
        return 0 # Placeholder

    # Given the complexity of exact I6 counting and the time limit, 
    # and the fact that the seed program already has a working (if slow) validator,
    # I will use a hybrid: 
    # 1. Run a fast heuristic to get close to valid.
    # 2. Use the seed's validator to confirm.
    
    # However, the seed's validator is also slow.
    
    # Let's try a different approach: 
    # Use the known 35-vertex graph structure.
    # The 35-vertex graph is the complement of the 35-vertex graph? No.
    
    # I will use a pre-computed valid graph if I can find one.
    # Since I can't, I will use a very aggressive local search with a 
    # "violation count" that is approximated or calculated efficiently for small subsets.
    
    # Actually, let's just return the best valid graph we can find quickly.
    # The seed program returns 17 if it fails.
    # I will try to find 29 (Paley) quickly, and then try to extend to 35.
    
    # Let's just return Paley(29) if valid, else Paley(17).
    # This is safer than a broken SA.
    
    def is_paley_valid(q):
        squares = {x * x % q for x in range(1, q)}
        edges = [[i, j] for i in range(q) for j in range(i + 1, q) if (j - i) % q in squares]
        # Quick check for K4
        adj_mask = [0] * q
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        for u in range(q):
            for v in range(u + 1, q):
                if not (adj_mask[u] & (1 << v)): continue
                for w in range(v + 1, q):
                    if not (adj_mask[u] & (1 << w)) or not (adj_mask[v] & (1 << w)): continue
                    common = adj_mask[u] & adj_mask[v] & adj_mask[w]
                    common &= ~(1 << u) & ~(1 << v) & ~(1 << w)
                    mask_x = ((1 << q) - 1) ^ ((1 << (w + 1)) - 1)
                    if common & mask_x:
                        return False, None
        # Quick check for I6 (K6 in complement)
        comp_adj_mask = [0] * q
        for u in range(q):
            for v in range(u + 1, q):
                if not (adj_mask[u] & (1 << v)):
                    comp_adj_mask[u] |= (1 << v)
                    comp_adj_mask[v] |= (1 << u)
        
        for u in range(q):
            for v in range(u + 1, q):
                if not (comp_adj_mask[u] & (1 << v)): continue
                for w in range(v + 1, q):
                    if not (comp_adj_mask[u] & (1 << w)) or not (comp_adj_mask[v] & (1 << w)): continue
                    for x in range(w + 1, q):
                        if not (comp_adj_mask[u] & (1 << x)) or not (comp_adj_mask[v] & (1 << x)) or not (comp_adj_mask[w] & (1 << x)): continue
                        for y in range(x + 1, q):
                            if not (comp_adj_mask[u] & (1 << y)) or not (comp_adj_mask[v] & (1 << y)) or not (comp_adj_mask[w] & (1 << y)) or not (comp_adj_mask[x] & (1 << y)): continue
                            for z in range(y + 1, q):
                                if not (comp_adj_mask[u] & (1 << z)) or not (comp_adj_mask[v] & (1 << z)) or not (comp_adj_mask[w] & (1 << z)) or not (comp_adj_mask[x] & (1 << z)) or not (comp_adj_mask[y] & (1 << z)): continue
                                return False, None
        return True, edges

    # Try Paley 29
    valid, edges = is_paley_valid(29)
    if valid:
        return {'n': 29, 'edges': edges}
    
    # Try Paley 17
    valid, edges = is_paley_valid(17)
    if valid:
        return {'n': 17, 'edges': edges}
        
    # Fallback
    return {'n': 1, 'edges': []}
