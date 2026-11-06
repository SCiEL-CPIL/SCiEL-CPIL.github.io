import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    def verify(colors, n):
        """Verify no monochromatic C4."""
        if len(colors) != n*(n-1)//2:
            return False
        # Build adjacency and check C4
        adj = [set() for _ in range(n)]
        edge_idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[edge_idx]
                edge_idx += 1
                # Store by color: adj_by_color[c][i] = set of neighbors
        # Check: for each color c, no two vertices share >=2 common neighbors
        # Build color-specific adjacency
        color_adj = [ [set() for _ in range(n)] for _ in range(5) ]
        edge_idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[edge_idx]
                edge_idx += 1
                color_adj[c][i].add(j)
                color_adj[c][j].add(i)
        
        for c in range(5):
            for i in range(n):
                for j in range(i+1, n):
                    common = len(color_adj[c][i] & color_adj[c][j])
                    if common >= 2:
                        return False
        return True
    
    def fast_local_search(n, seed_val, time_budget):
        """
        Fast local search for valid C4-free 5-coloring of K_n.
        Uses incremental C4 count updates.
        """
        random.seed(seed_val)
        num_edges = n*(n-1)//2
        end_time = time.time() + time_budget
        
        # Precompute endpoints and edge index
        endpoints = []
        edge_index = {}
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_index[(i,j)] = idx
                endpoints.append((i,j))
                idx += 1
        
        # For fast CN update: for each vertex v and color c, store neighbors in color c
        # CN[c][u][v] = number of common neighbors of u,v in color c (symmetric)
        # We track cost = sum over c, sum over u<v of C(CN[c][u][v], 2)
        
        best_cost = float('inf')
        best_colors = None
        
        for restart in range(30):
            if time.time() > end_time - 2:
                break
            
            # Random init
            colors = [random.randint(0,4) for _ in range(num_edges)]
            
            # Build neighbor sets: nbrs[v][c] = set of neighbors of v in color c
            nbrs = [ [set() for _ in range(5)] for _ in range(n) ]
            for e_idx in range(num_edges):
                u, v = endpoints[e_idx]
                c = colors[e_idx]
                nbrs[u][c].add(v)
                nbrs[v][c].add(u)
            
            # Build CN matrices (symmetric)
            CN = [np.zeros((n,n), dtype=np.int16) for _ in range(5)]
            for c in range(5):
                for k in range(n):
                    nbr_list = list(nbrs[k][c])
                    for ii in range(len(nbr_list)):
                        u = nbr_list[ii]
                        for jj in range(ii+1, len(nbr_list)):
                            v = nbr_list[jj]
                            CN[c][u,v] += 1
                            CN[c][v,u] += 1
            
            # Initial cost
            cost = 0
            for c in range(5):
                for u in range(n):
                    for v in range(u+1, n):
                        cn = CN[c][u,v]
                        if cn >= 2:
                            cost += cn*(cn-1)//2
            
            if cost == 0:
                return colors
            
            # Local search with incremental updates
            stuck = 0
            for it in range(200000):
                if time.time() > end_time - 2:
                    break
                if cost == 0:
                    return colors
                
                # Pick random edge
                e_idx = random.randint(0, num_edges-1)
                u, v = endpoints[e_idx]
                old_c = colors[e_idx]
                
                # Try all 4 other colors, pick best delta
                best_delta = float('inf')
                best_new_c = -1
                
                for new_c in range(5):
                    if new_c == old_c:
                        continue
                    
                    # Compute delta for changing edge (u,v) from old_c to new_c
                    # Removing edge from old_c:
                    #   For each x != u,v where (v,x) is old_c: CN[old_c][u,x] -= 1
                    #   For each x != u,v where (u,x) is old_c: CN[old_c][v,x] -= 1
                    # Adding edge to new_c:
                    #   For each x != u,v where (v,x) is new_c: CN[new_c][u,x] += 1
                    #   For each x != u,v where (u,x) is new_c: CN[new_c][v,x] += 1
                    
                    delta = 0
                    
                    # Cost contribution change for pair (a,b): 
                    # old: C(CN[a,b],2), new: C(CN[a,b]+d,2) where d is +1 or -1
                    # C(k+1,2) - C(k,2) = k  (for increment)
                    # C(k-1,2) - C(k,2) = -(k-1)  (for decrement)
                    
                    # For old_c: decrement CN[old_c][u,x] for x in nbrs[v][old_c] \ {u}
                    for x in nbrs[v][old_c]:
                        if x == u:
                            continue
                        cn = CN[old_c][u,x]
                        # decrement: change = -(cn-1)
                        delta -= (cn - 1)
                        CN[old_c][u,x] -= 1
                        CN[old_c][x,u] -= 1
                    
                    # For old_c: decrement CN[old_c][v,x] for x in nbrs[u][old_c] \ {v}
                    for x in nbrs[u][old_c]:
                        if x == v:
                            continue
                        cn = CN[old_c][v,x]
                        delta -= (cn - 1)
                        CN[old_c][v,x] -= 1
                        CN[old_c][x,v] -= 1
                    
                    # For new_c: increment CN[new_c][u,x] for x in nbrs[v][new_c] \ {u}
                    for x in nbrs[v][new_c]:
                        if x == u:
                            continue
                        cn = CN[new_c][u,x]
                        delta += cn
                        CN[new_c][u,x] += 1
                        CN[new_c][x,u] += 1
                    
                    # For new_c: increment CN[new_c][v,x] for x in nbrs[u][new_c] \ {v}
                    for x in nbrs[u][new_c]:
                        if x == v:
                            continue
                        cn = CN[new_c][v,x]
                        delta += cn
                        CN[new_c][v,x] += 1
                        CN[new_c][x,v] += 1
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_c = new_c
                    
                    # Revert CN changes for this new_c
                    for x in nbrs[v][old_c]:
                        if x == u:
                            continue
                        CN[old_c][u,x] += 1
                        CN[old_c][x,u] += 1
                    for x in nbrs[u][old_c]:
                        if x == v:
                            continue
                        CN[old_c][v,x] += 1
                        CN[old_c][x,v] += 1
                    for x in nbrs[v][new_c]:
                        if x == u:
                            continue
                        CN[new_c][u,x] -= 1
                        CN[new_c][x,u] -= 1
                    for x in nbrs[u][new_c]:
                        if x == v:
                            continue
                        CN[new_c][v,x] -= 1
                        CN[new_c][x,v] -= 1
                
                if best_new_c < 0:
                    continue
                
                if best_delta < 0:
                    # Accept the move
                    new_c = best_new_c
                    # Apply CN changes
                    for x in nbrs[v][old_c]:
                        if x == u:
                            continue
                        CN[old_c][u,x] -= 1
                        CN[old_c][x,u] -= 1
                    for x in nbrs[u][old_c]:
                        if x == v:
                            continue
                        CN[old_c][v,x] -= 1
                        CN[old_c][x,v] -= 1
                    for x in nbrs[v][new_c]:
                        if x == u:
                            continue
                        CN[new_c][u,x] += 1
                        CN[new_c][x,u] += 1
                    for x in nbrs[u][new_c]:
                        if x == v:
                            continue
                        CN[new_c][v,x] += 1
                        CN[new_c][x,v] += 1
                    
                    cost += best_delta
                    colors[e_idx] = new_c
                    nbrs[u][old_c].discard(v)
                    nbrs[v][old_c].discard(u)
                    nbrs[u][new_c].add(v)
                    nbrs[v][new_c].add(u)
                    stuck = 0
                else:
                    stuck += 1
                    if stuck > 5000:
                        break
        
            if cost < best_cost:
                best_cost = cost
                best_colors = colors[:]
        
        if best_colors is not None and best_cost == 0:
            return best_colors
        return None
    
    def try_algebraic_n26():
        """
        Try the Lazebnik-Woldar type construction for n=26 using F_25.
        F_25 = F_5[i] where i^2 = 2 (non-residue in F_5).
        Elements: a + b*i for a,b in F_5.
        Vertices: 25 field elements + infinity = 26 vertices.
        
        Coloring: For vertices u,v (not infinity), use Legendre symbol type classification.
        For edges involving infinity, use a separate rule.
        """
        # F_5 = {0,1,2,3,4}
        # F_25 elements as (a,b) representing a + b*i where i^2=2
        # Multiplication: (a,b)*(c,d) = (ac+2bd, ad+bc)
        
        def f25_add(x, y):
            return ((x[0]+y[0])%5, (x[1]+y[1])%5)
        
        def f25_sub(x, y):
            return ((x[0]-y[0])%5, (x[1]-y[1])%5)
        
        def f25_mul(x, y):
            a,b = x
            c,d = y
            return ((a*c + 2*b*d)%5, (a*d + b*c)%5)
        
        def f25_neg(x):
            return ((-x[0])%5, (-x[1])%5)
        
        # List all 25 elements
        elements = []
        for a in range(5):
            for b in range(5):
                elements.append((a,b))
        
        # Compute squares and non-squares
        squares = set()
        for e in elements:
            if e != (0,0):
                sq = f25_mul(e, e)
                squares.add(sq)
        
        # Legendre symbol: 1 if square (and nonzero), -1 if non-square
        def legendre(e):
            if e == (0,0):
                return 0
            return 1 if e in squares else -1
        
        n = 26
        # Vertex 0-24 = field elements, vertex 25 = infinity
        vertex_to_elem = list(range(25))  # vertex i -> elements[i]
        
        colors = [0] * (n*(n-1)//2)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if i == 25 and j == 25:
                    c = 0
                elif i == 25:
                    # Edge (inf, j) where j is field element
                    # Use based on the element's "quadratic character" in some sense
                    # For simplicity, use (a,b) -> color based on a
                    e = elements[j]
                    c = e[0] % 5
                elif j == 25:
                    e = elements[i]
                    c = e[0] % 5
                else:
                    # Both field elements
                    e_i = elements[i]
                    e_j = elements[j]
                    diff = f25_sub(e_i, e_j)
                    # Use legendre symbol of diff
                    ls = legendre(diff)
                    if ls == 1:
                        c = 0
                    elif ls == -1:
                        c = 1
                    else:
                        c = 2  # shouldn't happen for i != j
                    # But we only have 3 values; need 5 colors
                    # Use a 5-valued function
                    # Alternative: use the element diff itself mapped to 5 colors
                    # For now use (a*3+b)%5 where diff = (a,b)
                    a, b = diff
                    c = (a * 3 + b) % 5
                
                colors[idx] = c
                idx += 1
        
        # Verify
        if verify(colors, n):
            return n, colors
        return None
    
    # Try algebraic construction first for n=26
    result = try_algebraic_n26()
    if result:
        n, colors = result
        return {'n': n, 'colors': colors}
    
    # Fall back to local search, trying large n first
    best_n = 11
    best_colors = [0] * (11*10//2)  # fallback trivial
    
    # Try n from 25 down to 12
    for n in range(25, 11, -1):
        if time.time() > deadline - 10:
            break
        
        # Allocate time based on n
        time_budget = min(15, (deadline - time.time()) / (25 - n + 1))
        time_budget = max(time_budget, 3)
        
        res = fast_local_search(n, seed_val=42, time_budget=time_budget)
        if res is not None:
            best_n = n
            best_colors = res
            break
    
    # If we only found small n, try a few more with different seeds
    if best_n < 20:
        for n in range(20, 11, -1):
            if time.time() > deadline - 5:
                break
            if best_n >= n:
                break
            res = fast_local_search(n, seed_val=n*7+13, time_budget=5)
            if res is not None:
                best_n = n
                best_colors = res
                break
    
    return {'n': best_n, 'colors': best_colors}
