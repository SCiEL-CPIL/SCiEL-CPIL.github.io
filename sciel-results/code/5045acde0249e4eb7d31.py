import time
from math import gcd

def construct():
    """
    Returns the largest subset of [1, 255] containing 255 such that
    gcd(a, b) > 1 for all distinct pairs a, b.
    
    Uses exact Maximum Clique algorithm (Bron-Kerbosch with pivot and greedy coloring)
    on the subgraph induced by numbers in [1, 255] that share a factor with 255.
    Since 255 is a universal vertex in this subgraph, the maximum clique will contain 255.
    """
    deadline = time.time() + 85.0  # Safety margin from 90s limit
    t0 = time.time()
    
    # 1. Identify candidate vertices
    # A vertex v must be in the solution if it is part of a clique containing 255.
    # Since gcd(a,b)>1 for all pairs, and 255 is in the set, gcd(v, 255) must be > 1.
    # So v must be divisible by 3, 5, or 17.
    candidates = []
    for v in range(1, 256):
        if gcd(v, 255) > 1:
            candidates.append(v)
            
    n = len(candidates)
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # 2. Build adjacency bitsets
    # adj[i] is a bitset of neighbors of i
    adj = [0] * n
    for i in range(n):
        vi = candidates[i]
        for j in range(i + 1, n):
            vj = candidates[j]
            if gcd(vi, vj) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 3. Maximum Clique Search (Bron-Kerbosch with pivot and greedy coloring)
    # We want to find the size and the set of the maximum clique.
    best_clique = []
    best_size = 0
    
    # Greedy coloring of the subgraph induced by P (bitset)
    # Returns list of (color, vertex) sorted by color descending for processing order
    def greedy_coloring(P):
        """
        Assigns colors to vertices in P such that adjacent vertices have different colors.
        The color of a vertex is an upper bound on the clique size in its neighborhood.
        Returns:
        - colors: list of (color, vertex_index)
        """
        U = P
        colors = []
        color = 0
        # We process vertices and assign them colors.
        # Standard Tomita coloring:
        # While U is not empty:
        #   color++
        #   Pick a vertex v from U (e.g., first one)
        #   Assign color to v
        #   Remove v from U
        #   Remove neighbors of v from U (they must have higher color)
        #   Actually, the standard greedy coloring for bounds works like this:
        #   We want to order vertices v1, v2... such that color(vi) <= i.
        #   Then we can prune if current_size + color(vi) <= best_size.
        
        # Let's implement the specific greedy coloring used in Tomita's max clique:
        # It produces an ordering of vertices and a color for each.
        # The color of a vertex is the max color of its neighbors in the ordering + 1.
        # A simpler implementation:
        
        order = []
        color_of = {}
        U = P
        c = 0
        while U:
            c += 1
            # Pick a vertex from U
            # To make it fast, pick the first set bit
            # Find first set bit in U
            # Since n is small (~127), we can iterate
            v = None
            temp = U
            while temp:
                # lowest set bit
                lsb = temp & -temp
                idx = (lsb.bit_length() - 1)
                v = idx
                break
            if v is None:
                break
            color_of[v] = c
            order.append(v)
            # Remove v and its neighbors from U for the next color class?
            # No, in greedy coloring for clique bounds:
            # We partition U into independent sets.
            # Color 1: max independent set? No, just greedy independent set.
            # Let S_c be the set of vertices colored c. S_c must be an independent set.
            # So if v is colored c, its neighbors cannot be colored c.
            # So we remove v and its neighbors from U.
            U &= ~(1 << v)
            U &= ~adj[v]
            
        # Now we have color_of for all vertices in P.
        # We need to return them in an order suitable for branching.
        # Usually, we branch on vertices with highest color first.
        # So sort by color descending.
        sorted_vertices = sorted(order, key=lambda x: color_of[x], reverse=True)
        
        return sorted_vertices, color_of

    def bron_kerbosch(P, X, R):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        # Greedy coloring of P
        # P is a bitset
        if not P:
            return
            
        vertices, color_map = greedy_coloring(P)
        
        # Process vertices in order of decreasing color
        # The bound is: len(R) + color(v) <= best_size => prune
        for v in vertices:
            if time.time() > deadline:
                return
            
            c = color_map[v]
            if len(R) + c <= best_size:
                # Since we are processing in decreasing color,
                # if this one doesn't satisfy the bound, neither will subsequent ones
                # (which have lower or equal color)
                break
                
            # Expand
            new_R = R | {v}
            # New P is intersection of P and neighbors of v
            # Note: v is in P. We remove v from P for the next iterations of the loop.
            # But for the recursive call, P' = P \ {v} \ N(v)? No.
            # Standard BK:
            # P' = P \ {v} & N(v)  -> Wait, N(v) includes v? No, adj[v] usually excludes v.
            # Let's stick to: P_next = (P & adj[v])
            # X_next = (X & adj[v])
            
            P_next = P & adj[v]
            X_next = X & adj[v]
            
            bron_kerbosch(P_next, X_next, new_R)
            
            # Remove v from P and add to X
            P &= ~(1 << v)
            X |= (1 << v)

    # Initial call
    # P = all candidates
    # X = empty
    # R = empty
    full_P = (1 << n) - 1
    
    # Optimization: 255 is a universal vertex. We can force it into the solution.
    # If we force 255, the problem reduces to finding the max clique in N(255).
    # N(255) is exactly all other vertices in candidates (since 255 connects to all).
    # So we can just find max clique in candidates \ {255} and add 255.
    # This saves a layer of recursion and ensures 255 is included.
    
    idx_255 = val_to_idx[255]
    
    # Remove 255 from P, start R with 255
    P_init = full_P & ~(1 << idx_255)
    R_init = {255}
    
    # The neighbors of 255 in the graph are all other candidates.
    # So P_init is correct.
    # X_init is empty.
    
    bron_kerbosch(P_init, 0, R_init)
    
    # If for some reason the search failed or found nothing (shouldn't happen),
    # fallback to multiples of 3
    if not best_clique:
        # Multiples of 3 form a clique
        best_clique = [v for v in candidates if v % 3 == 0]
        
    return best_clique

# Test
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    valid = True
    if 255 not in res:
        valid = False
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if gcd(res[i], res[j]) == 1:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
    print(res)
