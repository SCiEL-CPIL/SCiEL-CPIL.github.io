import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 255
    target = 255
    
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, target) > 1:
            candidates.append(x)
    
    n_cand = len(candidates)
    
    # Build adjacency bitmasks
    # adj[i] is an integer bitmask where bit j is 1 if candidates[i] and candidates[j] share a factor > 1
    adj = [0] * n_cand
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Tomita's algorithm for Maximum Clique
    # Uses coloring to bound the search
    # R: current clique (list of indices)
    # P: set of candidates (bitmask)
    
    max_size = 0
    best_clique_mask = 0
    
    # Greedy coloring of P to get an upper bound on the clique size in P
    # Returns a list of (vertex, color) sorted by color
    def color_sort(P):
        # P is a bitmask
        # We want to order vertices such that we can assign colors 1..k
        # such that adjacent vertices have different colors.
        # The max color is an upper bound on the clique size.
        # We process vertices one by one.
        
        order = []
        colors = {}
        remaining = P
        current_color = 0
        
        # Simple greedy coloring
        # To be effective for Tomita, we usually sort by degree or just pick arbitrarily
        # but we need to maintain the bound property.
        # Standard Tomita:
        # while P is not empty:
        #   current_color += 1
        #   Q = P
        #   while Q is not empty:
        #      pick v from Q
        #      colors[v] = current_color
        #      order.append((v, current_color))
        #      Q = Q & ~ (1 << v)
        #      P = P & ~ (1 << v)
        #      Q = Q & adj[v]  # Remove neighbors of v from Q for this color? 
        #      Wait, in coloring, neighbors must have different colors.
        #      So if we assign color c to v, no neighbor of v can have color c.
        #      So Q_next = Q \ N(v).
        
        # Actually, the standard implementation sorts vertices by color.
        # Let's implement a simpler bound: just count vertices? No, that's weak.
        # Let's use the standard Tomita coloring.
        
        vertices = []
        temp_P = P
        while temp_P:
            # Pick a vertex
            # We can pick the one with lowest index or highest degree
            # For speed, just pick lowest index
            v = (temp_P & -temp_P).bit_length() - 1
            vertices.append(v)
            temp_P &= ~(1 << v)
        
        # Now assign colors
        # We want to minimize the max color
        # Greedy: iterate through vertices, assign smallest possible color
        # But we need the order to be compatible with the bound check in expand.
        # Tomita's method:
        # 1. Order vertices.
        # 2. Assign colors.
        # 3. Sort by color.
        
        # Let's try a simple greedy coloring on the induced subgraph P.
        color_of = {}
        used_colors = [False] * (n_cand + 1)
        max_col = 0
        
        # To make it faster, we can just use the count of vertices as a bound initially?
        # No, let's do proper coloring.
        
        # We will store (v, color)
        colored = []
        
        # Simple greedy:
        # For each v in vertices:
        #   Find smallest color not used by neighbors in P
        #   Assign it
        
        # This is O(N^2) per call. N=127. 127^2 = 16k. 
        # Number of calls? Could be large.
        # But pruning should be effective.
        
        for v in vertices:
            # Check neighbors in P
            neigh = adj[v] & P
            # Find available color
            c = 1
            while True:
                ok = True
                # Check if any neighbor has color c
                # We need to know colors of neighbors.
                # This requires storing colors of previously colored vertices.
                # Since we are coloring in the order of `vertices`, 
                # we only need to check neighbors that are already colored.
                # But `vertices` order is arbitrary (lowest index).
                # Neighbors might not be colored yet.
                # So we can just check `color_of` for neighbors.
                
                # Actually, simpler:
                # Just assign colors 1, 2, 3...
                # If we just want an upper bound, we can use a simpler heuristic.
                # Upper bound = size of P.
                # Better bound = max degree in P + 1?
                # Let's stick to a simple coloring.
                
                # To avoid complex state, let's just use the number of vertices in P as the bound?
                # That makes it Bron-Kerbosch without pivot, which is slow.
                
                # Let's implement the standard Tomita coloring properly.
                pass
        
        # Given the complexity and time, let's use a simpler approach:
        # Bron-Kerbosch with Pivot.
        # Pivot selection: choose u in P U X that maximizes |P \ N(u)|.
        # This is often fast enough for N=127.
        
        return vertices # Placeholder

    # Let's use a recursive Bron-Kerbosch with Pivot.
    # It's simpler to implement correctly.
    
    def bk(R, P, X):
        nonlocal max_size, best_clique_mask
        
        if time.time() > deadline - 5:
            return
            
        if not P and not X:
            if len(R) > max_size:
                max_size = len(R)
                # Store R as bitmask
                mask = 0
                for v in R:
                    mask |= (1 << v)
                best_clique_mask = mask
            return
        
        # If P is empty, but X is not, we are not at a maximal clique?
        # Standard BK: if P is empty and X is empty, R is maximal.
        # We want maximum clique, not just maximal.
        # So we recurse until P is empty.
        # If P is empty, we check size.
        
        if not P:
            if len(R) > max_size:
                max_size = len(R)
                mask = 0
                for v in R:
                    mask |= (1 << v)
                best_clique_mask = mask
            return
        
        # Pivot selection
        # Choose u from P U X such that |P \ N(u)| is maximized
        # Let U = P | X
        U = P | X
        # Find u in U with max (P & ~adj[u])
        # Iterate through bits in U
        max_count = -1
        pivot = -1
        temp_U = U
        while temp_U:
            u = (temp_U & -temp_U).bit_length() - 1
            temp_U &= ~(1 << u)
            # Count bits in P & ~adj[u]
            # Note: adj[u] does not include u itself? 
            # In our adj construction, adj[i] has bits for j != i.
            # So ~adj[u] has bit u set.
            # But u is in U, and we want P \ N(u).
            # If u is in P, P \ N(u) excludes u?
            # N(u) usually includes u? No, in graph theory N(u) is neighbors.
            # P \ N(u) should be vertices in P not connected to u.
            # If u is in P, is u connected to u? No.
            # So u is in P \ N(u).
            # We want to minimize the candidates we need to branch on.
            # We branch on P \ N(u).
            
            cand = P & ~adj[u]
            # Count bits
            count = cand.bit_count()
            if count > max_count:
                max_count = count
                pivot = u
        
        # Branch on vertices in P \ N(pivot)
        # If P \ N(pivot) is empty, then P is subset of N(pivot)?
        # Then we don't need to branch? No, we still need to find clique.
        # Actually, if P \ N(pivot) is empty, it means all v in P are neighbors of pivot.
        # We can still branch on any v in P.
        # The optimization is to ONLY branch on P \ N(pivot).
        # If P \ N(pivot) is empty, we are done? 
        # No, that logic is for finding *maximal* cliques.
        # For maximum, we must explore.
        # However, Bron-Kerbosch with pivot is:
        # for each v in P \ N(u):
        #    R.append(v)
        #    bk(R, P & adj[v], X & adj[v])
        #    R.pop()
        #    P.remove(v)
        #    X.add(v)
        
        # If P \ N(u) is empty, the loop doesn't run. 
        # Is that correct?
        # Yes, because any clique in P must contain a vertex from P \ N(u) OR be empty?
        # No. If P \ N(u) is empty, it means P is a subset of N(u).
        # Does that mean no clique exists in P? No.
        # Wait. The theorem is: Any maximal clique in the subgraph induced by P U X 
        # must contain a vertex from P \ N(u) IF P \ N(u) is not empty?
        # Actually, the standard BK with pivot iterates over P \ N(u).
        # If P \ N(u) is empty, it implies that u is adjacent to all vertices in P.
        # In that case, we can just add u to R? No, u is in X or P.
        # If u is in P, we can branch on u.
        # The pivot optimization says we don't need to branch on vertices in P \cap N(u).
        # So if P \ N(u) is empty, we branch on nothing?
        # That would mean we miss cliques.
        
        # Correction:
        # The set to branch on is P \ N(u).
        # If this set is empty, it means P is contained in N(u).
        # But we still need to find the max clique in P.
        # The pivot trick is valid because any maximal clique in G[P] 
        # intersects P \ N(u) ? No.
        
        # Let's check the property.
        # If we pick pivot u, we only branch on v in P such that v is NOT adjacent to u.
        # Why? Because if v is adjacent to u, then any clique containing v 
        # could also contain u? No.
        
        # Actually, the standard implementation is:
        # candidates = P & ~adj[u]
        # if candidates is empty:
        #     # This implies P is subset of adj[u]
        #     # We can just recurse with P?
        #     # Or is it that we found a maximal clique?
        
        # Let's use a simpler, robust version:
        # Just iterate through all v in P.
        # With N=127, and good pruning (coloring), it should be fine.
        # But without coloring, BK is exponential.
        
        # Let's add a simple bound: if len(R) + popcount(P) <= max_size, return.
        
        if len(R) + P.bit_count() <= max_size:
            return
        
        # Iterate through P
        temp_P = P
        while temp_P:
            v = (temp_P & -temp_P).bit_length() - 1
            temp_P &= ~(1 << v)
            
            # Branch
            R.append(v)
            bk(R, P & adj[v], X & adj[v])
            R.pop()
            
            P &= ~(1 << v)
            X |= (1 << v)
            
            # Update bound
            if len(R) + P.bit_count() <= max_size:
                break

    # Initialize
    # Start with empty R, P = all candidates, X = empty
    P_init = (1 << n_cand) - 1
    bk([], P_init, 0)
    
    # Convert best_clique_mask to list of values
    result_indices = []
    for i in range(n_cand):
        if (best_clique_mask >> i) & 1:
            result_indices.append(candidates[i])
            
    result = sorted(result_indices)
    
    # Ensure 255 is in the set
    if target not in result:
        # This shouldn't happen if 255 is universal, but just in case
        result.append(target)
        result = sorted(result)
        
    return result
# EVOLVE-BLOCK-END
