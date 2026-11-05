import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    N = 256

    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Identify forbidden vertices (self-loop: 2*v is a square)
    forbidden = set()
    for v in range(1, N + 1):
        if 2 * v in squares:
            forbidden.add(v)

    # Valid vertices (0-indexed internally)
    valid = [v for v in range(1, N + 1) if v not in forbidden]
    M = len(valid)  # number of candidates

    # Build conflict adjacency as bitsets
    # adj[i] is a Python int bitmask: bit j set iff valid[i] + valid[j] is a square
    adj = [0] * M
    for i in range(M):
        vi = valid[i]
        for sq in squares:
            u = sq - vi
            if 1 <= u <= N and u != vi:
                # Find index of u in valid
                # Use a lookup dict for speed
                pass
        # More efficient: iterate over all j
        for j in range(M):
            if i != j and (vi + valid[j]) in squares:
                adj[i] |= (1 << j)

    # Precompute degree
    degree = [adj[i].bit_count() for i in range(M)]

    # ---------- Local search helpers ----------

    def local_improve(mask):
        """Add any vertex that doesn't conflict."""
        improved = True
        while improved:
            improved = False
            not_in = ((1 << M) - 1) & ~mask
            m = not_in
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if (adj[v] & mask) == 0:
                    mask |= lsb
                    improved = True
                    break
        return mask

    def greedy_random_init():
        """Randomized greedy independent set construction."""
        available = (1 << M) - 1
        selected = 0
        while available:
            best_v = -1
            min_deg = M + 1
            m = available
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                deg = (adj[v] & available).bit_count()
                deg += random.uniform(0, 0.5)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            if best_v == -1:
                break
            selected |= (1 << best_v)
            available &= ~((1 << best_v) | (adj[best_v] & available))
        return selected

    # ---------- Exact Branch-and-Bound (Tomita-style) ----------
    # We solve MIS on the conflict graph G.
    # Equivalent to Maximum Clique on complement graph G'.
    # But we work directly on G with a recursive formulation:
    #   MIS(P, X): P = candidate set, X = current independent set
    #   Branch on a vertex v in P: include v (MIS(P \ N[v], X ∪ {v})) 
    #   or exclude v (MIS(P \ {v}, X)).
    # Upper bound: |X| + |P| (trivial) or tighter via coloring.

    # For tighter bounds, we use a greedy coloring of the induced subgraph on P
    # in the COMPLEMENT (i.e., coloring the conflict graph on P gives an upper 
    # bound on the clique in complement = upper bound on MIS in original).
    # Actually: MIS in G = Max Clique in G'. 
    # Upper bound on Max Clique in G' induced on P = number of colors in a greedy 
    # coloring of G' induced on P.
    # But G' is dense (since G is sparse), so coloring G' is expensive.
    # Alternative: upper bound on MIS in G induced on P = |P| (trivial) or 
    # use a packing bound.
    
    # Better approach: Use the fact that the graph is sparse. 
    # Use a simple branch-and-bound with the following:
    #   - Order vertices by degree (highest first for branching)
    #   - Upper bound: |X| + max possible from P, where we use a greedy 
    #     independent set on P as a lower bound and |P| as upper.
    #   - Actually, for MIS, a good upper bound is obtained by a greedy 
    #     coloring of the CONFLICT graph on P: the number of colors in a 
    #     proper coloring of G[P] gives an upper bound on the size of an 
    #     independent set in G[P]? NO. A proper coloring of G[P] partitions P 
    #     into independent sets. The size of the largest independent set in 
    #     G[P] is at most |P|. The number of colors is a lower bound on the 
    #     chromatic number, not directly an upper bound on alpha.
    
    # Let me reconsider. The standard Tomita algorithm works on Max Clique.
    # Max Clique in G' = MIS in G.
    # For Max Clique, upper bound = number of colors in greedy coloring of G'[P].
    # G' is the complement of G. Since G is sparse, G' is dense.
    # Coloring a dense graph: greedy coloring uses few colors (close to n).
    # This may not be tight.
    
    # Alternative: Work directly with MIS and use a different bound.
    # Upper bound on MIS in G[P]: We can use the fact that any independent set 
    # in G[P] has size at most the maximum degree-based bound, or use a 
    # simple LP relaxation.
    
    # Given time constraints, let's use a practical approach:
    # 1. Get a good initial solution via local search.
    # 2. Use branch-and-bound with the trivial upper bound |X| + |P|, 
    #    but with good vertex ordering and pruning.
    # 3. Since the graph is sparse and M ~ 245, and the optimal is ~104, 
    #    the search space may be manageable with good ordering.

    # Actually, let's use a different strategy: 
    # Use the complement graph and Max Clique with Tomita's coloring bound.
    # Build complement adjacency.
    
    # comp_adj[i] = bitmask of j where i and j are NOT adjacent in G (and i != j)
    comp_adj = [0] * M
    for i in range(M):
        # In complement, i is adjacent to all except itself and its neighbors in G
        all_mask = (1 << M) - 1
        comp_adj[i] = all_mask & ~adj[i] & ~(1 << i)

    # Tomita's algorithm for Max Clique
    # We'll use a recursive approach with greedy coloring for upper bounds.
    
    best_clique = 0  # This will be the MIS size
    best_clique_mask = 0

    def greedy_color(candidates_mask, order):
        """
        Greedy coloring of the complement graph induced on candidates.
        Returns (num_colors, color_assignment_dict).
        The color assignment gives an upper bound on clique size.
        """
        # order is a list of vertices in candidates, ordered by degree in complement
        colors = {}
        next_color = 0
        for v in order:
            # Find the smallest color not used by v's neighbors in the colored set
            used = set()
            m = comp_adj[v] & candidates_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                m ^= lsb
                if u in colors:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            if c > next_color:
                next_color = c
        return next_color + 1, colors

    def max_clique_tomita(P, X, deadline):
        """
        Tomita's branch-and-bound for Maximum Clique in the complement graph.
        P: candidate vertices (bitmask)
        X: current clique (bitmask)
        Returns (best_size, best_mask) found in this subtree.
        """
        nonlocal best_clique, best_clique_mask
        
        if time.time() > deadline:
            return best_clique, best_clique_mask
        
        if P == 0:
            if X.bit_count() > best_clique:
                best_clique = X.bit_count()
                best_clique_mask = X
            return best_clique, best_clique_mask
        
        # Get list of vertices in P
        verts = []
        m = P
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            verts.append(v)
        
        # Order by degree in complement (highest first)
        verts.sort(key=lambda v: (comp_adj[v] & P).bit_count(), reverse=True)
        
        # Greedy coloring for upper bound
        num_colors, color_assign = greedy_color(P, verts)
        
        # If current clique size + num_colors <= best, prune
        if X.bit_count() + num_colors <= best_clique:
            return best_clique, best_clique_mask
        
        # Branch: process vertices in order of decreasing color (last colored first)
        # Group vertices by color
        color_groups = {}
        for v in verts:
            c = color_assign[v]
            if c not in color_groups:
                color_groups[c] = []
            color_groups[c].append(v)
        
        # Process from highest color to lowest
        max_c = max(color_groups.keys()) if color_groups else -1
        
        for c in range(max_c, -1, -1):
            if time.time() > deadline:
                return best_clique, best_clique_mask
            
            if c not in color_groups:
                continue
            
            # Upper bound for this branch: X.size + c + 1
            if X.bit_count() + c + 1 <= best_clique:
                continue
            
            group = color_groups[c]
            for v in group:
                if time.time() > deadline:
                    return best_clique, best_clique_mask
                
                # Check if v is still in P
                if not (P & (1 << v)):
                    continue
                
                # Include v in clique
                new_X = X | (1 << v)
                # New candidates: intersection of P and neighbors of v in complement
                new_P = P & comp_adj[v]
                
                max_clique_tomita(new_P, new_X, deadline)
                
                # Exclude v
                P &= ~(1 << v)
                
                # Update upper bound check
                if X.bit_count() + (P.bit_count()) <= best_clique:
                    break
        
        return best_clique, best_clique_mask

    # ---------- Main solver ----------

    # Get initial solution via local search
    best_size = 0
    best_mask = 0

    # Multiple random greedy initializations
    for _ in range(20):
        if time.time() > DEADLINE - 5:
            break
        mask = greedy_random_init()
        mask = local_improve(mask)
        if mask.bit_count() > best_size:
            best_size = mask.bit_count()
            best_mask = mask

    # Local search improvement
    def local_search(mask, max_iters=100):
        for _ in range(max_iters):
            if time.time() > DEADLINE - 3:
                break
            mask = local_improve(mask)
            improved = False
            
            # Try 1-1 swaps
            verts_in = [v for v in range(M) if (mask >> v) & 1]
            random.shuffle(verts_in)
            for u in verts_in:
                if time.time() > DEADLINE - 3:
                    break
                mask_wo_u = mask & ~(1 << u)
                # Potential vertices to add: neighbors of u in G that are not in mask
                potential = adj[u] & ~mask
                pm = potential
                while pm:
                    lsb = pm & -pm
                    v = lsb.bit_length() - 1
                    pm ^= lsb
                    # v can be added if it only conflicts with u in mask
                    if (adj[v] & mask) == (1 << u):
                        new_mask = mask_wo_u | lsb
                        new_mask = local_improve(new_mask)
                        if new_mask.bit_count() > mask.bit_count():
                            mask = new_mask
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        return mask

    # Improve best with local search
    best_mask = local_search(best_mask, max_iters=50)
    best_size = best_mask.bit_count()

    # Now try exact branch-and-bound if time permits
    # Convert best_mask to clique in complement
    # The independent set in G corresponds to a clique in G'
    # So best_clique_mask = best_mask (same vertex set)
    best_clique = best_size
    best_clique_mask = best_mask

    # Try to improve with B&B
    if time.time() < DEADLINE - 10:
        # Start B&B with the full candidate set
        # But we need to be careful about recursion depth
        # M is ~245, so recursion depth could be large
        # Let's limit the search
        
        full_P = (1 << M) - 1
        # Order vertices by degree in complement
        all_verts = list(range(M))
        all_verts.sort(key=lambda v: comp_adj[v].bit_count(), reverse=True)
        
        # Run B&B with a time limit
        deadline_bnb = min(DEADLINE - 2, time.time() + 20)
        max_clique_tomita(full_P, 0, deadline_bnb)
        
        if best_clique > best_size:
            best_size = best_clique
            best_mask = best_clique_mask

    # Final local search polish
    if time.time() < DEADLINE - 2:
        best_mask = local_search(best_mask, max_iters=30)
        best_size = best_mask.bit_count()

    # Convert mask to list of 1-based values
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(valid[v])
        m ^= lsb

    return sorted(result)
