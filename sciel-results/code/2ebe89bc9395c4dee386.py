"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _generate_supports():
    """Generate all 4-element subsets of {0,...,10} as bitmasks."""
    supports = []
    indices = range(D)
    for comb in combinations(indices, 4):
        mask = 0
        for idx in comb:
            mask |= (1 << idx)
        supports.append(mask)
    return np.array(supports, dtype=np.int64)

def _max_independent_set_bnb(adj, deadline):
    """
    Exact Maximum Independent Set solver using Branch and Bound with greedy coloring.
    adj: list of sets, where adj[i] contains neighbors of i.
    Returns the maximum independent set as a list of node indices.
    """
    n = len(adj)
    # Precompute non-neighbors for fast intersection checks? 
    # Actually, standard B&B works on the graph. 
    # Independent Set in G is Clique in G_complement.
    # Let's work on the complement graph for Clique B&B, or stick to IS B&B.
    # IS B&B: 
    # State: (P, X, S) where P is candidates, X is excluded, S is current set.
    # Upper bound for P: color the subgraph induced by P. 
    # If |S| + color_bound < best_size, prune.
    
    # We need the adjacency matrix or list for the graph we are finding IS in.
    # The graph is the conflict graph. We want IS in this graph.
    
    best_set = []
    best_size = 0
    
    # To speed up coloring, we can use a simple greedy coloring on the subgraph.
    # Greedy coloring: order vertices by degree (descending) or just index.
    # For 330 nodes, we can precompute a static ordering or use dynamic.
    
    # We will use a recursive function with manual stack or recursion.
    # Recursion depth up to 330 is fine in Python if we increase limit? 
    # Python default is 1000. 330 is safe.
    
    # Optimization: Use bitsets for P and X? 
    # n=330 fits in 6 64-bit ints. 
    # Python integers are arbitrary precision, so we can use a single int as a bitset for P and X.
    # This is much faster for set operations.
    
    # Convert adj to bitsets
    adj_bits = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_bits[i] |= (1 << j)
    
    # Full set of nodes
    all_nodes = (1 << n) - 1
    
    # Greedy Coloring for Upper Bound
    # Input: bitset P
    # Returns: list of colors (size = number of colors used)
    # We want the number of colors, which is an upper bound on the clique size (and thus IS size? No.)
    # Wait. 
    # If we are finding an Independent Set in G, we are finding a Clique in G_complement.
    # Coloring G gives an upper bound on the Clique size in G.
    # So we should color the CONFLICT GRAPH (G).
    # If the conflict graph has k colors, any independent set has size <= k? 
    # No. An independent set in G is a set of vertices no two of which are adjacent in G.
    # This is a clique in G_complement.
    # The upper bound on the size of an independent set in G is NOT the chromatic number of G.
    # The upper bound on the size of a CLIQUE in G is the chromatic number of G.
    # So, if we want an IS in G, we should apply B&B on G_complement to find a Max Clique.
    # Let's switch to Max Clique in G_complement.
    
    # G_complement adjacency:
    comp_adj_bits = [0] * n
    for i in range(n):
        # i is connected to all j != i in complement, EXCEPT those in G
        # So comp_adj_bits[i] = all_nodes ^ (1 << i) ^ adj_bits[i]
        comp_adj_bits[i] = all_nodes ^ (1 << i) ^ adj_bits[i]
        
    # Now we find Max Clique in G_comp.
    # State: P (candidates), X (excluded), S (current clique)
    # Bound: Greedy Coloring of G_comp[P].
    
    def greedy_color(bitset_P):
        """
        Greedily color the subgraph induced by bitset_P in G_comp.
        Returns the number of colors used.
        """
        if bitset_P == 0:
            return 0
        
        # Extract nodes
        nodes = []
        temp = bitset_P
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            nodes.append(idx)
            temp ^= lsb
            
        # Order by degree in G_comp[P] (descending) for better coloring
        # Degree of v in G_comp[P] is popcount(comp_adj_bits[v] & bitset_P)
        degrees = [(np.count_nonzero(comp_adj_bits[v] & bitset_P), v) for v in nodes]
        degrees.sort(reverse=True)
        
        colors = 0
        colored = 0
        while colored < len(nodes):
            colors += 1
            # Find all uncolored nodes that are not adjacent to each other in G_comp
            # i.e., form an independent set in G_comp, which is a clique in G.
            # Wait, greedy coloring assigns one color to an independent set.
            # So we want to find a maximal independent set in G_comp[P] to assign color `colors`.
            # This is equivalent to finding a maximal clique in G[P].
            
            available = [v for _, v in degrees if (colored >> 0) & 0] # Placeholder
            # Let's just iterate through the sorted nodes and pick if compatible.
            picked = 0
            for _, v in degrees:
                if (colored & (1 << v)):
                    continue
                # Check if v is compatible with all previously picked in this color class
                # Compatible in G_comp means NOT adjacent in G_comp.
                # i.e., (comp_adj_bits[v] & picked) == 0
                if (comp_adj_bits[v] & picked) == 0:
                    picked |= (1 << v)
                    colored |= (1 << v)
                    
            if picked == 0:
                # Should not happen if nodes is not empty
                break
                
        return colors

    def expand(P, X, S, depth):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            return
            
        # Upper bound
        ub = greedy_color(P)
        if len(S) + ub <= best_size:
            return
            
        # Choose a vertex from P
        # Heuristic: choose vertex with highest degree in P?
        # Or just lowest index for simplicity, but degree heuristic is better.
        # Let's pick v with max degree in G_comp[P]
        max_deg = -1
        v = -1
        temp = P
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            deg = np.count_nonzero(comp_adj_bits[idx] & P)
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp ^= lsb
            
        if v == -1:
            return
            
        # Branch 1: Include v in clique
        S.append(v)
        new_P = P & comp_adj_bits[v] # Neighbors of v in G_comp
        new_X = X
        expand(new_P, new_X, S, depth + 1)
        S.pop()
        
        # Branch 2: Exclude v
        new_P = P & ~(1 << v)
        new_X = X | (1 << v)
        expand(new_P, new_X, S, depth + 1)

    # Start
    P = all_nodes
    X = 0
    S = []
    expand(P, X, S, 0)
    
    return best_set

def construct():
    deadline = time.time() + 110
    
    # 1. Generate supports for m=4
    supports_list = []
    indices = range(D)
    for comb in combinations(indices, 4):
        mask = 0
        for idx in comb:
            mask |= (1 << idx)
        supports_list.append(mask)
    
    n_sup = len(supports_list) # 330
    
    # 2. Build adjacency list for conflict graph (intersection size 3)
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        for j in range(i + 1, n_sup):
            inter = supports_list[i] & supports_list[j]
            if inter.bit_count() == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # 3. Find Maximum Independent Set using B&B
    # The B&B solver expects adj as list of sets.
    mis_nodes = _max_independent_set_bnb(adj, deadline)
    
    # 4. Generate vectors for MIS supports
    # We need to assign signs to each support such that pairwise dot products are small.
    # With MIS, any two supports intersect in <= 2 indices.
    # If intersection is 2, max |u.v| = 2.
    # We want to minimize max |u.v| to potentially allow adding more vectors or just ensure validity.
    # Actually, the condition is dist^2 >= norm^2 => 8 - 2u.v >= 4 => u.v <= 2.
    # Since max u.v is 2 (from intersection 2 with same signs), ANY sign assignment works?
    # Wait. If u.v = 2, dist^2 = 4. Valid.
    # If u.v = -2, dist^2 = 12. Valid.
    # So for MIS supports, ANY sign assignment is valid!
    # This means we can just take all 2^4 = 16 sign patterns for each support.
    # Total vectors = 16 * |MIS|.
    
    vectors = []
    for idx in mis_nodes:
        mask = supports_list[idx]
        inds = [i for i in range(D) if (mask >> i) & 1]
        for sign_mask in range(16):
            v = np.zeros(D, dtype=np.int64)
            for j, idx_val in enumerate(inds):
                if sign_mask & (1 << j):
                    v[idx_val] = 1
                else:
                    v[idx_val] = -1
            vectors.append(v)
    
    S4 = np.array(vectors, dtype=np.int64)
    
    # 5. Try to add Shell 2 vectors
    # Shell 2 vectors have norm^2 = 2.
    # Condition for shell 2 vector s and shell 4 vector u:
    # dist^2 >= max(norm^2) = 4.
    # ||s||^2 + ||u||^2 - 2 s.u >= 4
    # 2 + 4 - 2 s.u >= 4
    # 6 - 2 s.u >= 4
    # 2 s.u <= 2 => s.u <= 1.
    
    # Also for two shell 2 vectors s1, s2:
    # 2 + 2 - 2 s1.s2 >= 4
    # 4 - 2 s1.s2 >= 4 => s1.s2 <= 0.
    
    shell2_vecs = []
    for i in range(D):
        for j in range(i+1, D):
            v1 = np.zeros(D, dtype=np.int64)
            v1[i] = 1; v1[j] = -1
            shell2_vecs.append(v1)
            v2 = np.zeros(D, dtype=np.int64)
            v2[i] = -1; v2[j] = 1
            shell2_vecs.append(v2)
    shell2 = np.array(shell2_vecs, dtype=np.int64)
    
    if len(S4) > 0:
        dots_24 = shell2 @ S4.T
        max_ip_24 = dots_24.max(axis=1)
        valid_2_mask = max_ip_24 <= 1
        S2_candidates = shell2[valid_2_mask]
        
        n2 = len(S2_candidates)
        if n2 > 0:
            dots_22 = S2_candidates @ S2_candidates.T
            selected2 = []
            alive2 = list(range(n2))
            while alive2:
                # Pick a candidate, say the first one
                v_idx = alive2[0]
                selected2.append(v_idx)
                # Remove v_idx and any that conflict with v_idx (dot > 0)
                conflicts = np.where(dots_22[v_idx] > 0)[0]
                alive2 = [i for i in alive2 if i not in conflicts and i != v_idx]
            
            S2 = S2_candidates[selected2]
            result = np.vstack([S4, S2])
        else:
            result = S4
    else:
        result = S4
        
    return result

# EVOLVE-BLOCK-END
