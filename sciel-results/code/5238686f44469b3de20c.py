import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # 1. Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330
    
    # Convert supports to bitmasks for fast intersection
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
        
    # 2. Build Conflict Graph and Complement Graph
    # Conflict if intersection size is 3
    # We want Maximum Independent Set in Conflict Graph = Maximum Clique in Complement Graph
    # Let's build the adjacency matrix for the COMPLEMENT graph (edges where they are NOT conflicting)
    # Actually, it's easier to work with the Conflict graph and find MIS.
    # But standard Max Clique algorithms work on the graph where we want the clique.
    # So, let's build the graph G where an edge exists if the supports are COMPATIBLE (intersection < 3).
    # Then we find the Max Clique in G.
    
    adj_complement = [0] * N_S
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = (masks[i] & masks[j]).bit_count()
            if inter < 3:
                # Compatible
                adj_complement[i] |= (1 << j)
                adj_complement[j] |= (1 << i)

    # 3. Solve Max Clique using Branch and Bound with Greedy Coloring
    # We represent the set of candidate vertices as a bitmask (Python int).
    # Since N_S = 330, we need big integers. Python handles this natively.
    
    best_clique_size = 0
    best_clique = []
    
    # Precompute degrees in the complement graph for initial ordering
    degrees = [adj_complement[i].bit_count() for i in range(N_S)]
    
    # Initial greedy solution to set a lower bound
    def greedy_clique():
        # Pick nodes with high degree in the remaining candidate set
        candidates = (1 << N_S) - 1
        clique = []
        while candidates:
            # Find node with max degree in candidates
            # Iterate through bits
            # Optimization: We can just pick the first one, or sort by degree.
            # For speed, let's pick a random one or the one with highest global degree among candidates.
            # Let's use a simple heuristic: pick the candidate with the highest degree in the subgraph induced by candidates.
            best_v = -1
            max_deg = -1
            c = candidates
            while c:
                # Extract lowest set bit
                lsb = c & -c
                v = lsb.bit_length() - 1
                # Calculate degree of v in candidates
                deg = (adj_complement[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                c -= lsb
                
            if best_v == -1:
                break
                
            clique.append(best_v)
            # Remove best_v and its non-neighbors from candidates
            # New candidates must be neighbors of best_v in the complement graph
            # AND must be in the current candidates
            # AND must not be best_v itself
            candidates &= adj_complement[best_v]
            candidates &= ~(1 << best_v)
            
        return clique

    init_clique = greedy_clique()
    best_clique_size = len(init_clique)
    best_clique = list(init_clique)
    
    # Greedy Coloring for Upper Bound
    # We color the subgraph induced by the candidate set in the COMPLEMENT graph?
    # No, the standard Max Clique B&B uses coloring on the graph where we are finding the clique.
    # Wait, if we find Max Clique in G (complement of conflict), we color G.
    # The number of colors is an upper bound on the clique size.
    
    def greedy_color(candidates):
        # candidates is a bitmask of nodes
        # Returns max_color + 1
        # We need to order the nodes. A good ordering is by degree in the induced subgraph.
        # However, extracting and sorting 330 nodes at every node in B&B might be slow.
        # Let's use a simpler ordering: natural order, or pre-computed global degree order.
        # Let's try ordering by global degree descending.
        
        # Get list of nodes in candidates
        nodes = []
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            nodes.append(v)
            c -= lsb
            
        # Sort by global degree descending (pre-computed)
        nodes.sort(key=lambda x: -degrees[x])
        
        colors = {}
        max_color = 0
        for v in nodes:
            # Find smallest color not used by neighbors in the colored set
            # Neighbors of v in the induced subgraph: adj_complement[v] & candidates
            # We only care about neighbors that have already been colored
            used_colors = set()
            for u, c_u in colors.items():
                # Check if u is a neighbor of v in G
                if (adj_complement[v] >> u) & 1:
                    used_colors.add(c_u)
            
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
                
        return max_color + 1

    def branch_and_bound(candidates, current_clique):
        nonlocal best_clique_size, best_clique
        
        if time.time() > deadline - 5.0:
            return
            
        if not candidates:
            if len(current_clique) > best_clique_size:
                best_clique_size = len(current_clique)
                best_clique = list(current_clique)
            return
            
        # Upper bound
        # If current size + upper bound <= best, prune
        # We need a fast upper bound.
        # Greedy coloring is O(N^2) roughly. With N=330, it might be okay if called sparingly.
        # But B&B can have many nodes.
        # Let's use a simpler bound first: len(candidates)
        if len(current_clique) + candidates.bit_count() <= best_clique_size:
            return
            
        # Try coloring bound if the simple bound is tight
        # To save time, we can skip coloring if the gap is large, or always do it?
        # Let's do coloring.
        
        # Optimization: If candidates is small, just do it.
        # The coloring function above is a bit slow due to Python loops.
        # Let's optimize the coloring by using a bitset for colors?
        # No, let's stick to the list approach but ensure it's efficient.
        
        ub = greedy_color(candidates)
        if len(current_clique) + ub <= best_clique_size:
            return
            
        # Choose a pivot
        # Heuristic: node with highest degree in the candidate subgraph
        pivot = -1
        max_deg = -1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            deg = (adj_complement[v] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = v
            c -= lsb
            
        if pivot == -1:
            return
            
        # Branch 1: Include pivot
        new_clique = current_clique + [pivot]
        # New candidates: neighbors of pivot in G that are in candidates
        new_candidates = candidates & adj_complement[pivot]
        new_candidates &= ~(1 << pivot)
        
        branch_and_bound(new_candidates, new_clique)
        
        # Branch 2: Exclude pivot
        # New candidates: candidates without pivot
        new_candidates_excl = candidates & ~(1 << pivot)
        
        branch_and_bound(new_candidates_excl, current_clique)

    # Initial candidates: all nodes
    initial_candidates = (1 << N_S) - 1
    
    # We can order the initial branching by picking nodes with high degree first?
    # The B&B pivot selection handles this dynamically.
    
    branch_and_bound(initial_candidates, [])
    
    # 4. Construct Vectors
    vectors = []
    for s_idx in best_clique:
        support = supports[s_idx]
        # For each support, we can add all 16 sign combinations
        # The vectors are orthogonal to each other within the same support?
        # v1 = (1, 1, 1, 1)
        # v2 = (1, 1, 1, -1)
        # dot = 1+1+1-1 = 2. Norm^2 = 4. Distance^2 = 2*4 - 2*2 = 4. Norm^2 = 4.
        # So min distance = max norm. This is valid.
        
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
            
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors

