import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    # Generate all D_11 vectors with squared norm 4
    candidates = []
    
    # Type 1: One ±2, rest 0 → 2*11 = 22 vectors
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four ±1's, rest 0 → C(11,4) * 2^4 = 330 * 16 = 5280
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Build conflict graph: edge i-j means M[i,j] > 2 (incompatible)
    # We want maximum independent set in conflict graph = maximum clique in compatibility graph
    # Conflict: M[i,j] > 2
    conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Tomita's branch-and-bound algorithm for maximum clique
    # Work on the compatibility graph (edges = compatible pairs)
    compat = (1 - conflict).astype(np.uint8)
    np.fill_diagonal(compat, 0)
    
    # Convert to adjacency list for efficiency
    # adj[i] = set of j such that compat[i,j] == 1
    adj = []
    for i in range(N):
        adj.append(np.where(compat[i] == 1)[0])
    
    # Greedy coloring for upper bound
    def greedy_color(order):
        """Color vertices in given order, return color count."""
        colors = {}
        for v in order:
            # Find smallest available color
            used = set()
            for u in adj[v]:
                if u in colors:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
        return max(colors.values()) if colors else 0
    
    # Tomita's MCQ algorithm (Maximum Clique with Coloring)
    best_size = 0
    best_clique = []
    
    def expand(P, C, current_clique):
        """Branch and bound: P = candidate set, C = colored order, current_clique = current path."""
        nonlocal best_size, best_clique
        
        if not P:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = current_clique[:]
            return
        
        if time.time() > deadline:
            return
        
        # Get the colored order of P
        # C should be a list of (vertex, color) pairs, sorted by color
        # Reorder P by color for the branching
        
        # Simple approach: process vertices in order of decreasing color
        # For each vertex v in P (in color order), branch on including v
        
        # Process from highest color to lowest
        for i in range(len(P) - 1, -1, -1):
            v = P[i]
            
            if len(current_clique) + i + 1 <= best_size:
                # Upper bound: even if we take all remaining vertices in this color class,
                # we can't beat best
                break
            
            # Recurse: include v
            # New candidate set: neighbors of v that are in P[0:i]
            new_P = [u for u in P[:i] if v in adj_set[u]]
            
            expand(new_P, None, current_clique + [v])
            
            if time.time() > deadline:
                return
    
    # Better implementation with proper coloring
    # Let's use a cleaner Tomita implementation
    
    # Convert adj to sets for fast lookup
    adj_set = [set(a) for a in adj]
    
    # Greedy coloring function that returns ordered list
    def color_sort(P):
        """Color vertices in P and return them sorted by color (ascending)."""
        if not P:
            return []
        
        P_list = list(P)
        colors = {}
        uncolored = set(P_list)
        
        while uncolored:
            # Take one vertex, assign smallest available color
            v = next(iter(uncolored))
            used = set()
            for u in adj_set[v]:
                if u in colors:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            uncolored.discard(v)
        
        # Sort by color
        sorted_P = sorted(P_list, key=lambda x: colors[x])
        return sorted_P
    
    # Tomita's algorithm
    def tomita_mcq(P, current_clique):
        """Maximum clique via branch and bound with coloring."""
        nonlocal best_size, best_clique
        
        if not P:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = current_clique[:]
            return
        
        if time.time() > deadline:
            return
        
        # Color and sort P
        sorted_P = color_sort(P)
        
        # Branch from last (highest color) to first
        for i in range(len(sorted_P) - 1, -1, -1):
            v = sorted_P[i]
            
            # Upper bound: len(current_clique) + i + 1
            if len(current_clique) + i + 1 <= best_size:
                break
            
            # Recurse with v included
            # New P: intersection of P[:i] and neighbors of v
            new_P = [u for u in sorted_P[:i] if u in adj_set[v]]
            
            tomita_mcq(new_P, current_clique + [v])
            
            if time.time() > deadline:
                return
    
    # Start with all vertices
    # Use a good initial ordering: sort by degree (descending)
    degrees = np.array([len(adj[i]) for i in range(N)])
    order = np.argsort(-degrees)
    
    # Initial greedy to get a good lower bound
    # Greedy clique: repeatedly pick vertex with max degree in remaining graph
    P = list(order)
    greedy_clique = []
    remaining = set(P)
    while remaining:
        # Pick vertex with max degree in remaining
        best_v = None
        best_deg = -1
        for v in remaining:
            deg = len(adj_set[v] & remaining)
            if deg > best_deg:
                best_deg = deg
                best_v = v
        if best_v is None:
            break
        greedy_clique.append(best_v)
        remaining.discard(best_v)
        # Remove non-neighbors
        non_neighbors = remaining - adj_set[best_v]
        remaining -= non_neighbors
    
    if len(greedy_clique) > best_size:
        best_size = len(greedy_clique)
        best_clique = greedy_clique[:]
    
    # Now run Tomita's algorithm
    tomita_mcq(list(order), [])
    
    if best_clique:
        result = candidates[best_clique].astype(np.int64)
        return result
    else:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
