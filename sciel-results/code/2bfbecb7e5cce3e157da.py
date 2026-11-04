import numpy as np
import itertools
import time
import random

D = 11

def construct():
    deadline = time.time() + 115
    
    # Generate all 4-subsets of range(11)
    subsets = list(itertools.combinations(range(D), 4))
    n = len(subsets)  # 330
    
    # Convert subsets to bitmasks
    masks = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(subsets):
        m = 0
        for pos in s:
            m |= (1 << pos)
        masks[i] = m
    
    # Build conflict matrix: conflict[i,j] = True if |S_i ∩ S_j| >= 3
    # Use popcount lookup table for efficiency
    popcount_table = np.zeros(2048, dtype=np.int32)
    for i in range(2048):
        popcount_table[i] = bin(i).count('1')
    
    # Compute pairwise intersections
    m1 = masks[:, np.newaxis]
    m2 = masks[np.newaxis, :]
    and_matrix = m1 & m2
    popcount_matrix = popcount_table[and_matrix]
    
    # Conflict if intersection size >= 3 (incompatible)
    conflict = (popcount_matrix >= 3)  # (n, n) boolean
    np.fill_diagonal(conflict, False)  # No self-conflict
    
    # Adjacency list for conflict graph
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if conflict[i, j]:
                adj[i].add(j)
                adj[j].add(i)
    
    # Greedy coloring for upper bound
    # Returns a list of color classes; the number of colors is an upper bound on MIS size
    def greedy_coloring(vertices):
        """Greedy coloring of the induced subgraph on vertices.
        Returns list of color classes (each a set of vertices)."""
        if not vertices:
            return []
        
        remaining = set(vertices)
        color_classes = []
        
        while remaining:
            # Greedy: pick vertices that can share a color (independent set)
            color_class = set()
            # Sort by degree for better coloring
            sorted_verts = sorted(remaining, key=lambda v: len(adj[v] & remaining), reverse=True)
            
            for v in sorted_verts:
                if v not in remaining:
                    continue
                # Check if v can join current color class (no conflicts with members)
                can_join = True
                for u in color_class:
                    if conflict[v, u]:
                        can_join = False
                        break
                if can_join:
                    color_class.add(v)
                    remaining.discard(v)
            
            if color_class:
                color_classes.append(color_class)
        
        return color_classes
    
    # Get upper bound from coloring
    def coloring_bound(vertices):
        """Upper bound on MIS size in induced subgraph on vertices."""
        if not vertices:
            return 0
        color_classes = greedy_coloring(vertices)
        return len(color_classes)
    
    # Branch and bound for Maximum Independent Set
    def branch_and_bound(vertices, current_size, best_size, best_solution, time_limit):
        """Recursive B&B. vertices: set of candidate vertices.
        Returns (best_size, best_solution)."""
        start_time = time.time()
        
        if not vertices:
            return current_size, best_solution
        
        # Upper bound: current_size + coloring_bound(vertices)
        ub = current_size + coloring_bound(vertices)
        if ub <= best_size:
            return best_size, best_solution
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in the subgraph
        vert_list = list(vertices)
        if not vert_list:
            return current_size, best_solution
        
        # Score by degree within subgraph
        def degree_in_subgraph(v):
            return len(adj[v] & vertices)
        
        # Pick vertex with max degree (or min for faster pruning - try both)
        v = max(vert_list, key=degree_in_subgraph)
        
        # Branch 1: Include v
        # Remove v and all its neighbors from candidates
        candidates_without_v = vertices - {v} - adj[v]
        new_size = current_size + 1
        new_solution = set(best_solution)
        new_solution.add(v)
        
        if time.time() < time_limit:
            best_size, best_solution = branch_and_bound(
                candidates_without_v, new_size, best_size, new_solution, time_limit
            )
        
        # Branch 2: Exclude v
        candidates_excluding_v = vertices - {v}
        if time.time() < time_limit:
            best_size, best_solution = branch_and_bound(
                candidates_excluding_v, current_size, best_size, best_solution, time_limit
            )
        
        return best_size, best_solution
    
    # Greedy initialization
    def greedy_mis():
        """Greedy MIS: repeatedly pick vertex with min degree in remaining."""
        remaining = set(range(n))
        selected = set()
        
        while remaining:
            if not remaining:
                break
            # Pick vertex with minimum degree in remaining subgraph
            min_deg = float('inf')
            min_v = None
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            
            if min_v is None:
                break
            
            selected.add(min_v)
            remaining.discard(min_v)
            # Remove neighbors
            for u in list(remaining):
                if conflict[min_v, u]:
                    remaining.discard(u)
        
        return selected
    
    # Get initial greedy solution
    initial_solution = greedy_mis()
    best_size = len(initial_solution)
    best_solution = set(initial_solution)
    
    # Run branch and bound with time limit
    bnb_time_limit = min(60, deadline - 20)
    if time.time() < deadline - 25:
        best_size, best_solution = branch_and_bound(
            set(range(n)), 0, best_size, best_solution, time.time() + bnb_time_limit
        )
    
    # Local search refinement: try to improve by swaps
    def local_search_improve(solution, time_limit):
        """Try to improve solution via 1-swap moves."""
        current = set(solution)
        current_size = len(current)
        start = time.time()
        
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            current_list = list(current)
            for i in range(len(current_list)):
                if time.time() > deadline - 10:
                    return current
                # Try removing current_list[i]
                candidate = current - {current_list[i]}
                # Try adding any node not in candidate
                for j in range(n):
                    if j not in candidate:
                        new_candidate = candidate | {j}
                        # Check validity
                        valid = True
                        for a in new_candidate:
                            for b in new_candidate:
                                if a != b and conflict[a, b]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            if len(new_candidate) > len(current):
                                current = new_candidate
                                current_size = len(current)
                                improved = True
                                current_list = list(current)
                                break
        return current
    
    # Refine with local search if time permits
    if time.time() < deadline - 15:
        refined = local_search_improve(best_solution, time_limit=10)
        if len(refined) > len(best_solution):
            best_solution = refined
            best_size = len(refined)
    
    # Build vectors from best_solution
    vectors = []
    for idx in best_solution:
        s = subsets[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
