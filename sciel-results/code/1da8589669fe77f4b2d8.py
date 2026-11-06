import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Precompute all 8-vertex subsets
    all_8_subsets = list(combinations(range(n), 8))
    
    # Precompute all 4-subsets of each 8-vertex subset for Q3 checking
    # For each 8-subset, store the list of (side_a_tuple, side_b_tuple) pairs
    # where side_a and side_b are complementary 4-subsets
    subset_bipartitions = {}
    for subset in all_8_subsets:
        biparts = []
        for side_a in combinations(subset, 4):
            side_a_set = set(side_a)
            side_b = tuple(v for v in subset if v not in side_a_set)
            biparts.append((side_a, side_b))
        subset_bipartitions[subset] = biparts
    
    def has_q3_fast(vertices, edge_set):
        """
        Check if the 8 vertices contain a Q3 subgraph.
        vertices: tuple of 8 unique integers.
        edge_set: set of tuples (u, v) with u < v.
        """
        biparts = subset_bipartitions.get(vertices)
        if biparts is None:
            # Compute on the fly
            biparts = []
            for side_a in combinations(vertices, 4):
                side_a_set = set(side_a)
                side_b = tuple(v for v in vertices if v not in side_a_set)
                biparts.append((side_a, side_b))
        
        for side_a, side_b in biparts:
            # Build non-edge matrix
            non_edge = [[False]*4 for _ in range(4)]
            for i, a in enumerate(side_a):
                for j, b in enumerate(side_b):
                    u, v = (a, b) if a < b else (b, a)
                    if (u, v) not in edge_set:
                        non_edge[i][j] = True
            
            # Check for perfect matching in non_edge
            # Iterate over permutations of 0..3
            for p in permutations(range(4)):
                if all(non_edge[i][p[i]] for i in range(4)):
                    return True
        return False
    
    def find_q3_edge(vertices, edge_set):
        """
        Find a Q3 subgraph on the 8 vertices and return one of its edges.
        Returns None if no Q3 exists.
        """
        biparts = subset_bipartitions.get(vertices)
        if biparts is None:
            biparts = []
            for side_a in combinations(vertices, 4):
                side_a_set = set(side_a)
                side_b = tuple(v for v in vertices if v not in side_a_set)
                biparts.append((side_a, side_b))
        
        for side_a, side_b in biparts:
            non_edge = [[False]*4 for _ in range(4)]
            for i, a in enumerate(side_a):
                for j, b in enumerate(side_b):
                    u, v = (a, b) if a < b else (b, a)
                    if (u, v) not in edge_set:
                        non_edge[i][j] = True
            
            for p in permutations(range(4)):
                if all(non_edge[i][p[i]] for i in range(4)):
                    # Found a Q3. Return one of its edges.
                    # The edges are (side_a[i], side_b[j]) where j != p[i]
                    for i in range(4):
                        for j in range(4):
                            if j != p[i]:
                                a = side_a[i]
                                b = side_b[j]
                                u, v = (a, b) if a < b else (b, a)
                                return (u, v)
        return None
    
    def edge_set_to_list(edge_set):
        return [[u, v] for (u, v) in sorted(edge_set)]
    
    random.seed(42)
    
    best_edges = set()
    best_count = 0
    
    # Strategy: Multiple restarts with random initialization
    # Then local search to remove Q3s and add edges
    
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Choose a starting density
        p = 0.6 + 0.04 * restart
        
        current_edges = set()
        for e in all_edges:
            if random.random() < p:
                current_edges.add(e)
        
        # Phase 1: Remove edges to make Q3-free
        max_removals = 50
        for _ in range(max_removals):
            if time.time() > deadline - 10:
                break
            
            # Find a Q3
            q3_found = None
            for subset in all_8_subsets:
                if has_q3_fast(subset, current_edges):
                    q3_found = subset
                    break
            
            if q3_found is None:
                break
            
            # Find an edge in the Q3 to remove
            e_to_remove = find_q3_edge(q3_found, current_edges)
            if e_to_remove is not None:
                current_edges.discard(e_to_remove)
            else:
                break
        
        # Phase 2: Local search - add edges
        missing_edges = [e for e in all_edges if e not in current_edges]
        random.shuffle(missing_edges)
        
        for e in missing_edges:
            if time.time() > deadline - 5:
                break
            
            a, b = e
            # Check if adding (a,b) creates a Q3
            # Only check 8-vertex subsets containing both a and b
            new_edges = current_edges | {e}
            is_valid = True
            
            other_vertices = [v for v in range(n) if v != a and v != b]
            for subset6 in combinations(other_vertices, 6):
                subset8 = tuple(sorted((a, b) + tuple(subset6)))
                if has_q3_fast(subset8, new_edges):
                    is_valid = False
                    break
            
            if is_valid:
                current_edges = new_edges
        
        # Update best
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
    
    return edge_set_to_list(best_edges)
