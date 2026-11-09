import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial graph: K_{3,13}
    # U = {0, 1, 2}, V = {3, ..., 15}
    edges = set()
    for u in range(3):
        for v in range(3, 16):
            edges.add((u, v))
    
    # Helper to check if a set of 8 vertices contains Q3 as a subgraph
    def contains_q3(vertices):
        if len(vertices) != 8:
            return False
        
        # Vertices are a tuple/list of 8 ints
        # We check if there exists a bipartition A, B of size 4 such that
        # the missing edges between A and B contain a perfect matching.
        # If so, the present edges contain a 3-regular bipartite subgraph (Q3).
        
        # Precompute adjacency for these 8 vertices?
        # It's faster to just check (u,v) in edges directly.
        
        # Generate all 70 partitions of 8 vertices into two sets of 4
        # We can represent the set of 8 vertices as a list
        v_list = list(vertices)
        
        # To speed up, we can use the fact that if a perfect matching of missing edges exists,
        # we are good.
        
        # Iterate over combinations of 4 vertices to form A
        for i in range(8):
            for j in range(i+1, 8):
                for k in range(j+1, 8):
                    for l in range(k+1, 8):
                        A = {v_list[i], v_list[j], v_list[k], v_list[l]}
                        B = set(v_list) - A
                        
                        # Check for perfect matching in missing edges between A and B
                        # Missing edge (a,b) exists if (a,b) NOT in edges and (b,a) NOT in edges
                        # We need to match each a in A to a unique b in B such that (a,b) is missing.
                        
                        a_list = list(A)
                        b_list = list(B)
                        
                        # Try all 24 permutations of B
                        found_matching = False
                        for perm in itertools.permutations(b_list):
                            match = True
                            for idx in range(4):
                                a = a_list[idx]
                                b = perm[idx]
                                # Check if (a,b) is a missing edge
                                if (a, b) in edges or (b, a) in edges:
                                    match = False
                                    break
                            if match:
                                found_matching = True
                                break
                        if found_matching:
                            return True
        return False

    # Greedy addition
    all_pairs = set(itertools.combinations(range(16), 2))
    non_edges = list(all_pairs - edges)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        current_non_edges = [e for e in non_edges if e not in edges]
        for e in current_non_edges:
            if time.time() > deadline:
                break
            u, v = e
            remaining = [x for x in range(16) if x != u and x != v]
            safe = True
            edges.add((u, v))
            
            for combo in itertools.combinations(remaining, 6):
                verts = tuple([u, v] + list(combo))
                if contains_q3(verts):
                    safe = False
                    break
            
            if safe:
                changed = True
            else:
                edges.remove((u, v))

    return [list(e) for e in edges]
