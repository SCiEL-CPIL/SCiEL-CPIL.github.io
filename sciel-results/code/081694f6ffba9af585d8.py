def construct():
    import itertools
    import time
    import random
    
    start = time.time()
    deadline = start + 85
    
    def is_2_colorable(edges):
        n = 9
        # Build constraint: for each edge, not all three same
        # Use SAT-like backtracking
        edges_set = set(tuple(sorted(e)) for e in edges)
        
        def check(assignment, idx):
            if idx == n:
                return True
            for color in (0, 1):
                assignment[idx] = color
                # Check if any edge is now fully determined and monochromatic
                conflict = False
                for e in edges_set:
                    if idx in e:
                        others = [v for v in e if v != idx]
                        if all(v < idx for v in others):
                            if assignment[others[0]] == assignment[others[1]] == color:
                                conflict = True
                                break
                if not conflict:
                    if check(assignment, idx + 1):
                        return True
            return False
        
        assignment = [None] * n
        return check(assignment, 0)
    
    # Search for 6-edge 3-chromatic hypergraph
    all_triples = list(itertools.combinations(range(9), 3))
    random.seed(123)
    
    found_6 = None
    attempts = 0
    while time.time() < deadline - 5 and found_6 is None:
        if attempts % 1000 == 0 and attempts > 0:
            pass  # progress check
        candidate = random.sample(all_triples, 6)
        if not is_2_colorable(candidate):
            found_6 = candidate
        attempts += 1
    
    if found_6 is not None:
        edges = [list(e) for e in found_6]
    else:
        # Fall back to 7-edge Fano plane
        edges = [
            [0, 1, 2],
            [0, 3, 4],
            [0, 5, 6],
            [1, 3, 5],
            [1, 4, 6],
            [2, 3, 6],
            [2, 4, 5]
        ]
    
    # Find valid 3-coloring
    def find_3_coloring(edges, n=9):
        assignment = [None] * n
        edges_set = set(tuple(sorted(e)) for e in edges)
        def bt(idx):
            if idx == n:
                for e in edges_set:
                    if assignment[e[0]] == assignment[e[1]] == assignment[e[2]]:
                        return False
                return True
            for c in range(3):
                assignment[idx] = c
                ok = True
                for e in edges_set:
                    if idx in e:
                        others = [v for v in e if v != idx]
                        if all(v < idx for v in others):
                            if assignment[others[0]] == assignment[others[1]] == c:
                                ok = False
                                break
                if ok and bt(idx + 1):
                    return True
            return False
        if bt(0):
            return assignment
        return None
    
    colors = find_3_coloring(edges)
    if colors is None:
        colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    edges_out = [[int(e[0]), int(e[1]), int(e[2])] for e in edges]
    colors_out = [int(c) for c in colors]
    
    return {'edges': edges_out, 'colors': colors_out}
