import time
import random
import itertools

def construct():
    deadline = time.time() + 145
    n = 16
    edges = list(itertools.combinations(range(n), 2))
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    quads = list(itertools.combinations(range(n), 4))
    quad_edges = []
    for a, b, c, d in quads:
        e1 = edge_to_idx[(a, b)]
        e2 = edge_to_idx[(a, c)]
        e3 = edge_to_idx[(a, d)]
        e4 = edge_to_idx[(b, c)]
        e5 = edge_to_idx[(b, d)]
        e6 = edge_to_idx[(c, d)]
        quad_edges.append([e1, e2, e3, e4, e5, e6])
        
    edge_to_quads = [[] for _ in range(len(edges))]
    for qi, q_edges in enumerate(quad_edges):
        for ei in q_edges:
            edge_to_quads[ei].append(qi)
            
    def check_formula(formula, num_colors):
        colors = [formula(i, j) % num_colors for i, j in edges]
        # Validate
        for qi in range(len(quads)):
            q_edges = quad_edges[qi]
            c = {colors[e] for e in q_edges}
            if len(c) < 5:
                return False
        return colors

    # Try specific formulas
    for nc in [13, 14, 15, 16]:
        # i*j
        res = check_formula(lambda i, j: i * j, nc)
        if res: return res
        # i+j
        res = check_formula(lambda i, j: i + j, nc)
        if res: return res
        # i*j+i+j
        res = check_formula(lambda i, j: i * j + i + j, nc)
        if res: return res
        # i^2+j^2
        res = check_formula(lambda i, j: i*i + j*j, nc)
        if res: return res

    def fast_local_search(num_colors):
        colors = [random.randint(0, num_colors - 1) for _ in range(len(edges))]
        
        # Initialize quad color sets
        quad_sets = [set() for _ in range(len(quads))]
        for qi in range(len(quads)):
            q_edges = quad_edges[qi]
            for e in q_edges:
                quad_sets[qi].add(colors[e])
                
        invalid_set = set(qi for qi, s in enumerate(quad_sets) if len(s) < 5)
        
        for it in range(50000):
            if time.time() > deadline:
                return None
            if not invalid_set:
                return colors
                
            qi = random.choice(list(invalid_set))
            q_edges = quad_edges[qi]
            random.shuffle(q_edges)
            
            for ei in q_edges:
                old_c = colors[ei]
                # Try all colors
                for new_c in range(num_colors):
                    if new_c == old_c:
                        continue
                    
                    colors[ei] = new_c
                    
                    # Update quad sets
                    for uqi in edge_to_quads[ei]:
                        # Remove old_c if it's no longer present in this quad
                        # We need to know if old_c is still in the quad via other edges
                        # This is hard to check quickly without scanning.
                        # Alternative: Just rebuild the set for the 91 quads?
                        # 91 * 6 = 546 lookups. 50000 * 546 = 27M. Might be ok.
                        
                        # Let's try the rebuild approach for simplicity and correctness
                        new_set = set()
                        for e in quad_edges[uqi]:
                            new_set.add(colors[e])
                        quad_sets[uqi] = new_set
                        if len(new_set) < 5:
                            invalid_set.add(uqi)
                        else:
                            invalid_set.discard(uqi)
                            
                    # If qi is fixed, break
                    if len(quad_sets[qi]) >= 5:
                        break
                        
        return None

    # Try 13
    res = fast_local_search(13)
    if res: return res
    
    # Try 14
    res = fast_local_search(14)
    if res: return res
    
    # Try 15
    res = fast_local_search(15)
    if res: return res
    
    # Fallback 17
    return [ (i + j) % 17 for i, j in edges ]
