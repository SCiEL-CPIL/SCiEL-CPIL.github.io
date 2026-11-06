import time
import random
import itertools
import numpy as np

def construct():
    deadline = time.time() + 115
    n = 16
    edges = list(itertools.combinations(range(n), 2))
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    num_edges = len(edges)
    
    quads = list(itertools.combinations(range(n), 4))
    quad_edges = []
    for a, b, c, d in quads:
        e1 = edge_to_idx[(a, b)]
        e2 = edge_to_idx[(a, c)]
        e3 = edge_to_idx[(a, d)]
        e4 = edge_to_idx[(b, c)]
        e5 = edge_to_idx[(b, d)]
        e6 = edge_to_idx[(c, d)]
        quad_edges.append((e1, e2, e3, e4, e5, e6))
    num_quads = len(quads)
    
    edge_to_quads = [[] for _ in range(num_edges)]
    for qi, q_edges in enumerate(quad_edges):
        for ei in q_edges:
            edge_to_quads[ei].append(qi)
            
    def is_valid(colors):
        for qi in range(num_quads):
            q_edges = quad_edges[qi]
            c = {colors[e] for e in q_edges}
            if len(c) < 5:
                return False
        return True

    for k in [13, 14, 15]:
        colors = [ (i + j) % k for i, j in edges ]
        if is_valid(colors):
            return colors
        colors = [ (i * j) % k for i, j in edges ]
        if is_valid(colors):
            return colors

    def local_search(num_colors):
        colors = np.array([ (i + j) % num_colors for i, j in edges ], dtype=np.int32)
        
        distinct_counts = np.zeros(num_quads, dtype=np.int32)
        color_counts = np.zeros((num_quads, num_colors), dtype=np.int32)
        
        for qi in range(num_quads):
            q_edges = quad_edges[qi]
            for ei in q_edges:
                c = colors[ei]
                color_counts[qi, c] += 1
            distinct_counts[qi] = np.sum(color_counts[qi] > 0)
            
        invalid_count = np.sum(distinct_counts < 5)
        
        if invalid_count == 0:
            return colors.tolist()
            
        last_improvement = 0
        it = 0
        max_iters = 5000000
        
        while it < max_iters:
            it += 1
            if time.time() > deadline:
                return None
                
            if invalid_count == 0:
                return colors.tolist()
                
            qi = random.randint(0, num_quads - 1)
            if distinct_counts[qi] >= 5:
                continue
                
            q_edges = quad_edges[qi]
            ei = random.choice(q_edges)
            
            old_c = colors[ei]
            new_c = random.randint(0, num_colors - 1)
            if new_c == old_c:
                continue
                
            delta_invalid = 0
            d_changes = []
            
            for uqi in edge_to_quads[ei]:
                old_count_old = color_counts[uqi, old_c]
                old_count_new = color_counts[uqi, new_c]
                
                d_change = 0
                if old_count_old == 1:
                    d_change -= 1
                if old_count_new == 0:
                    d_change += 1
                d_changes.append((uqi, d_change))
                
                old_dc = distinct_counts[uqi]
                new_dc = old_dc + d_change
                
                was_invalid = old_dc < 5
                is_invalid = new_dc < 5
                
                if was_invalid and not is_invalid:
                    delta_invalid -= 1
                elif not was_invalid and is_invalid:
                    delta_invalid += 1

            if delta_invalid < 0:
                colors[ei] = new_c
                invalid_count += delta_invalid
                for uqi, d_change in d_changes:
                    color_counts[uqi, old_c] -= 1
                    color_counts[uqi, new_c] += 1
                    distinct_counts[uqi] += d_change
                last_improvement = it
            else:
                pass
            
            if it - last_improvement > 10000:
                for _ in range(10):
                    re = random.randint(0, num_edges - 1)
                    new_c = random.randint(0, num_colors - 1)
                    old_c = colors[re]
                    if new_c != old_c:
                        colors[re] = new_c
                        for uqi in edge_to_quads[re]:
                            color_counts[uqi, old_c] -= 1
                            color_counts[uqi, new_c] += 1
                            distinct_counts[uqi] = np.sum(color_counts[uqi] > 0)
                invalid_count = np.sum(distinct_counts < 5)
                last_improvement = it

        return None

    for k in [13, 14, 15]:
        res = local_search(k)
        if res:
            return res

    return [ (i + j) % 17 for i, j in edges ]
