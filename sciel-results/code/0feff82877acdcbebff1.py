import numpy as np
import random
import itertools
import time

def construct():
    n = 18
    random.seed(42)
    
    # Initialize with a random 3-regular graph structure
    # Start with a cycle
    edges = set()
    for i in range(n):
        edges.add((i, (i + 1) % n))
        
    # Add random edges to reach ~3-regular
    while len(edges) < 30:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v and (u, v) not in edges and (v, u) not in edges:
            edges.add((u, v))
            
    def creates_k4(edge_set, u, v):
        # Find common neighbors of u and v
        common = set()
        for (a, b) in edge_set:
            if a == u: common.add(b)
            elif b == u: common.add(a)
            if a == v: common.add(b)
            elif b == v: common.add(a)
            
        # Check if any pair of common neighbors is connected
        common_list = list(common)
        for i in range(len(common_list)):
            for j in range(i + 1, len(common_list)):
                w, x = common_list[i], common_list[j]
                if (w, x) in edge_set or (x, w) in edge_set:
                    return True
        return False

    def get_score(edge_set):
        # Precompute triangles
        triangles = []
        for u, v, w in itertools.combinations(range(n), 3):
            if ((u, v) in edge_set or (v, u) in edge_set) and \
               ((u, w) in edge_set or (w, u) in edge_set) and \
               ((v, w) in edge_set or (w, v) in edge_set):
                mask = (1 << u) | (1 << v) | (1 << w)
                triangles.append(mask)
                
        if not triangles:
            return n
            
        # Iterate subsets in decreasing size
        for k in range(n, 0, -1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for x in subset:
                    mask |= (1 << x)
                is_tf = True
                for tm in triangles:
                    if (mask & tm) == tm:
                        is_tf = False
                        break
                if is_tf:
                    return k
        return 0

    # Ensure initial graph is K4-free
    current_edges = set(edges)
    changed = True
    while changed:
        changed = False
        for quad in itertools.combinations(range(n), 4):
            u, v, w, x = quad
            if ((u,v) in current_edges or (v,u) in current_edges) and \
               ((u,w) in current_edges or (w,u) in current_edges) and \
               ((u,x) in current_edges or (x,u) in current_edges) and \
               ((v,w) in current_edges or (w,v) in current_edges) and \
               ((v,x) in current_edges or (x,v) in current_edges) and \
               ((w,x) in current_edges or (x,w) in current_edges):
                e = random.choice([(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)])
                if e in current_edges:
                    current_edges.remove(e)
                else:
                    current_edges.remove((e[1], e[0]))
                changed = True
                break
                
    current_score = get_score(current_edges)
    best_score = current_score
    best_edges = set(current_edges)
    
    start_time = time.time()
    
    while time.time() - start_time < 110:
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) in current_edges:
                    continue
                if creates_k4(current_edges, u, v):
                    continue
                
                new_edges = set(current_edges)
                new_edges.add((u, v))
                new_score = get_score(new_edges)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edges = new_edges
                    current_edges = new_edges
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Perturbation
            valid_moves = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in current_edges and not creates_k4(current_edges, u, v):
                        valid_moves.append((u, v))
            if valid_moves:
                u, v = random.choice(valid_moves)
                current_edges.add((u, v))
                current_score = get_score(current_edges)
            else:
                break

    result = []
    for e in best_edges:
        result.append([e[0], e[1]])
    return result
