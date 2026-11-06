import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_edges_count = 0
    best_edge_list = []
    
    def creates_q3(edge_set, neighbors, u, v):
        """
        Checks if adding edge (u, v) creates a Q3.
        edge_set contains sorted tuples (min, max).
        """
        A_cands = neighbors[u] - {v}
        B_cands = neighbors[v] - {u}
        
        if len(A_cands) < 3 or len(B_cands) < 3:
            return False
            
        # Helper to check if edge exists
        def has_edge(x, y):
            if x < y:
                return (x, y) in edge_set
            else:
                return (y, x) in edge_set

        for B in itertools.combinations(B_cands, 3):
            B_set = set(B)
            count_u_in_B = 0
            for b in B_set:
                if has_edge(u, b):
                    count_u_in_B += 1
            if count_u_in_B != 2:
                continue
                
            for A in itertools.combinations(A_cands, 3):
                A_set = set(A)
                count_v_in_A = 0
                for a in A_set:
                    if has_edge(v, a):
                        count_v_in_A += 1
                if count_v_in_A != 2:
                    continue
                
                U_set = A_set | {u}
                V_set = B_set | {v}
                
                # Check 3-regularity for A and B vertices
                # u and v are already known to be degree 3 in this subgraph
                valid = True
                for a in A_set:
                    deg_a = 0
                    for w in V_set:
                        if has_edge(a, w):
                            deg_a += 1
                    if deg_a != 3:
                        valid = False
                        break
                if not valid:
                    continue
                    
                for b in B_set:
                    deg_b = 0
                    for w in U_set:
                        if has_edge(b, w):
                            deg_b += 1
                    if deg_b != 3:
                        valid = False
                        break
                if not valid:
                    continue
                
                return True
                
        return False

    for restart in range(50):
        if time.time() > deadline:
            break
            
        edges = set()
        neighbors = [set() for _ in range(16)]
        
        # Add K3,13 edges
        for u in range(3):
            for v in range(3, 16):
                e = (u, v) if u < v else (v, u)
                edges.add(e)
                neighbors[u].add(v)
                neighbors[v].add(u)
        
        candidates = []
        for u in range(16):
            for v in range(u + 1, 16):
                if (u, v) not in edges:
                    candidates.append((u, v))
        
        random.shuffle(candidates)
        
        while candidates:
            if time.time() > deadline:
                break
                
            idx = random.randint(0, len(candidates) - 1)
            u, v = candidates[idx]
            candidates.pop(idx)
            
            if not creates_q3(edges, neighbors, u, v):
                edges.add((u, v))
                neighbors[u].add(v)
                neighbors[v].add(u)
                
        if len(edges) > best_edges_count:
            best_edges_count = len(edges)
            best_edge_list = list(edges)
            
    return [[u, v] for u, v in best_edge_list]
