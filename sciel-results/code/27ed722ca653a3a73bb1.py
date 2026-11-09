import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    N = 16
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    adj_mask = [0] * N
    
    def set_edge(u, v, present):
        if present:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        else:
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
    
    def has_edge(u, v):
        return (adj_mask[u] >> v) & 1
    
    def check_q3_8(verts):
        """Check if 8 vertices contain a Q3 (exactly 12 edges + bipartite matching condition)."""
        # Quick edge count
        cnt = 0
        for i in range(8):
            for j in range(i+1, 8):
                if has_edge(verts[i], verts[j]):
                    cnt += 1
        if cnt != 12:
            return False
        
        # Fix first vertex in A
        v0 = verts[0]
        rest = verts[1:]
        for A_rest in itertools.combinations(rest, 3):
            A = (v0,) + A_rest
            A_set = set(A)
            B = tuple(v for v in verts if v not in A_set)
            
            # Check partial matching condition on missing edges
            rows_used = 0
            cols_used = 0
            ok = True
            for ia in range(4):
                a = A[ia]
                for ib in range(4):
                    b = B[ib]
                    if not has_edge(a, b):
                        if (rows_used >> ia) & 1 or (cols_used >> ib) & 1:
                            ok = False
                            break
                        rows_used |= (1 << ia)
                        cols_used |= (1 << ib)
                if not ok:
                    break
            if ok:
                return True
        return False
    
    def find_q3():
        """Find a Q3 in the graph. Returns 8-tuple or None."""
        order = list(range(N))
        random.shuffle(order)
        for combo in itertools.combinations(order, 8):
            if check_q3_8(combo):
                return combo
        return None
    
    # Phase 1: Initialize and prune
    edges_set = set()
    for u, v in all_pairs:
        if random.random() < 0.78:
            edges_set.add((u, v))
            set_edge(u, v, True)
    
    for _ in range(300):
        if time.time() > deadline - 50:
            break
        q3 = find_q3()
        if q3 is None:
            break
        # Remove random edge from the Q3
        q3_edges = []
        for i in range(8):
            for j in range(i+1, 8):
                if has_edge(q3[i], q3[j]):
                    q3_edges.append((q3[i], q3[j]))
        if q3_edges:
            e = random.choice(q3_edges)
            edges_set.discard(e)
            set_edge(e[0], e[1], False)
    
    best_count = len(edges_set)
    best_edges = set(edges_set)
    
    # Phase 2: Local search to add edges
    # For each non-edge, check if adding it creates a Q3
    # Optimize: only check 8-sets that contain the new edge
    non_edges_list = [(u, v) for u, v in all_pairs if (u, v) not in edges_set]
    random.shuffle(non_edges_list)
    
    for u, v in non_edges_list:
        if time.time() > deadline - 10:
            break
        
        # Temporarily add
        set_edge(u, v, True)
        edges_set.add((u, v))
        
        # Check if any 8-set containing u and v has a Q3
        # Choose 6 from the other 14 vertices
        others = [x for x in range(N) if x != u and x != v]
        safe = True
        for six in itertools.combinations(others, 6):
            verts = (u, v) + six
            if check_q3_8(verts):
                safe = False
                break
        
        if not safe:
            # Remove edge
            set_edge(u, v, False)
            edges_set.discard((u, v))
        else:
            if len(edges_set) > best_count:
                best_count = len(edges_set)
                best_edges = set(edges_set)
    
    # Restore best state
    for i in range(N):
        adj_mask[i] = 0
    for (u, v) in best_edges:
        set_edge(u, v, True)
    
    return [list(e) for e in best_edges]

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
