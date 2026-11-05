import time
import random

def construct():
    deadline = time.time() + 140.0
    n = 18
    
    # Helper to check if adding edge (u, v) creates a K4
    def creates_k4(adj, u, v):
        # Check if there is an edge in the common neighborhood of u and v
        common = adj[u] & adj[v]
        # Check if common contains an edge
        # Iterate through vertices in common
        m = common
        while m:
            lb = m & (-m)
            x = lb.bit_length() - 1
            m ^= lb
            if adj[x] & common:
                return True
        return False

    # Helper to evaluate largest triangle-free induced subgraph
    def eval_max_tf(adj):
        best = 0
        # Iterate over all subsets? 2^18 is 262k. 
        # We can optimize by checking supersets or using B&B, but 262k is small.
        # However, we call this many times.
        # Let's stick to simple bitmask iteration for now, or use a faster check.
        # A set is triangle-free if for all u in S, (N(u) & S) is independent? No.
        # Just check all triangles in S.
        # If S has no triangles, it's triangle-free.
        # Fast check: 
        # For each v in S, if (adj[v] & S) has an edge, then S has a triangle.
        # Precompute edges in S?
        
        # Better: Iterate masks.
        for mask in range(1, 1 << n):
            # Check if mask is triangle-free
            is_tf = True
            m = mask
            while m:
                lb = m & (-m)
                v = lb.bit_length() - 1
                m ^= lb
                rest = m & adj[v]
                # Check if rest has an edge
                # rest is a subset of S \ {v}
                # If rest has an edge, triangle exists
                # Check edges in rest
                rm = rest
                while rm:
                    lb2 = rm & (-rm)
                    w = lb2.bit_length() - 1
                    rm ^= lb2
                    if adj[w] & rest:
                        is_tf = False
                        break
                if not is_tf:
                    break
            
            if is_tf:
                c = mask.bit_count()
                if c > best:
                    best = c
        return best

    best_score = n # Max possible is 18
    best_edges = []
    
    # Try a few random constructions
    # 1. Turan graph K_{6,6,6}
    adj_turan = [0] * n
    edges_turan = []
    for i in range(n):
        for j in range(i + 1, n):
            if i // 6 != j // 6:
                adj_turan[i] |= (1 << j)
                adj_turan[j] |= (1 << i)
                edges_turan.append([i, j])
    
    score_turan = eval_max_tf(adj_turan)
    if score_turan < best_score:
        best_score = score_turan
        best_edges = edges_turan.copy()
    
    # 2. Random Greedy K4-free
    for _ in range(50): # 50 restarts
        if time.time() > deadline - 10:
            break
            
        adj = [0] * n
        edges = []
        
        # Random order of edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Check if adding (u, v) creates K4
            if not creates_k4(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        
        score = eval_max_tf(adj)
        if score < best_score:
            best_score = score
            best_edges = edges.copy()
            
        # Local search: try to add more edges? 
        # The greedy already added all possible edges in that order.
        # But order matters. A different order might allow more edges.
        # We did random shuffle, so this covers some space.
        
        # Try to improve by removing edges that allow adding more?
        # Too complex. Random restarts should find good maximal graphs.
        
    return best_edges
