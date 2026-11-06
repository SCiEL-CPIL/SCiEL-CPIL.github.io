import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute Q3 structure
    # Q3 vertices: 0..7, edges where Hamming distance = 1
    q3_edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            d = i ^ j
            if d & (d - 1) == 0:
                q3_edges.append((i, j))
    
    # Precompute all 8-subsets
    subsets_8 = list(itertools.combinations(range(n), 8))
    
    def has_q3(adj_masks):
        for s in subsets_8:
            # Check if induced subgraph on s contains Q3 as subgraph
            # Use DFS with pruning
            s_list = list(s)
            s_set = set(s_list)
            
            # Quick rejection: need at least 12 edges
            ec = 0
            for v in s_list:
                for u in s_list:
                    if u > v and (adj_masks[v] >> u) & 1:
                        ec += 1
            if ec < 12:
                continue
            
            # DFS: assign Q3 vertices 0..7 to distinct vertices in s
            assign = [-1] * 8
            used = [False] * n
            # Order Q3 vertices to improve pruning: start with high-degree in Q3 (all are 3)
            # Just use natural order
            
            def dfs(idx):
                if idx == 8:
                    for (i, j) in q3_edges:
                        if not ((adj_masks[assign[i]] >> assign[j]) & 1):
                            return False
                    return True
                
                for v in s_list:
                    if used[v]:
                        continue
                    # Check compatibility with already-assigned neighbors
                    ok = True
                    for (i, j) in q3_edges:
                        if i == idx and assign[j] != -1:
                            if not ((adj_masks[v] >> assign[j]) & 1):
                                ok = False
                                break
                        elif j == idx and assign[i] != -1:
                            if not ((adj_masks[v] >> assign[i]) & 1):
                                ok = False
                                break
                    if not ok:
                        continue
                    
                    assign[idx] = v
                    used[v] = True
                    if dfs(idx + 1):
                        return True
                    assign[idx] = -1
                    used[v] = False
                return False
            
            if dfs(0):
                return True
        
        return False
    
    best_edges = []
    best_count = 0
    rng = random.Random(42)
    
    for restart in range(30):
        if time.time() > deadline - 20:
            break
        
        p = rng.uniform(0.3, 0.5)
        current_edges = set()
        for e in all_edges:
            if rng.random() < p:
                current_edges.add(e)
        
        adj_masks = [0] * n
        for u, v in current_edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        
        # Remove Q3s
        guard = 0
        while guard < 500 and has_q3(adj_masks):
            if time.time() > deadline - 15:
                break
            if not current_edges:
                break
            e = rng.choice(list(current_edges))
            u, v = e
            current_edges.remove(e)
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            guard += 1
        
        # Hill climb
        missing = [e for e in all_edges if e not in current_edges]
        rng.shuffle(missing)
        for e in missing:
            if time.time() > deadline - 15:
                break
            u, v = e
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            if not has_q3(adj_masks):
                current_edges.add(e)
            else:
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
    
    return [[u, v] for u, v in best_edges]
