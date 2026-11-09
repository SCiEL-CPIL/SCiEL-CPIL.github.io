import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 20
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def init_adj():
        return [0] * n
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def edge_exists(adj, u, v):
        return (adj[u] >> v) & 1
    
    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges
    
    def can_add_fast(adj, u, v):
        if edge_exists(adj, u, v):
            return False
        c_y = adj[u] & ~(1 << v)
        c_x = adj[v] & ~(1 << u)
        y_cands = [i for i in range(n) if (c_y >> i) & 1]
        if len(y_cands) >= 2:
            for i in range(len(y_cands)):
                y1 = y_cands[i]
                for j in range(i + 1, len(y_cands)):
                    y2 = y_cands[j]
                    common = adj[y1] & adj[y2] & c_x
                    if common & (common - 1):
                        return False
        c_x2 = adj[u] & ~(1 << v)
        c_y2 = adj[v] & ~(1 << u)
        x_cands = [i for i in range(n) if (c_x2 >> i) & 1]
        if len(x_cands) >= 2:
            for i in range(len(x_cands)):
                x1 = x_cands[i]
                for j in range(i + 1, len(x_cands)):
                    x2 = x_cands[j]
                    common = adj[x1] & adj[x2] & c_y2
                    if common & (common - 1):
                        return False
        return True
    
    def greedy_fill(adj):
        for (u, v) in all_edges:
            if not edge_exists(adj, u, v):
                if can_add_fast(adj, u, v):
                    add_edge(adj, u, v)
    
    def two_opt(adj):
        """Try 2-edge swaps to improve."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            existing = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        existing.append((i, j))
                    mask ^= lsb
            
            if len(existing) < 2:
                break
            
            random.shuffle(existing)
            for idx in range(min(100, len(existing))):
                if time.time() > deadline - 5:
                    break
                e1 = existing[idx]
                e2_idx = random.randint(0, len(existing) - 1)
                if e2_idx == idx:
                    e2_idx = (e2_idx + 1) % len(existing)
                e2 = existing[e2_idx]
                
                # Remove both
                remove_edge(adj, e1[0], e1[1])
                remove_edge(adj, e2[0], e2[1])
                
                # Try to add 2 new edges
                added = []
                shuffled = all_edges[:]
                random.shuffle(shuffled)
                for (u, v) in shuffled:
                    if len(added) >= 2:
                        break
                    if edge_exists(adj, u, v):
                        continue
                    if (u, v) == e1 or (u, v) == e2 or (v, u) == e1 or (v, u) == e2:
                        continue
                    if can_add_fast(adj, u, v):
                        add_edge(adj, u, v)
                        added.append((u, v))
                
                if len(added) == 2:
                    new_count = count_edges(adj)
                    old_count = count_edges(adj) - 2 + 2  # same
                    # We removed 2 and added 2, so count is same
                    # But the structure changed, might allow further improvements
                    improved = True
                else:
                    # Revert
                    add_edge(adj, e1[0], e1[1])
                    add_edge(adj, e2[0], e2[1])
        
        greedy_fill(adj)
    
    best_adj = init_adj()
    best_count = 0
    random.seed(42)
    
    # Strategy 1: Structured starts
    # Circulant graphs
    for d in range(3, 12):
        adj = init_adj()
        for i in range(n):
            for k in range(1, d + 1):
                j = (i + k) % n
                if i < j:
                    add_edge(adj, i, j)
        greedy_fill(adj)
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
    
    # Strategy 2: Random starts with greedy fill
    for _ in range(30):
        if time.time() > deadline - 10:
            break
        adj = init_adj()
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        for (u, v) in shuffled:
            if can_add_fast(adj, u, v):
                add_edge(adj, u, v)
        greedy_fill(adj)
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
    
    # Strategy 3: Local search (2-opt) from best
    current_adj = best_adj[:]
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        two_opt(current_adj)
        cnt = count_edges(current_adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = current_adj[:]
    
    # Strategy 4: Perturbation + local search
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        adj = best_adj[:]
        # Remove k random edges
        k = random.randint(3, 10)
        existing = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    existing.append((i, j))
                mask ^= lsb
        random.shuffle(existing)
        for (u, v) in existing[:k]:
            remove_edge(adj, u, v)
        greedy_fill(adj)
        two_opt(adj)
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
    
    return adj_to_edges(best_adj)
