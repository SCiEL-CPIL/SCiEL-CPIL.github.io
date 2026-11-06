import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 20
    
    # State variables
    adj = [0] * n
    # cc[i][j][k] = number of common neighbors of i, j, k
    # Using a flat list for speed: index = (i*n + j)*n + k
    cc = [0] * (n * n * n)
    
    def idx(i, j, k):
        return (i * n + j) * n + k

    def update_cc_add(u, v):
        # Update cc for triples involving u or v
        # Triples {u, a, b}: if v in N(a) and N(b), then v becomes common neighbor of {u,a,b}
        for a in range(n):
            if a == u or a == v:
                continue
            if adj[a] & (1 << v):
                for b in range(a + 1, n):
                    if b == u or b == v:
                        continue
                    if adj[b] & (1 << v):
                        # Increment all permutations of (u, a, b)
                        cc[idx(u, a, b)] += 1
                        cc[idx(u, b, a)] += 1
                        cc[idx(a, u, b)] += 1
                        cc[idx(a, b, u)] += 1
                        cc[idx(b, u, a)] += 1
                        cc[idx(b, a, u)] += 1

        # Triples {v, a, b}: if u in N(a) and N(b), then u becomes common neighbor of {v,a,b}
        for a in range(n):
            if a == u or a == v:
                continue
            if adj[a] & (1 << u):
                for b in range(a + 1, n):
                    if b == u or b == v:
                        continue
                    if adj[b] & (1 << u):
                        cc[idx(v, a, b)] += 1
                        cc[idx(v, b, a)] += 1
                        cc[idx(a, v, b)] += 1
                        cc[idx(a, b, v)] += 1
                        cc[idx(b, v, a)] += 1
                        cc[idx(b, a, v)] += 1

    def update_cc_remove(u, v):
        for a in range(n):
            if a == u or a == v:
                continue
            if adj[a] & (1 << v):
                for b in range(a + 1, n):
                    if b == u or b == v:
                        continue
                    if adj[b] & (1 << v):
                        cc[idx(u, a, b)] -= 1
                        cc[idx(u, b, a)] -= 1
                        cc[idx(a, u, b)] -= 1
                        cc[idx(a, b, u)] -= 1
                        cc[idx(b, u, a)] -= 1
                        cc[idx(b, a, u)] -= 1

        for a in range(n):
            if a == u or a == v:
                continue
            if adj[a] & (1 << u):
                for b in range(a + 1, n):
                    if b == u or b == v:
                        continue
                    if adj[b] & (1 << u):
                        cc[idx(v, a, b)] -= 1
                        cc[idx(v, b, a)] -= 1
                        cc[idx(a, v, b)] -= 1
                        cc[idx(a, b, v)] -= 1
                        cc[idx(b, v, a)] -= 1
                        cc[idx(b, a, v)] -= 1

    def is_valid_add(u, v):
        if adj[u] & (1 << v):
            return False
        
        # Check triples containing u: {u, a, b}
        # If v is in N(a) and N(b), then v is added to N({u, a, b}).
        # We need old cc[u][a][b] <= 1.
        for a in range(n):
            if a == u or a == v:
                continue
            if adj[a] & (1 << v):
                for b in range(a + 1, n):
                    if b == u or b == v:
                        continue
                    if adj[b] & (1 << v):
                        if cc[idx(u, a, b)] > 1:
                            return False
        
        # Check triples containing v: {v, a, b}
        # If u is in N(a) and N(b), then u is added to N({v, a, b}).
        # We need old cc[v][a][b] <= 1.
        for a in range(n):
            if a == u or a == v:
                continue
            if adj[a] & (1 << u):
                for b in range(a + 1, n):
                    if b == u or b == v:
                        continue
                    if adj[b] & (1 << u):
                        if cc[idx(v, a, b)] > 1:
                            return False
        
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        update_cc_add(u, v)

    def remove_edge(u, v):
        update_cc_remove(u, v)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges():
        return sum(x.bit_count() for x in adj) // 2

    def local_search(adj, cc):
        current_adj = adj[:]
        current_cc = cc[:]
        current_edges = count_edges()
        best_adj = current_adj[:]
        best_cc = current_cc[:]
        best_edges = current_edges
        
        # Phase 1: Greedy Add
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if time.time() > deadline:
                    break
                if is_valid_add(u, v):
                    add_edge(u, v)
                    current_edges += 1
                    improved = True
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                        best_cc = current_cc[:]
        
        # Phase 2: Edge Swaps (1 in, 2 out)
        steps = 0
        max_steps = 500
        while steps < max_steps:
            if time.time() > deadline:
                break
            steps += 1
            
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
            if not edges:
                break
            random.shuffle(edges)
            
            found_swap = False
            for u, v in edges:
                if time.time() > deadline:
                    break
                if found_swap:
                    break
                
                remove_edge(u, v)
                
                added = []
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j)) and (i != u or j != v) and (i != v or j != u)]
                random.shuffle(missing)
                
                for a, b in missing:
                    if time.time() > deadline:
                        break
                    if is_valid_add(a, b):
                        add_edge(a, b)
                        added.append((a, b))
                        if len(added) == 2:
                            break
                
                if len(added) == 2:
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                        best_cc = current_cc[:]
                    found_swap = True
                elif len(added) == 1:
                    current_edges += 0
                    found_swap = True
                else:
                    add_edge(u, v)
                    
        return best_adj, best_cc, best_edges

    best_adj = [0] * n
    best_cc = [0] * (n * n * n)
    best_edges = 0
    
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Reset state
        adj = [0] * n
        cc = [0] * (n * n * n)
        
        # Greedy construction
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.seed(i)
        random.shuffle(pairs)
        
        for u, v in pairs:
            if is_valid_add(u, v):
                add_edge(u, v)
        
        # Local search
        adj, cc, edges = local_search(adj, cc)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
            best_cc = cc

    # Final polish
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            adj, cc, edges = local_search(best_adj, best_cc)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj
                best_cc = cc
            else:
                break

    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
