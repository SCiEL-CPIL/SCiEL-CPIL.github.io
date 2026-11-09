import time
import random

def construct():
    n = 40
    m = 401
    deadline = time.time() + 140.0  # Safety margin
    
    def calc_max_book(adj, edges):
        max_b = 0
        for u, v in edges:
            common = adj[u] & adj[v]
            cnt = common.bit_count()
            if cnt > max_b:
                max_b = cnt
        return max_b

    def init_random():
        adj = [0] * n
        edges = set()
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs[:m]:
            edges.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj, edges

    best_b = 100
    best_adj = None
    best_edges = None
    
    while time.time() < deadline - 5.0:
        adj, edges = init_random()
        current_b = calc_max_book(adj, edges)
        
        if current_b < best_b:
            best_b = current_b
            best_adj = adj[:]
            best_edges = set(edges)
            
        if best_b <= 7:
            break

        T = 1.0
        no_improve = 0
        max_no_improve = 3000
        
        while time.time() < deadline - 5.0:
            if len(edges) == 0: break
            
            # Pick edge to remove (biased towards high book)
            sample_size = min(10, len(edges))
            candidates = random.sample(list(edges), sample_size)
            u, v = max(candidates, key=lambda e: (adj[e[0]] & adj[e[1]]).bit_count())
            
            # Pick edge to add (biased towards low common neighbors)
            best_add = None
            best_common = 100
            for _ in range(30):
                x, y = random.randint(0, n-1), random.randint(0, n-1)
                if x == y: continue
                if x > y: x, y = y, x
                if (x, y) in edges: continue
                common = (adj[x] & adj[y]).bit_count()
                if common < best_common:
                    best_common = common
                    best_add = (x, y)
            
            if not best_add:
                continue
            
            x, y = best_add
            
            # Apply swap
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edges.remove((u, v))
            
            adj[x] |= (1 << y)
            adj[y] |= (1 << x)
            edges.add((x, y))
            
            new_b = calc_max_book(adj, edges)
            
            delta = new_b - current_b
            accept = False
            if delta < 0:
                accept = True
            else:
                if random.random() < pow(2.71828, -delta / T):
                    accept = True
            
            if accept:
                current_b = new_b
                if current_b < best_b:
                    best_b = current_b
                    best_adj = adj[:]
                    best_edges = set(edges)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                edges.remove((x, y))
                adj[x] &= ~(1 << y)
                adj[y] &= ~(1 << x)
                
                edges.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                no_improve += 1
                
            T *= 0.99
            if T < 0.05:
                T = 0.05
                
            if no_improve > max_no_improve:
                break

    return [[u, v] for u, v in best_edges]
