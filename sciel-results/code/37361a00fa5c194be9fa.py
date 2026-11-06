import random
import time
import numpy as np

def construct():
    deadline = time.time() + 115
    n = 16
    target_edges = 65
    
    def compute_score(adj, deg):
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    continue
                for w in range(v + 1, n):
                    if adj[u, w] and adj[v, w]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
        return max_sum
    
    # Try multiple random initializations and local searches
    best_score = float('inf')
    best_adj = None
    best_deg = None
    
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        random.seed(seed)
        
        # Start from K_{8,8} + random edge, or random near-bipartite
        adj = np.zeros((n, n), dtype=bool)
        
        # Create a random bipartite-like structure
        # Partition vertices into two sets
        perm = list(range(n))
        random.shuffle(perm)
        split = random.randint(6, 10)
        A = set(perm[:split])
        B = set(perm[split:])
        
        # Add all edges between A and B
        for u in A:
            for v in B:
                adj[u, v] = True
                adj[v, u] = True
        
        edge_count = len(A) * len(B)
        
        # Add remaining edges within A or B to reach target
        remaining = target_edges - edge_count
        if remaining > 0:
            # Add random edges within parts
            internal_edges = []
            for part in [A, B]:
                part_list = list(part)
                for i in range(len(part_list)):
                    for j in range(i + 1, len(part_list)):
                        u, v = part_list[i], part_list[j]
                        if not adj[u, v]:
                            internal_edges.append((u, v))
            random.shuffle(internal_edges)
            for u, v in internal_edges[:remaining]:
                adj[u, v] = True
                adj[v, u] = True
        elif remaining < 0:
            # Remove some cross edges
            cross_edges = [(u, v) for u in A for v in B]
            random.shuffle(cross_edges)
            for u, v in cross_edges[:abs(remaining)]:
                adj[u, v] = False
                adj[v, u] = False
        
        deg = adj.sum(axis=1).astype(int)
        score = compute_score(adj, deg)
        
        if score < best_score:
            best_score = score
            best_adj = adj.copy()
            best_deg = deg.copy()
        
        # Local search from this starting point
        current_adj = adj.copy()
        current_deg = deg.copy()
        current_score = score
        
        T = 3.0
        cooling = 0.999
        
        all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick random edge to remove and non-edge to add
            for attempt in range(10):
                e1 = random.choice(all_possible_edges)
                if not current_adj[e1[0], e1[1]]:
                    continue
                e2 = random.choice(all_possible_edges)
                if current_adj[e2[0], e2[1]]:
                    continue
                break
            else:
                continue
            
            u1, v1 = e1
            u2, v2 = e2
            
            new_adj = current_adj.copy()
            new_deg = current_deg.copy()
            
            new_adj[u1, v1] = False
            new_adj[v1, u1] = False
            new_deg[u1] -= 1
            new_deg[v1] -= 1
            
            new_adj[u2, v2] = True
            new_adj[v2, u2] = True
            new_deg[u2] += 1
            new_deg[v2] += 1
            
            new_score = compute_score(new_adj, new_deg)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_adj = new_adj
                current_deg = new_deg
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj.copy()
                    best_deg = current_deg.copy()
            
            T *= cooling
    
    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}
