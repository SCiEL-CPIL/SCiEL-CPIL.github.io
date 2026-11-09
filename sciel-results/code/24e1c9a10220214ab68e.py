import numpy as np
import time
import random

def count_conflicts(adj, u, v, c):
    """Count number of non-rainbow 4-cycles involving edge (u,v) with color c."""
    n = adj.shape[0]
    conflicts = 0
    for x in range(n):
        if x == u or x == v:
            continue
        if adj[v, x] == -1:
            continue
        for y in range(n):
            if y == u or y == v or y == x:
                continue
            if adj[x, y] == -1:
                continue
            if adj[y, u] == -1:
                continue
            c_vx = adj[v, x]
            c_xy = adj[x, y]
            c_yu = adj[y, u]
            if len({c, c_vx, c_xy, c_yu}) < 4:
                conflicts += 1
    return conflicts

def total_conflicts(adj):
    """Total number of non-rainbow 4-cycles in the graph."""
    n = adj.shape[0]
    total = 0
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == -1:
                continue
            c = adj[i, j]
            total += count_conflicts(adj, i, j, c)
    return total

def construct():
    n = 20
    deadline = time.time() + 115  # 35s margin
    
    random.seed(42)
    np.random.seed(42)
    
    best_edges = 0
    best_adj = None
    best_conflicts = float('inf')
    
    # Try multiple restarts with different initializations
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Initialize with a random dense graph
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        
        # Add random edges with random colors
        # Start with about 50% density
        edge_prob = 0.5 + 0.3 * (restart / 50)  # Vary density
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < edge_prob:
                    c = random.randint(0, n-1)
                    adj[i, j] = c
                    adj[j, i] = c
        
        # Simulated annealing
        current_conflicts = total_conflicts(adj)
        current_edges = np.sum(adj != -1) // 2
        
        if current_conflicts == 0:
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj.copy()
        
        T = 10.0  # Initial temperature
        cooling = 0.995
        
        for step in range(5000):
            if time.time() > deadline:
                break
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Recolor a random edge
                i, j = random.randint(0, n-1), random.randint(0, n-1)
                if i == j or adj[i, j] == -1:
                    continue
                
                old_c = adj[i, j]
                new_c = random.randint(0, n-1)
                if new_c == old_c:
                    continue
                
                # Compute delta conflicts
                old_conf = count_conflicts(adj, i, j, old_c)
                adj[i, j] = new_c
                adj[j, i] = new_c
                new_conf = count_conflicts(adj, i, j, new_c)
                
                delta = new_conf - old_conf
                
                if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                    # Accept
                    pass
                else:
                    # Revert
                    adj[i, j] = old_c
                    adj[j, i] = old_c
                
                current_conflicts += delta
                
            elif move_type < 0.7:
                # Remove a random edge
                i, j = random.randint(0, n-1), random.randint(0, n-1)
                if i == j or adj[i, j] == -1:
                    continue
                
                old_c = adj[i, j]
                old_conf = count_conflicts(adj, i, j, old_c)
                
                adj[i, j] = -1
                adj[j, i] = -1
                
                # Removing edge reduces conflicts by old_conf (approximately)
                # But other edges might have had conflicts with this edge
                # For simplicity, recompute locally
                new_edges = np.sum(adj != -1) // 2
                delta_edges = new_edges - current_edges
                
                # Accept removal if it reduces conflicts or with some probability
                if old_conf > 0 or random.random() < np.exp(2.0 / max(T, 1e-10)):
                    current_edges = new_edges
                    # Recompute conflicts approximately
                    # This is expensive, so we do it less frequently
                    if step % 50 == 0:
                        current_conflicts = total_conflicts(adj)
                else:
                    # Revert
                    adj[i, j] = old_c
                    adj[j, i] = old_c
                
            else:
                # Add a random edge
                i, j = random.randint(0, n-1), random.randint(0, n-1)
                if i == j or adj[i, j] != -1:
                    continue
                
                # Try a few colors
                best_c = None
                best_conf = float('inf')
                for c in random.sample(range(n), min(5, n)):
                    adj[i, j] = c
                    adj[j, i] = c
                    conf = count_conflicts(adj, i, j, c)
                    if conf < best_conf:
                        best_conf = conf
                        best_c = c
                    adj[i, j] = -1
                    adj[j, i] = -1
                
                if best_c is not None:
                    adj[i, j] = best_c
                    adj[j, i] = best_c
                    current_edges += 1
                    current_conflicts += best_conf
                    
                    # Check if this creates too many conflicts
                    if best_conf > 3 and random.random() > np.exp(-best_conf / max(T, 1e-10)):
                        # Revert
                        adj[i, j] = -1
                        adj[j, i] = -1
                        current_edges -= 1
                        current_conflicts -= best_conf
            
            # Update temperature
            T *= cooling
            
            # Track best valid solution
            if current_conflicts == 0 and current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj.copy()
        
        # After SA, try to add more edges greedily
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            
            for c in range(n):
                if count_conflicts(adj, u, v, c) == 0:
                    adj[u, v] = c
                    adj[v, u] = c
                    break
        
        # Final check
        final_conflicts = total_conflicts(adj)
        final_edges = np.sum(adj != -1) // 2
        
        if final_conflicts == 0 and final_edges > best_edges:
            best_edges = final_edges
            best_adj = adj.copy()
        
        # Early stop if we reach high density
        if best_edges > 140:
            break
    
    # Convert best_adj to list of lists
    res = []
    if best_adj is None:
        # Fallback to a valid sparse graph
        for v in range(1, n):
            res.append([0, v, v % n])
        return res
    
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] != -1:
                res.append([i, j, int(best_adj[i, j])])
    return res
