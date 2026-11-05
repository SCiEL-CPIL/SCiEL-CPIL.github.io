import random
import time
import math
import numpy as np

def construct():
    n = 16
    target_edges = 65
    
    # Precompute all possible edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_pairs_set = set(all_pairs)
    m = len(all_pairs)  # 120
    
    # Precompute triangles: list of (i, j, k) where all three pairs exist
    # For each pair (u,v), find common neighbors w > v
    triangles = []
    adj = [set() for _ in range(n)]
    for u, v in all_pairs:
        adj[u].add(v)
        adj[v].add(u)
    
    for u in range(n):
        for v in range(u+1, n):
            common = adj[u] & adj[v]
            for w in common:
                if w > v:
                    triangles.append((u, v, w))
    
    num_triangles = len(triangles)
    
    # Initialize with K_{8,8} + 1 edge
    def initial_edges():
        edges = set()
        for u in range(8):
            for v in range(8, 16):
                edges.add((u, v))
        edges.add((0, 1))
        return edges
    
    def compute_degrees(edge_set):
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def compute_score(edge_set):
        deg = compute_degrees(edge_set)
        max_sum = 0
        for u, v, w in triangles:
            s = deg[u] + deg[v] + deg[w]
            if s > max_sum:
                max_sum = s
        return max_sum, deg
    
    def smooth_score(edge_set, beta=5.0):
        """Smooth surrogate using log-sum-exp over triangle degree sums."""
        deg = compute_degrees(edge_set)
        if num_triangles == 0:
            return 0.0
        sums = np.array([deg[u] + deg[v] + deg[w] for u, v, w in triangles], dtype=np.float64)
        # Log-sum-exp for smooth max
        max_s = np.max(sums)
        if max_s == 0:
            return 0.0
        # Soft max: (1/beta) * log(sum(exp(beta*s_i)))
        exp_sums = np.exp(beta * (sums - max_s))
        soft_max = max_s + np.log(np.sum(exp_sums)) / beta
        return soft_max
    
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    best_edges = initial_edges()
    best_score, _ = compute_score(best_edges)
    
    # Multiple restarts with simulated annealing
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() - start_time > deadline - 10:
            break
        
        if restart == 0:
            current_edges = best_edges
        else:
            # Perturb from best
            current_edges = set(best_edges)
            # Randomly swap some edges
            for _ in range(5 + restart * 2):
                if time.time() - start_time > deadline - 10:
                    break
                edge_list = list(current_edges)
                if not edge_list:
                    break
                rem = random.choice(edge_list)
                # Find a non-edge
                add = None
                for _ in range(20):
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    if (u, v) not in current_edges:
                        add = (u, v)
                        break
                if add is None:
                    continue
                current_edges.discard(rem)
                current_edges.add(add)
        
        current_score, _ = compute_score(current_edges)
        T = 0.5 + restart * 0.1
        cooling = 0.99995
        no_improve_count = 0
        max_no_improve = 2000 + restart * 500
        
        while time.time() - start_time < deadline - 5:
            if no_improve_count > max_no_improve:
                break
            
            # Generate move: edge swap
            edge_list = list(current_edges)
            if not edge_list:
                break
            remove_edge = random.choice(edge_list)
            
            # Find non-edge to add
            add_edge = None
            for _ in range(15):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) not in current_edges:
                    add_edge = (u, v)
                    break
            
            if add_edge is None or add_edge == remove_edge:
                no_improve_count += 1
                continue
            
            # Apply move
            new_edges = set(current_edges)
            new_edges.discard(remove_edge)
            new_edges.add(add_edge)
            
            new_score, _ = compute_score(new_edges)
            
            if new_score < current_score:
                current_edges = new_edges
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_edges = new_edges
            else:
                delta = new_score - current_score
                if delta > 0 and T > 0.01:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_edges = new_edges
                        current_score = new_score
                no_improve_count += 1
            
            T *= cooling
        
        # Final hill-climbing polish on best
        if time.time() - start_time < deadline - 2:
            polish_edges = set(best_edges)
            polish_score, _ = compute_score(polish_edges)
            improved = True
            while improved and time.time() - start_time < deadline - 1:
                improved = False
                for i in range(len(all_pairs)):
                    if time.time() - start_time > deadline - 1:
                        break
                    e1 = all_pairs[i]
                    # Try removing e1 if it's in the set
                    if e1 in polish_edges:
                        new_e = set(polish_edges)
                        new_e.discard(e1)
                        # Add a random non-edge
                        for _ in range(5):
                            u = random.randint(0, n-1)
                            v = random.randint(0, n-1)
                            if u == v:
                                continue
                            if u > v:
                                u, v = v, u
                            if (u, v) not in new_e:
                                new_e.add((u, v))
                                break
                        else:
                            continue
                        new_s, _ = compute_score(new_e)
                        if new_s < polish_score:
                            polish_edges = new_e
                            polish_score = new_s
                            improved = True
                            if polish_score < best_score:
                                best_score = polish_score
                                best_edges = new_e
                    # Try adding e1 if it's not in the set, and removing a random edge
                    elif e1 not in polish_edges:
                        # Remove a random edge from polish_edges
                        if not polish_edges:
                            continue
                        rem = random.choice(list(polish_edges))
                        new_e = set(polish_edges)
                        new_e.discard(rem)
                        new_e.add(e1)
                        new_s, _ = compute_score(new_e)
                        if new_s < polish_score:
                            polish_edges = new_e
                            polish_score = new_s
                            improved = True
                            if polish_score < best_score:
                                best_score = polish_score
                                best_edges = new_e
    
    # Convert to list of lists
    best_list = sorted([list(e) for e in best_edges])
    return {'n': n, 'edges': best_list}
