"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 16
    target_edges = 65
    
    # Precompute all possible edges
    all_edges = []
    edge_index = {}
    for u in range(n):
        for v in range(u+1, n):
            edge_index[(u, v)] = len(all_edges)
            all_edges.append((u, v))
    
    num_possible_edges = len(all_edges)  # 120
    
    # Build adjacency from edge bitmask
    def build_adj(edge_mask):
        adj = [set() for _ in range(n)]
        for (u, v), idx in edge_index.items():
            if edge_mask & (1 << idx):
                adj[u].add(v)
                adj[v].add(u)
        return adj
    
    def get_triangles(adj):
        """Return list of triangles (u, v, w) with u < v < w"""
        triangles = []
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        triangles.append((u, v, w))
        return triangles
    
    def compute_score(adj, triangles):
        """Return (max_degree_sum, sum_of_squares) over triangles"""
        if not triangles:
            return 0, 0.0
        deg = [len(adj[i]) for i in range(n)]
        max_sum = 0
        sum_sq = 0.0
        for (u, v, w) in triangles:
            s = deg[u] + deg[v] + deg[w]
            if s > max_sum:
                max_sum = s
            sum_sq += s * s
        return max_sum, sum_sq
    
    def edge_mask_to_list(edge_mask):
        edges = []
        for (u, v), idx in edge_index.items():
            if edge_mask & (1 << idx):
                edges.append([u, v])
        return edges
    
    # Initial: K_{8,8} + one edge
    init_mask = 0
    for u in range(8):
        for v in range(8, 16):
            init_mask |= (1 << edge_index[(u, v)])
    init_mask |= (1 << edge_index[(0, 1)])
    
    adj = build_adj(init_mask)
    triangles = get_triangles(adj)
    max_score, sum_sq = compute_score(adj, triangles)
    
    best_mask = init_mask
    best_score = max_score
    best_sum_sq = sum_sq
    
    # Simulated annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.9997
    
    # Precompute for fast updates
    # For each edge, which triangles contain it
    # We'll recompute triangles when needed
    
    def fast_score_update(adj, triangles, deg):
        """Compute max degree sum and sum of squares from precomputed triangles"""
        if not triangles:
            return 0, 0.0
        max_sum = 0
        sum_sq = 0.0
        for (u, v, w) in triangles:
            s = deg[u] + deg[v] + deg[w]
            if s > max_sum:
                max_sum = s
            sum_sq += s * s
        return max_sum, sum_sq
    
    # Track degrees
    deg = [len(adj[i]) for i in range(n)]
    
    # For each edge, track which triangles it belongs to
    def build_edge_triangle_map(adj, triangles):
        """Map edge (u,v) to list of triangles containing it"""
        edge_tris = {}
        for (u, v, w) in triangles:
            for e in [(u, v), (u, w), (v, w)]:
                e_key = (min(e[0], e[1]), max(e[0], e[1]))
                if e_key not in edge_tris:
                    edge_tris[e_key] = []
                edge_tris[e_key].append((u, v, w))
        return edge_tris
    
    edge_tris = build_edge_triangle_map(adj, triangles)
    
    iteration = 0
    no_improve = 0
    
    while time.time() < deadline:
        iteration += 1
        
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove an edge from a high-degree-sum triangle
            if not triangles:
                continue
            # Pick a triangle with high degree sum
            # Weight by degree sum
            weights = []
            for (u, v, w) in triangles:
                s = deg[u] + deg[v] + deg[w]
                weights.append(s)
            total_w = sum(weights)
            if total_w == 0:
                continue
            r = random.random() * total_w
            cum = 0
            chosen_tri = None
            for i, w in enumerate(weights):
                cum += w
                if cum >= r:
                    chosen_tri = triangles[i]
                    break
            if chosen_tri is None:
                chosen_tri = triangles[-1]
            
            u, v, w = chosen_tri
            # Pick one of the three edges to remove
            edges_to_consider = [(u, v), (u, w), (v, w)]
            # Prefer edges that are in more high-degree triangles
            edge_scores = []
            for e in edges_to_consider:
                e_key = (min(e[0], e[1]), max(e[0], e[1]))
                if e_key in edge_tris:
                    edge_scores.append(len(edge_tris[e_key]))
                else:
                    edge_scores.append(0)
            # Weighted random selection
            total_es = sum(edge_scores)
            if total_es == 0:
                edge_to_remove = random.choice(edges_to_consider)
            else:
                r = random.random() * total_es
                cum = 0
                edge_to_remove = edges_to_consider[0]
                for i, es in enumerate(edge_scores):
                    cum += es
                    if cum >= r:
                        edge_to_remove = edges_to_consider[i]
                        break
            
            e_key = (min(edge_to_remove[0], edge_to_remove[1]), max(edge_to_remove[0], edge_to_remove[1]))
            e_idx = edge_index[e_key]
            
            if not (init_mask & (1 << e_idx)):
                # Edge not in current graph, skip
                continue
            
            # Try removing this edge
            new_mask = init_mask & ~(1 << e_idx)
            new_adj = build_adj(new_mask)
            new_triangles = get_triangles(new_adj)
            new_deg = [len(new_adj[i]) for i in range(n)]
            new_max, new_sq = compute_score(new_adj, new_triangles)
            
            # Also try to add a random edge to maintain count
            # Find a random edge not in the graph
            available_edges = []
            for (u2, v2), idx in edge_index.items():
                if not (new_mask & (1 << idx)):
                    available_edges.append((u2, v2, idx))
            
            if available_edges:
                # Prefer edges that don't create many triangles
                # For now, just pick randomly
                u2, v2, idx2 = random.choice(available_edges)
                new_mask2 = new_mask | (1 << idx2)
                new_adj2 = build_adj(new_mask2)
                new_triangles2 = get_triangles(new_adj2)
                new_deg2 = [len(new_adj2[i]) for i in range(n)]
                new_max2, new_sq2 = compute_score(new_adj2, new_triangles2)
                
                # Use the better of just-removing vs remove+add
                # We need exactly 65 edges, so we must add
                delta_score = new_max2 - max_score
                delta_sq = new_sq2 - sum_sq
                
                if delta_score < 0 or (delta_score == 0 and delta_sq < 0):
                    init_mask = new_mask2
                    adj = new_adj2
                    triangles = new_triangles2
                    deg = new_deg2
                    edge_tris = build_edge_triangle_map(adj, triangles)
                    max_score, sum_sq = new_max2, new_sq2
                    if max_score < best_score:
                        best_score = max_score
                        best_mask = init_mask
                        best_sum_sq = sum_sq
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Accept with probability
                    if T > T_min:
                        accept_prob = math.exp(-delta_score / T)
                        if random.random() < accept_prob:
                            init_mask = new_mask2
                            adj = new_adj2
                            triangles = new_triangles2
                            deg = new_deg2
                            edge_tris = build_edge_triangle_map(adj, triangles)
                            max_score, sum_sq = new_max2, new_sq2
                            if max_score < best_score:
                                best_score = max_score
                                best_mask = init_mask
                                best_sum_sq = sum_sq
                                no_improve = 0
                            else:
                                no_improve += 1
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
            else:
                no_improve += 1
        
        elif move_type < 0.7:
            # Add an edge outside high-degree triangles
            available_edges = []
            for (u2, v2), idx in edge_index.items():
                if not (init_mask & (1 << idx)):
                    available_edges.append((u2, v2, idx))
            
            if not available_edges:
                continue
            
            # Score each available edge by how many triangles it would create
            # and the degree sums of those triangles
            edge_scores = []
            for u2, v2, idx in available_edges:
                # Count triangles that would be formed
                common_neighbors = adj[u2] & adj[v2]
                tri_count = len(common_neighbors)
                # Estimate the degree sum increase
                score = tri_count * 3  # Rough penalty
                edge_scores.append(score)
            
            # Prefer edges with lower score (fewer triangles)
            # Weighted random selection inversely
            total_inv = sum(1.0 / (s + 1) for s in edge_scores)
            if total_inv > 0:
                r = random.random() * total_inv
                cum = 0
                chosen_idx = 0
                for i, s in enumerate(edge_scores):
                    cum += 1.0 / (s + 1)
                    if cum >= r:
                        chosen_idx = i
                        break
                u2, v2, idx2 = available_edges[chosen_idx]
            else:
                u2, v2, idx2 = random.choice(available_edges)
            
            # Now we need to remove an edge to maintain count
            # Remove an edge from a high-degree triangle
            if not triangles:
                continue
            
            # Pick an edge to remove
            weights = []
            for (u, v, w) in triangles:
                s = deg[u] + deg[v] + deg[w]
                weights.append(s)
            total_w = sum(weights)
            if total_w == 0:
                continue
            r = random.random() * total_w
            cum = 0
            chosen_tri = None
            for i, w in enumerate(weights):
                cum += w
                if cum >= r:
                    chosen_tri = triangles[i]
                    break
            if chosen_tri is None:
                chosen_tri = triangles[-1]
            
            u, v, w = chosen_tri
            edges_to_consider = [(u, v), (u, w), (v, w)]
            edge_to_remove = random.choice(edges_to_consider)
            e_key = (min(edge_to_remove[0], edge_to_remove[1]), max(edge_to_remove[0], edge_to_remove[1]))
            e_idx = edge_index[e_key]
            
            if not (init_mask & (1 << e_idx)):
                continue
            
            new_mask = (init_mask & ~(1 << e_idx)) | (1 << idx2)
            new_adj = build_adj(new_mask)
            new_triangles = get_triangles(new_adj)
            new_deg = [len(new_adj[i]) for i in range(n)]
            new_max, new_sq = compute_score(new_adj, new_triangles)
            
            delta_score = new_max - max_score
            delta_sq = new_sq - sum_sq
            
            if delta_score < 0 or (delta_score == 0 and delta_sq < 0):
                init_mask = new_mask
                adj = new_adj
                triangles = new_triangles
                deg = new_deg
                edge_tris = build_edge_triangle_map(adj, triangles)
                max_score, sum_sq = new_max, new_sq
                if max_score < best_score:
                    best_score = max_score
                    best_mask = init_mask
                    best_sum_sq = sum_sq
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if T > T_min:
                    accept_prob = math.exp(-delta_score / T)
                    if random.random() < accept_prob:
                        init_mask = new_mask
                        adj = new_adj
                        triangles = new_triangles
                        deg = new_deg
                        edge_tris = build_edge_triangle_map(adj, triangles)
                        max_score, sum_sq = new_max, new_sq
                        if max_score < best_score:
                            best_score = max_score
                            best_mask = init_mask
                            best_sum_sq = sum_sq
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
        
        else:
            # Swap: remove one edge, add another
            available_edges = []
            for (u2, v2), idx in edge_index.items():
                if not (init_mask & (1 << idx)):
                    available_edges.append((u2, v2, idx))
            
            if not available_edges or not triangles:
                continue
            
            u2, v2, idx2 = random.choice(available_edges)
            
            # Pick edge to remove from high-degree triangle
            weights = []
            for (u, v, w) in triangles:
                s = deg[u] + deg[v] + deg[w]
                weights.append(s)
            total_w = sum(weights)
            if total_w == 0:
                continue
            r = random.random() * total_w
            cum = 0
            chosen_tri = None
            for i, w in enumerate(weights):
                cum += w
                if cum >= r:
                    chosen_tri = triangles[i]
                    break
            if chosen_tri is None:
                chosen_tri = triangles[-1]
            
            u, v, w = chosen_tri
            edges_to_consider = [(u, v), (u, w), (v, w)]
            edge_to_remove = random.choice(edges_to_consider)
            e_key = (min(edge_to_remove[0], edge_to_remove[1]), max(edge_to_remove[0], edge_to_remove[1]))
            e_idx = edge_index[e_key]
            
            if not (init_mask & (1 << e_idx)):
                continue
            
            new_mask = (init_mask & ~(1 << e_idx)) | (1 << idx2)
            new_adj = build_adj(new_mask)
            new_triangles = get_triangles(new_adj)
            new_deg = [len(new_adj[i]) for i in range(n)]
            new_max, new_sq = compute_score(new_adj, new_triangles)
            
            delta_score = new_max - max_score
            delta_sq = new_sq - sum_sq
            
            if delta_score < 0 or (delta_score == 0 and delta_sq < 0):
                init_mask = new_mask
                adj = new_adj
                triangles = new_triangles
                deg = new_deg
                edge_tris = build_edge_triangle_map(adj, triangles)
                max_score, sum_sq = new_max, new_sq
                if max_score < best_score:
                    best_score = max_score
                    best_mask = init_mask
                    best_sum_sq = sum_sq
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if T > T_min:
                    accept_prob = math.exp(-delta_score / T)
                    if random.random() < accept_prob:
                        init_mask = new_mask
                        adj = new_adj
                        triangles = new_triangles
                        deg = new_deg
                        edge_tris = build_edge_triangle_map(adj, triangles)
                        max_score, sum_sq = new_max, new_sq
                        if max_score < best_score:
                            best_score = max_score
                            best_mask = init_mask
                            best_sum_sq = sum_sq
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
        
        # Cool down
        T = max(T_min, T * cooling)
        
        # Restart if stuck
        if no_improve > 500:
            T = 5.0
            no_improve = 0
        
        # Check deadline
        if time.time() > deadline - 5.0:
            break
    
    # Final polish: try local search on best
    adj = build_adj(best_mask)
    triangles = get_triangles(adj)
    deg = [len(adj[i]) for i in range(n)]
    max_score, sum_sq = compute_score(adj, triangles)
    
    # Try to improve with targeted moves
    for _ in range(1000):
        if time.time() > deadline - 1.0:
            break
        improved = False
        for (u, v, w) in triangles:
            s = deg[u] + deg[v] + deg[w]
            if s <= max_score:
                continue
            # Try removing one of the three edges
            for e in [(u, v), (u, w), (v, w)]:
                e_key = (min(e[0], e[1]), max(e[0], e[1]))
                e_idx = edge_index[e_key]
                if not (best_mask & (1 << e_idx)):
                    continue
                # Find an edge to add
                available = []
                for (u2, v2), idx in edge_index.items():
                    if not (best_mask & (1 << idx)):
                        available.append((u2, v2, idx))
                if not available:
                    continue
                for u2, v2, idx2 in available[:20]:  # Limit
                    new_mask = (best_mask & ~(1 << e_idx)) | (1 << idx2)
                    new_adj = build_adj(new_mask)
                    new_triangles = get_triangles(new_adj)
                    new_deg = [len(new_adj[i]) for i in range(n)]
                    new_max, new_sq = compute_score(new_adj, new_triangles)
                    if new_max < max_score:
                        best_mask = new_mask
                        adj = new_adj
                        triangles = new_triangles
                        deg = new_deg
                        max_score, sum_sq = new_max, new_sq
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    
    # Return the best result
    edges = edge_mask_to_list(best_mask)
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END

