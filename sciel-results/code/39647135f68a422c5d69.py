import numpy as np
import time
import random
from collections import defaultdict

def construct():
    n = 16
    target_edges = 65
    
    # Precompute all triangles
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                triangles.append((u, v, w))
    
    tri_array = np.array(triangles, dtype=int)
    num_tris = len(triangles)
    
    # Incremental state tracking
    # For each triangle, track which of its 3 edges are present
    tri_edge_state = np.zeros((num_tris, 3), dtype=np.int8)  # 0 or 1 for each edge
    tri_active = np.zeros(num_tris, dtype=bool)
    tri_deg_sum = np.zeros(num_tris, dtype=np.int32)
    
    def init_state(A):
        nonlocal tri_edge_state, tri_active, tri_deg_sum
        d = A.sum(axis=1)
        for i in range(num_tris):
            u, v, w = tri_array[i]
            tri_edge_state[i, 0] = 1 if A[u, v] else 0
            tri_edge_state[i, 1] = 1 if A[v, w] else 0
            tri_edge_state[i, 2] = 1 if A[w, u] else 0
            tri_active[i] = (tri_edge_state[i, 0] == 1 and tri_edge_state[i, 1] == 1 and tri_edge_state[i, 2] == 1)
            tri_deg_sum[i] = d[u] + d[v] + d[w]
    
    def get_score():
        if not np.any(tri_active):
            return 0
        return int(np.max(tri_deg_sum[tri_active]))
    
    def update_triangle_for_edge(u, v, added):
        """Update triangle states when edge (u,v) is added or removed."""
        nonlocal tri_active, tri_deg_sum
        d = A.sum(axis=1)
        for i in range(num_tris):
            tu, tv, tw = tri_array[i]
            # Check if this triangle contains edge (u,v)
            if (tu == u and tv == v) or (tu == v and tv == u):
                tri_edge_state[i, 0] = 1 if added else 0
            elif (tu == v and tv == u) or (tu == u and tv == v):
                # This shouldn't happen with our triangle ordering (u < v < w)
                pass
            elif (tu == u and tw == v) or (tu == v and tw == u):
                tri_edge_state[i, 2] = 1 if added else 0
            elif (tv == u and tw == v) or (tv == v and tw == u):
                tri_edge_state[i, 1] = 1 if added else 0
            
            if (tri_edge_state[i, 0] == 1 and tri_edge_state[i, 1] == 1 and tri_edge_state[i, 2] == 1) != tri_active[i]:
                tri_active[i] = (tri_edge_state[i, 0] == 1 and tri_edge_state[i, 1] == 1 and tri_edge_state[i, 2] == 1)
            
            # Update degree sum
            tri_deg_sum[i] = d[tu] + d[tv] + d[tw]
    
    def fast_swap_score(A, rem_u, rem_v, add_u, add_v):
        """Compute new score after swapping edge (rem_u,rem_v) -> (add_u,add_v)."""
        d_old = A.sum(axis=1)
        
        # Compute new degrees
        d_new = d_old.copy()
        d_new[rem_u] -= 1
        d_new[rem_v] -= 1
        d_new[add_u] += 1
        d_new[add_v] += 1
        
        # Compute new triangle states
        # For each triangle, determine if it's active in the new graph
        # A triangle (a,b,c) is active if all three edges exist
        # We need to check: for each triangle, are all 3 edges present in new graph?
        
        # Optimize: only triangles affected by the edge changes can change status
        # Triangles containing rem_u-rem_v edge: lose one edge
        # Triangles containing add_u-add_v edge: gain one edge
        # Other triangles: unchanged (but their degree sums change)
        
        # Let's just compute directly for simplicity, but efficiently
        # Build a set of edges for fast lookup
        edge_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j]:
                    if i != rem_u or j != rem_v:
                        if i != rem_v or j != rem_u:
                            edge_set.add((i, j))
        if add_u < add_v:
            edge_set.add((add_u, add_v))
        else:
            edge_set.add((add_v, add_u))
        
        # Compute score
        max_sum = 0
        u, v, w = tri_array[:, 0], tri_array[:, 1], tri_array[:, 2]
        for i in range(num_tris):
            ti_u, ti_v, ti_w = tri_array[i]
            # Check if all 3 edges exist
            if (ti_u, ti_v) in edge_set and (ti_v, ti_w) in edge_set and (ti_w, ti_u) in edge_set:
                s = d_new[ti_u] + d_new[ti_v] + d_new[ti_w]
                if s > max_sum:
                    max_sum = s
        return max_sum
    
    # Initial Construction: K_{8,8} + 1 edge
    A = np.zeros((n, n), dtype=bool)
    for u in range(8):
        for v in range(8, 16):
            A[u, v] = True
            A[v, u] = True
    A[0, 1] = True
    A[1, 0] = True
    
    init_state(A)
    current_score = get_score()
    best_A = A.copy()
    best_score = current_score
    
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # Simulated Annealing with targeted moves
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    # Precompute edge lists for faster access
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Get initial edge lists
    current_edges_list = [(u, v) for u, v in all_edges if A[u, v]]
    non_edges_list = [(u, v) for u, v in all_edges if not A[u, v]]
    
    iter_count = 0
    while time.time() - start_time < time_limit:
        iter_count += 1
        
        # Targeted move: try to remove an edge from a high-degree-sum triangle
        d = A.sum(axis=1)
        u, v, w = tri_array[:, 0], tri_array[:, 1], tri_array[:, 2]
        active_mask = tri_active
        
        if np.any(active_mask):
            active_sums = tri_deg_sum[active_mask]
            max_sum = np.max(active_sums)
            
            # Get triangles with sum close to max
            threshold = max_sum - 2
            high_tri_indices = np.where(active_mask & (tri_deg_sum >= threshold))[0]
            
            # Pick a random high triangle and remove one of its edges
            if len(high_tri_indices) > 0 and random.random() < 0.7:
                tri_idx = high_tri_indices[random.randint(0, len(high_tri_indices) - 1)]
                tri_u, tri_v, tri_w = tri_array[tri_idx]
                # Pick one of the three edges to remove (only if it exists)
                candidate_edges = []
                if A[tri_u, tri_v]:
                    candidate_edges.append((tri_u, tri_v))
                if A[tri_v, tri_w]:
                    candidate_edges.append((tri_v, tri_w))
                if A[tri_w, tri_u]:
                    candidate_edges.append((tri_w, tri_u))
                if candidate_edges:
                    rem_u, rem_v = random.choice(candidate_edges)
                else:
                    rem_u, rem_v = random.choice(current_edges_list)
            else:
                rem_u, rem_v = random.choice(current_edges_list)
        else:
            rem_u, rem_v = random.choice(current_edges_list)
        
        # Pick a non-edge
        if len(non_edges_list) == 0:
            break
        add_u, add_v = random.choice(non_edges_list)
        if (add_u, add_v) == (rem_u, rem_v):
            continue
            
        # Perform the swap
        A_new = A.copy()
        A_new[rem_u, rem_v] = False
        A_new[rem_v, rem_u] = False
        A_new[add_u, add_v] = True
        A_new[add_v, add_u] = True
        
        new_score = fast_swap_score(A, rem_u, rem_v, add_u, add_v)
        
        # Acceptance criteria
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / T):
            A = A_new
            current_score = new_score
            
            # Update edge lists
            current_edges_list = [(rem_u, rem_v) if e != (add_u, add_v) else e for e in current_edges_list]
            current_edges_list = [e for e in current_edges_list if e != (rem_u, rem_v)]
            current_edges_list.append((add_u, add_v))
            
            non_edges_list = [e for e in non_edges_list if e != (rem_u, rem_v)]
            non_edges_list.append((rem_u, rem_v))
            
            if current_score < best_score:
                best_score = current_score
                best_A = A.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            if random.random() < 0.02:
                T = 5.0

    # Final polish: Hill Climbing with more exhaustive local search
    A_final = best_A.copy()
    score_final = best_score
    
    # Rebuild edge lists for final
    final_edges_list = [(u, v) for u, v in all_edges if A_final[u, v]]
    final_non_edges_list = [(u, v) for u, v in all_edges if not A_final[u, v]]
    
    improved = True
    polish_iter = 0
    while improved and time.time() - start_time < time_limit and polish_iter < 100:
        improved = False
        polish_iter += 1
        
        # Get current edges for A_final
        d = A_final.sum(axis=1)
        
        # Targeted improvement: try to remove edges from worst triangles
        u, v, w = tri_array[:, 0], tri_array[:, 1], tri_array[:, 2]
        active_mask = np.zeros(num_tris, dtype=bool)
        for i in range(num_tris):
            ti_u, ti_v, ti_w = tri_array[i]
            active_mask[i] = A_final[ti_u, ti_v] and A_final[ti_v, ti_w] and A_final[ti_w, ti_u]
        
        if np.any(active_mask):
            tri_sums = d[u] + d[v] + d[w]
            active_sums = tri_sums[active_mask]
            max_sum = np.max(active_sums)
            threshold = max_sum - 1
            high_tri_indices = np.where(active_mask & (tri_sums >= threshold))[0]
            
            for tri_idx in high_tri_indices:
                if time.time() - start_time > time_limit:
                    break
                tri_u, tri_v, tri_w = tri_array[tri_idx]
                # Try removing each edge of this triangle
                for rem_u, rem_v in [(tri_u, tri_v), (tri_v, tri_w), (tri_w, tri_u)]:
                    if not A_final[rem_u, rem_v]:
                        continue
                    # Try adding each non-edge
                    for add_u, add_v in final_non_edges_list:
                        if (add_u, add_v) == (rem_u, rem_v):
                            continue
                        
                        A_new = A_final.copy()
                        A_new[rem_u, rem_v] = False
                        A_new[rem_v, rem_u] = False
                        A_new[add_u, add_v] = True
                        A_new[add_v, add_u] = True
                        
                        new_score = fast_swap_score(A_final, rem_u, rem_v, add_u, add_v)
                        if new_score < score_final:
                            A_final = A_new
                            score_final = new_score
                            improved = True
                            
                            # Update edge lists
                            final_edges_list = [e for e in final_edges_list if e != (rem_u, rem_v)]
                            final_edges_list.append((add_u, add_v))
                            final_non_edges_list = [e for e in final_non_edges_list if e != (add_u, add_v)]
                            final_non_edges_list.append((rem_u, rem_v))
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # If no targeted improvement, try random swaps
        if not improved and len(final_edges_list) > 0 and len(final_non_edges_list) > 0:
            for _ in range(100):
                if time.time() - start_time > time_limit:
                    break
                rem_u, rem_v = random.choice(final_edges_list)
                add_u, add_v = random.choice(final_non_edges_list)
                if (add_u, add_v) == (rem_u, rem_v):
                    continue
                
                A_new = A_final.copy()
                A_new[rem_u, rem_v] = False
                A_new[rem_v, rem_u] = False
                A_new[add_u, add_v] = True
                A_new[add_v, add_u] = True
                
                new_score = fast_swap_score(A_final, rem_u, rem_v, add_u, add_v)
                if new_score < score_final:
                    A_final = A_new
                    score_final = new_score
                    improved = True
                    
                    final_edges_list = [e for e in final_edges_list if e != (rem_u, rem_v)]
                    final_edges_list.append((add_u, add_v))
                    final_non_edges_list = [e for e in final_non_edges_list if e != (add_u, add_v)]
                    final_non_edges_list.append((rem_u, rem_v))
                    break

    # Pick the best
    score_best_A = get_score() if best_A is not None else 999
    # Recompute for A_final
    d_final = A_final.sum(axis=1)
    max_score_final = 0
    for i in range(num_tris):
        ti_u, ti_v, ti_w = tri_array[i]
        if A_final[ti_u, ti_v] and A_final[ti_v, ti_w] and A_final[ti_w, ti_u]:
            s = d_final[ti_u] + d_final[ti_v] + d_final[ti_w]
            if s > max_score_final:
                max_score_final = s
    
    if max_score_final < score_best_A:
        final_A = A_final
    else:
        final_A = best_A
        
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if final_A[u, v]:
                edges.append([u, v])
                
    # Verify edge count
    if len(edges) != 65:
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        
    return {'n': n, 'edges': edges}
