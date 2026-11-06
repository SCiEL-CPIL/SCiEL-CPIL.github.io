"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 145  # safety margin before 150s limit
    n = 16
    edge_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edge_idx)
    
    # Map edge (i,j) to index
    edge_to_idx = {}
    for idx, (i, j) in enumerate(edge_idx):
        edge_to_idx[(i, j)] = idx
        edge_to_idx[(j, i)] = idx
        
    # Precompute all K4s and their edge indices
    k4s = list(combinations(range(n), 4))
    num_k4s = len(k4s)
    k4_edges = []
    for k4 in k4s:
        edges = []
        for a, b in combinations(k4, 2):
            edges.append(edge_to_idx[(a, b)])
        k4_edges.append(edges)
    
    # Convert to numpy arrays for fast operations
    k4_edges_arr = np.array(k4_edges, dtype=np.int32)  # shape (num_k4s, 6)
    
    # For each edge, which K4s does it belong to?
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_k4s[e].append(k4_idx)
    
    # Convert edge_k4s to numpy for fast access
    edge_k4s_arr = [np.array(e, dtype=np.int32) if e else np.array([], dtype=np.int32) for e in edge_k4s]
    
    def count_colors_in_k4_numpy(colors, edge_indices):
        """Count distinct colors in a K4 using numpy."""
        c = colors[edge_indices]
        # Count distinct values
        return len(set(c))
    
    def compute_violations_numpy(colors):
        """Compute distinct color count for each K4 using numpy. Returns array of distinct counts."""
        # colors is a numpy array of shape (num_edges,)
        # k4_edges_arr is shape (num_k4s, 6)
        # Get colors for all edges in all K4s
        k4_colors = colors[k4_edges_arr]  # shape (num_k4s, 6)
        
        # Count distinct colors per K4
        # For each K4, count how many unique colors among its 6 edges
        distinct_counts = np.zeros(num_k4s, dtype=np.int32)
        for k in range(num_k4s):
            distinct_counts[k] = len(set(k4_colors[k]))
        
        return distinct_counts
    
    def compute_violations_fast(colors_np):
        """Fast computation of distinct colors per K4."""
        k4_colors = colors_np[k4_edges_arr]  # shape (num_k4s, 6)
        
        # Use a trick: for each K4, sort the 6 colors and count transitions
        k4_sorted = np.sort(k4_colors, axis=1)  # shape (num_k4s, 6)
        # Count number of distinct = 1 + number of positions where value changes
        diffs = np.diff(k4_sorted, axis=1)  # shape (num_k4s, 5)
        distinct_counts = 1 + (diffs != 0).sum(axis=1)
        
        return distinct_counts
    
    def multiplicative_init(m, vals=None):
        """Initialize using multiplicative structure: color edge (i,j) by (v_i * v_j) mod m"""
        if m < n:
            return None
        if vals is None:
            vals = list(range(1, n + 1))
            random.shuffle(vals)
        colors = np.zeros(num_edges, dtype=np.int32)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                colors[idx] = (vals[i] * vals[j]) % m
                idx += 1
        return colors
    
    def is_valid_numpy(colors_np):
        """Check if coloring is valid (every K4 has >= 5 distinct colors)."""
        distinct = compute_violations_fast(colors_np)
        return (distinct >= 5).all()
    
    # Try multiplicative structure first with various moduli
    best_result = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_colors = 17
    
    for m in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
        if time.time() > deadline - 5:
            break
        colors = multiplicative_init(m)
        if colors is None:
            continue
        distinct = compute_violations_fast(colors)
        if (distinct >= 5).all():
            used = len(set(colors.tolist()))
            if used < best_colors:
                best_colors = used
                best_result = colors.tolist()
    
    # Search Loop
    for num_c in [13, 14, 15, 16]:
        if time.time() > deadline - 2:
            break
        
        for restart in range(200):
            if time.time() > deadline - 2:
                break
            
            # Initialize
            if num_c >= n:
                colors = multiplicative_init(num_c)
                if colors is None:
                    colors = np.random.randint(0, num_c, size=num_edges, dtype=np.int32)
            else:
                colors = np.random.randint(0, num_c, size=num_edges, dtype=np.int32)
            
            # Setup state
            distinct = compute_violations_fast(colors)
            violations = np.where(distinct < 5)[0]
            
            # Local Search
            stuck_counter = 0
            max_stuck = 50
            
            for step in range(200000):
                if time.time() > deadline - 2:
                    break
                
                if len(violations) == 0:
                    # Valid!
                    used = len(set(colors.tolist()))
                    if used < best_colors:
                        best_colors = used
                        best_result = colors.tolist()
                    break
                
                # Pick a violated K4 - prefer the one with fewest distinct colors
                v_idx = violations[random.randint(0, len(violations) - 1)]
                v_k4_edges = k4_edges_arr[v_idx]
                v_colors = colors[v_k4_edges]
                v_distinct = distinct[v_idx]
                
                # Find colors missing in this K4
                v_color_set = set(v_colors.tolist())
                missing_colors = [c for c in range(num_c) if c not in v_color_set]
                if not missing_colors:
                    # Can't add a new color, try to fix by rearranging
                    # Just skip this K4 for now
                    continue
                
                edges_to_try = v_k4_edges.tolist()
                random.shuffle(edges_to_try)
                
                move_made = False
                for e in edges_to_try:
                    old_c = colors[e]
                    # Count how many times old_c appears in this K4
                    count_old_in_k4 = int(np.sum(v_colors == old_c))
                    
                    # We want to change an edge where old_c appears more than once
                    # so we don't lose a distinct color
                    if count_old_in_k4 == 1:
                        continue
                    
                    new_c = random.choice(missing_colors)
                    if new_c == old_c:
                        continue
                    
                    # Check all affected K4s
                    affected_k4s = edge_k4s_arr[e]
                    if len(affected_k4s) == 0:
                        continue
                    
                    delta_violations = 0
                    can_apply = True
                    
                    for k_idx in affected_k4s:
                        k_dist = distinct[k_idx]
                        was_violated = k_dist < 5
                        
                        # Get colors in this K4
                        k_colors = colors[k4_edges_arr[k_idx]]
                        cnt_old = int(np.sum(k_colors == old_c))
                        cnt_new = int(np.sum(k_colors == new_c))
                        
                        # Compute new distinct count
                        new_dist = k_dist
                        if cnt_old == 1:
                            new_dist -= 1
                        if cnt_new == 0:
                            new_dist += 1
                        
                        is_violated = new_dist < 5
                        
                        if was_violated and not is_violated:
                            delta_violations -= 1
                        elif not was_violated and is_violated:
                            delta_violations += 1
                            # Don't create new violations
                            can_apply = False
                            break
                    
                    if can_apply and delta_violations < 0:
                        # Apply the move
                        colors[e] = new_c
                        for k_idx in affected_k4s:
                            k_colors = colors[k4_edges_arr[k_idx]]
                            new_dist = len(set(k_colors.tolist()))
                            distinct[k_idx] = new_dist
                        # Recompute violations
                        violations = np.where(distinct < 5)[0]
                        move_made = True
                        stuck_counter = 0
                        break
                    else:
                        continue
                
                if not move_made:
                    stuck_counter += 1
                    if stuck_counter > max_stuck:
                        break
                else:
                    stuck_counter = 0
    
    if best_colors > 17:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
    return best_result
# EVOLVE-BLOCK-END
