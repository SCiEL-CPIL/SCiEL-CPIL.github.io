"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    
    # Precompute all K4s and their edge indices
    k4s = []
    for quad in combinations(range(n), 4):
        v0, v1, v2, v3 = quad
        edge_idx_map = {e: i for i, e in enumerate(edges)}
        e_indices = [
            edge_idx_map[(v0, v1)],
            edge_idx_map[(v0, v2)],
            edge_idx_map[(v0, v3)],
            edge_idx_map[(v1, v2)],
            edge_idx_map[(v1, v3)],
            edge_idx_map[(v2, v3)]
        ]
        k4s.append(e_indices)
    
    num_k4s = len(k4s)  # 1820
    
    # Precompute which K4s contain each edge
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx in range(num_k4s):
        for e_idx in k4s[k4_idx]:
            edge_to_k4s[e_idx].append(k4_idx)
    
    # Fallback solution: (i+j) % 17
    seed_col = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    def count_colors(col):
        return len(set(col))
    
    def is_valid(col):
        for k in range(num_k4s):
            if len(set(col[e] for e in k4s[k])) < 5:
                return False
        return True

    best_result = None
    best_color_count = float('inf')
    
    # Check seed
    if is_valid(seed_col):
        best_result = seed_col
        best_color_count = count_colors(seed_col)
        
    # Try to find better solutions
    for target_colors in [13, 14, 15]:
        if time.time() > deadline - 20.0:
            break
            
        # Multiple restarts
        for restart in range(100):
            if time.time() > deadline - 10.0:
                break
                
            # Random initialization
            col = [random.randint(0, target_colors - 1) for _ in range(num_edges)]
            
            # Initialize violation counts and distinct counts
            k4_distinct = [0] * num_k4s
            current_violations = 0
            for k in range(num_k4s):
                d = len(set(col[e] for e in k4s[k]))
                k4_distinct[k] = d
                if d < 5:
                    current_violations += 1
                    
            if current_violations == 0:
                c_count = count_colors(col)
                if c_count < best_color_count:
                    best_color_count = c_count
                    best_result = col[:]
                if c_count <= 13:
                    return best_result
                break # Found a valid solution for this target, try next (lower) target?
                # Wait, the loop is 13, 14, 15.
                # If we are at 14 and find a 14-solution, we break the restart loop.
                # Then the outer loop moves to 15.
                # This is inefficient. If we find a 14-solution, we should try to reduce it to 13.
                # Or just continue searching for 13 in the next outer iteration?
                # No, the outer iteration for 13 is already done.
                # So we should order the loop 15, 14, 13?
                # No, finding 13 is hard. Finding 15 is easy.
                # Let's stick to 13, 14, 15.
                # If we fail 13, we try 14. If we succeed 14, we record it.
                # Then we try 15. If we succeed 15, we record it if better (no).
                # So we end up with the best of 14 and 15.
                # But we missed the chance to reduce 14 to 13.
                # Let's add a polish step.
                pass
            
            # Local Search
            T = 1.0
            T_min = 0.01
            cooling = 0.999
            max_iters = 5000
            
            for it in range(max_iters):
                if time.time() > deadline - 5.0:
                    break
                    
                e_idx = random.randint(0, num_edges - 1)
                old_color = col[e_idx]
                new_color = random.randint(0, target_colors - 1)
                
                if old_color == new_color:
                    continue
                
                delta_viol = 0
                affected_k4s = edge_to_k4s[e_idx]
                
                col[e_idx] = new_color
                
                for k in affected_k4s:
                    old_distinct = k4_distinct[k]
                    new_distinct = len(set(col[ei] for ei in k4s[k]))
                    k4_distinct[k] = new_distinct
                    
                    if old_distinct < 5 and new_distinct >= 5:
                        delta_viol -= 1
                    elif old_distinct >= 5 and new_distinct < 5:
                        delta_viol += 1
                        
                if delta_viol <= 0:
                    current_violations += delta_viol
                else:
                    if random.random() < np.exp(-delta_viol / T):
                        current_violations += delta_viol
                    else:
                        col[e_idx] = old_color
                        for k in affected_k4s:
                            k4_distinct[k] = len(set(col[ei] for ei in k4s[k]))
                
                T *= cooling
                if T < T_min:
                    T = T_min
                    
                if current_violations == 0:
                    c_count = count_colors(col)
                    if c_count < best_color_count:
                        best_color_count = c_count
                        best_result = col[:]
                    break
            
            if current_violations == 0:
                c_count = count_colors(col)
                if c_count < best_color_count:
                    best_color_count = c_count
                    best_result = col[:]
                if c_count <= 13:
                    return best_result

    # Polish: Try to reduce best_result if it has > 13 colors
    if best_result is not None and best_color_count > 13:
        # Try to eliminate colors from best_result
        col = best_result[:]
        colors_in_use = set(col)
        
        # Try to remove each color
        for c in sorted(colors_in_use, key=lambda x: col.count(x)):
            if time.time() > deadline - 5.0:
                break
            if len(set(col)) <= 13:
                break
                
            # Try to recolor all edges with color c to other colors
            # This is a sub-problem.
            # Simple heuristic: pick a random edge with color c, try to change it to a random other color.
            # If it doesn't create violations, accept.
            # Repeat until no edges with color c remain.
            
            edges_with_c = [i for i, v in enumerate(col) if v == c]
            random.shuffle(edges_with_c)
            
            success = True
            for e in edges_with_c:
                # Try to change col[e] from c to some other color
                # Check if changing it creates violations
                # We need to check the K4s containing e.
                # If we change to a new color, we must ensure no K4 drops to < 5 colors.
                # This is complex to do incrementally without full state.
                # Let's just do a local search on the sub-problem.
                # Or simpler: just try random changes.
                
                found_new_color = False
                for _ in range(100):
                    new_c = random.randint(0, best_color_count - 1)
                    if new_c == c:
                        continue
                    # Check if changing col[e] to new_c is valid
                    # Temporarily set
                    old_v = col[e]
                    col[e] = new_c
                    
                    # Check affected K4s
                    valid = True
                    for k in edge_to_k4s[e]:
                        if len(set(col[ei] for ei in k4s[k])) < 5:
                            valid = False
                            break
                            
                    if valid:
                        found_new_color = True
                        break
                    else:
                        col[e] = old_v
                        
                if not found_new_color:
                    # Failed to recolor this edge
                    # Revert all changes made in this color elimination attempt?
                    # This is tricky. Let's just abandon this color elimination.
                    success = False
                    break
                    
            if success:
                # Color c is eliminated
                pass
            else:
                # Restore best_result
                col = best_result[:]
                break

    if best_result is not None:
        return best_result
    
    return seed_col
# EVOLVE-BLOCK-END
