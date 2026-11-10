"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120 edges
    
    # Precompute edge index mapping
    idx_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_map[(i, j)] = idx
            idx += 1
    
    # Precompute all K4s and their edge indices
    k4_edges = []
    edge_to_k4 = [[] for _ in range(num_edges)]
    for v in itertools.combinations(range(n), 4):
        a, b, c, d = v
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        e_indices = []
        for e in edges:
            i, j = e
            if i > j:
                i, j = j, i
            ei = idx_map[(i, j)]
            e_indices.append(ei)
            edge_to_k4[ei].append(len(k4_edges))
        k4_edges.append(e_indices)
    
    num_k4s = len(k4_edges)
    
    # Convert to numpy arrays for speed
    k4_edges_arr = np.array(k4_edges, dtype=np.int32)  # shape (1820, 6)
    edge_to_k4_arr = [np.array(lst, dtype=np.int32) for lst in edge_to_k4]
    
    # Precompute pair indices for violation checking
    pairs_idx = [(0,1),(0,2),(0,3),(0,4),(0,5),
                 (1,2),(1,3),(1,4),(1,5),
                 (2,3),(2,4),(2,5),
                 (3,4),(3,5),
                 (4,5)]
    
    def count_violations_all(colors_np):
        """Count violations for all K4s."""
        k4_colours = colors_np[k4_edges_arr]  # shape (1820, 6)
        num_equal_pairs = np.zeros(len(k4_colours), dtype=np.int32)
        for (a, b) in pairs_idx:
            num_equal_pairs += (k4_colours[:, a] == k4_colours[:, b]).astype(np.int32)
        violations = np.sum(num_equal_pairs >= 2)
        return violations
    
    def count_violations_for_k4s(colors_np, k4_indices):
        """Count violations only for specified K4s."""
        if len(k4_indices) == 0:
            return 0
        k4_colours = colors_np[k4_edges_arr[k4_indices]]
        num_equal_pairs = np.zeros(len(k4_colours), dtype=np.int32)
        for (a, b) in pairs_idx:
            num_equal_pairs += (k4_colours[:, a] == k4_colours[:, b]).astype(np.int32)
        violations = np.sum(num_equal_pairs >= 2)
        return violations
    
    # Base solution: (i+j) mod 17, uses 17 colours
    base_colors_list = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_colors = base_colors_list[:]
    best_num = len(set(best_colors))  # 17
    
    best_colors_np = np.array(best_colors, dtype=np.int32)
    assert count_violations_all(best_colors_np) == 0
    
    def try_eliminate_colour_greedy(cur_colors_np, target_colour):
        """Greedy approach: assign edges one by one, trying colours in order."""
        cur = cur_colors_np.copy()
        target_edges = np.where(cur == target_colour)[0].tolist()
        
        if not target_edges:
            return None
        
        other_colours = [c for c in set(cur.tolist()) if c != target_colour]
        if not other_colours:
            return None
        
        # Sort target_edges by number of K4s they affect (most constrained first)
        target_edges.sort(key=lambda ei: len(edge_to_k4_arr[ei]), reverse=True)
        
        for ei in target_edges:
            orig = cur[ei]
            found = False
            affected_k4s = edge_to_k4_arr[ei]
            for new_c in other_colours:
                cur[ei] = new_c
                if count_violations_for_k4s(cur, affected_k4s) == 0:
                    found = True
                    break
            if not found:
                cur[ei] = orig
                return None
        
        if count_violations_all(cur) == 0:
            return cur
        return None
    
    def simulated_annealing(current_np, max_time):
        """Simulated annealing to minimize number of colours."""
        cur = current_np.copy()
        cur_violations = count_violations_all(cur)
        cur_palette = set(cur.tolist())
        cur_num = len(cur_palette)
        
        best = cur.copy()
        best_violations = cur_violations
        best_num = cur_num
        
        T = 10.0
        cooling = 0.9995
        min_T = 0.01
        
        while time.time() < max_time and T > min_T:
            # Pick random edge
            ei = random.randint(0, num_edges - 1)
            
            # Get affected K4s
            affected_k4s = edge_to_k4_arr[ei]
            
            # Try a few random colours
            for _ in range(3):
                new_c = random.randint(0, 15)  # Try colours 0-15
                if new_c == cur[ei]:
                    continue
                
                orig = cur[ei]
                cur[ei] = new_c
                
                # Check violations only for affected K4s
                new_violations = count_violations_for_k4s(cur, affected_k4s)
                
                # Check if palette changed
                new_palette = set(cur.tolist())
                new_num = len(new_palette)
                
                # Accept if valid or better
                if new_violations == 0:
                    if new_num < best_num:
                        best = cur.copy()
                        best_violations = new_violations
                        best_num = new_num
                else:
                    # Reject invalid moves
                    cur[ei] = orig
                    continue
            
            T *= cooling
        
        return best, best_num
    
    # Iteratively try to eliminate colours
    current_np = best_colors_np.copy()
    
    # First pass: greedy elimination
    for iteration in range(50):
        if time.time() > deadline - 30:
            break
        
        cur_cols = sorted(set(current_np.tolist()))
        if len(cur_cols) <= best_num:
            break
        
        eliminated = False
        for c in reversed(cur_cols):
            if time.time() > deadline - 30:
                break
            result = try_eliminate_colour_greedy(current_np, c)
            if result is not None:
                current_np = result
                eliminated = True
                break
        
        if not eliminated:
            break
    
    # Second pass: simulated annealing to find better solutions
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        best_annealed, annealed_num = simulated_annealing(current_np, time.time() + remaining_time)
        if annealed_num < best_num:
            best_colors_np = best_annealed
            best_num = annealed_num
    
    # Final check
    if count_violations_all(best_colors_np) != 0:
        best_colors_np = np.array(base_colors_list, dtype=np.int32)
        best_num = 17
    
    best_colors = best_colors_np.tolist()
    
    # Renormalize colours to 0..best_num-1
    col_map = {}
    for i, c in enumerate(sorted(set(best_colors))):
        col_map[c] = i
    best_colors = [col_map[c] for c in best_colors]
    
    return best_colors
# EVOLVE-BLOCK-END
