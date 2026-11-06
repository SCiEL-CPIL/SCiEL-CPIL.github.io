import random
import time

def construct():
    n = 24
    # Edge order from the problem
    edge_order = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]
    
    # Build edge weight matrix (1-based arrival index)
    weight = [[0] * n for _ in range(n)]
    for idx, (u, v) in enumerate(edge_order, 1):
        weight[u][v] = idx
        weight[v][u] = idx
    
    def cycle_max_weight(cycle):
        """Compute max edge weight in a Hamilton cycle."""
        max_w = 0
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            w = weight[u][v]
            if w > max_w:
                max_w = w
        return max_w
    
    def two_opt(cycle):
        """Try 2-opt moves to reduce max edge weight."""
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    # Remove edges (cycle[i], cycle[i+1]) and (cycle[j], cycle[j+1])
                    # Add edges (cycle[i], cycle[j]) and (cycle[i+1], cycle[j+1])
                    a, b = cycle[i], cycle[(i + 1) % n]
                    c, d = cycle[j], cycle[(j + 1) % n]
                    w_ab = weight[a][b]
                    w_cd = weight[c][d]
                    w_ac = weight[a][c]
                    w_bd = weight[b][d]
                    old_max = max(w_ab, w_cd)
                    new_max = max(w_ac, w_bd)
                    if new_max < old_max:
                        # Reverse the segment between i+1 and j
                        cycle[i+1:j+1] = cycle[i+1:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return cycle
    
    def or_opt(cycle):
        """Try Or-opt moves (relocate 1 or 2 consecutive vertices)."""
        improved = True
        while improved:
            improved = False
            # Or-opt-1: move single vertex
            for i in range(n):
                v = cycle[i]
                prev_v = cycle[(i - 1) % n]
                next_v = cycle[(i + 1) % n]
                w_prev_v = weight[prev_v][v]
                w_v_next = weight[v][next_v]
                for j in range(n):
                    if j == i or j == (i - 1) % n or j == (i + 1) % n:
                        continue
                    a = cycle[j]
                    b = cycle[(j + 1) % n]
                    w_ab = weight[a][b]
                    w_av = weight[a][v]
                    w_vb = weight[v][b]
                    old_max = max(w_prev_v, w_v_next, w_ab)
                    new_max = max(w_prev_v, w_v_next, w_av, w_vb)
                    # Actually we remove w_ab and add w_av, w_vb; keep w_prev_v, w_v_next
                    # Old edges: w_prev_v, w_v_next, w_ab
                    # New edges: w_prev_v, w_v_next, w_av, w_vb, and we removed w_ab
                    # Wait, this isn't quite right. Let me reconsider.
                    # Or-opt: remove v from position i, insert between a and b
                    # Old: ...prev_v-v-next_v... and ...a-b...
                    # New: ...prev_v-next_v... and ...a-v-b...
                    # Old max: max(w_prev_v, w_v_next, w_ab)
                    # New max: max(w_prev_v, w_v_next, w_av, w_vb)
                    if new_max < old_max:
                        # Perform the move
                        del cycle[i]
                        # Find new position of a (index may have shifted)
                        # a was at index j, but if j > i, it's now at j-1
                        new_j = j - 1 if j > i else j
                        cycle.insert(new_j + 1, v)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Or-opt-2: move two consecutive vertices
            for i in range(n):
                v1 = cycle[i]
                v2 = cycle[(i + 1) % n]
                prev_v = cycle[(i - 1) % n]
                next_v = cycle[(i + 2) % n]
                w_prev_v1 = weight[prev_v][v1]
                w_v1_v2 = weight[v1][v2]
                w_v2_next = weight[v2][next_v]
                for j in range(n):
                    if j in [i, (i+1)%n, (i-1)%n, (i+2)%n]:
                        continue
                    a = cycle[j]
                    b = cycle[(j + 1) % n]
                    w_ab = weight[a][b]
                    w_av1 = weight[a][v1]
                    w_v1_v2_new = weight[v1][v2]  # same
                    w_v2_b = weight[v2][b]
                    old_max = max(w_prev_v1, w_v1_v2, w_v2_next, w_ab)
                    new_max = max(w_prev_v1, w_v2_next, w_av1, w_v1_v2_new, w_v2_b)
                    if new_max < old_max:
                        # Remove v1, v2 and insert between a and b
                        # This is complex; skip for now
                        pass
        return cycle
    
    # Use multiple random starts with local search
    rng = random.Random(42)
    best_cycle = None
    best_score = float('inf')
    
    start_time = time.time()
    time_limit = 55  # Leave some margin
    
    # Generate several random initial cycles
    num_starts = 50
    for start in range(num_starts):
        if time.time() - start_time > time_limit:
            break
        
        # Random permutation
        cycle = list(range(n))
        rng.shuffle(cycle)
        
        # Local search
        cycle = two_opt(cycle)
        cycle = or_opt(cycle)
        # Another round of 2-opt
        cycle = two_opt(cycle)
        
        score = cycle_max_weight(cycle)
        if score < best_score:
            best_score = score
            best_cycle = cycle[:]
        
        # If we've found a very good solution, we can stop early
        if best_score <= 30:
            break
    
    # Final refinement on best cycle
    if best_cycle is not None:
        best_cycle = two_opt(best_cycle)
        best_cycle = or_opt(best_cycle)
        best_cycle = two_opt(best_cycle)
        best_score = cycle_max_weight(best_cycle)
    
    if best_cycle is None:
        return list(range(n))
    
    return best_cycle

