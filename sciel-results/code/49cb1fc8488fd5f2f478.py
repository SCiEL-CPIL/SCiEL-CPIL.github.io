import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try n=26 first, then 25, etc.
    for n in [26, 25, 24, 23, 22]:
        if time.time() > deadline - 15:
            break
        result = local_search_optimized(n, time_limit=15)
        if result is not None:
            return result
            
    # Fallback to seed
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def local_search_optimized(n, time_limit=15):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Map edge to index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # Initialize colors randomly
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # neighbors[c][v] is a bitmask of neighbors of v in color c
    neighbors = [[0] * n for _ in range(5)]
    
    def update_neighbors(idx, old_color, new_color):
        i, j = edges[idx]
        neighbors[old_color][i] &= ~(1 << j)
        neighbors[old_color][j] &= ~(1 << i)
        neighbors[new_color][i] |= (1 << j)
        neighbors[new_color][j] |= (1 << i)
        
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
    # Precompute common neighbor counts for each color and pair of vertices
    # k[c][u][v] = number of common neighbors of u and v in color c
    # We only need this for delta calculation.
    # Storing full k matrix is 5 * n * n = 5 * 676 = 3380 ints. Very small.
    k = [[[0] * n for _ in range(n)] for _ in range(5)]
    
    def calc_k_for_color(c):
        for u in range(n):
            for v in range(u + 1, n):
                common = neighbors[c][u] & neighbors[c][v]
                k[c][u][v] = common.bit_count()
                k[c][v][u] = common.bit_count()
                
    for c in range(5):
        calc_k_for_color(c)
        
    def total_violations():
        v = 0
        for c in range(5):
            for u in range(n):
                for v_idx in range(u + 1, n):
                    kc = k[c][u][v_idx]
                    if kc >= 2:
                        v += kc * (kc - 1) // 2
        return v
        
    current_violations = total_violations()
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.999
    
    # Local search loop
    while time.time() < start_time + time_limit:
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        i, j = edges[e_idx]
        
        # Calculate delta violations
        # Remove (i,j) from old_color
        # Add (i,j) to new_color
        
        delta = 0
        
        # Effect on old_color
        # k[old][i][w] decreases by 1 if j is in N_old(w) AND i is in N_old(w)?
        # No. k[c][u][v] = |N_c(u) \cap N_c(v)|.
        # N_old(i) loses j. N_old(j) loses i.
        # For any w != i, j:
        # k_old(i, w) = |N_old(i) \cap N_old(w)|.
        # New N_old(i) = N_old(i) \ {j}.
        # So k_old(i, w) decreases by 1 if j \in N_old(w).
        # k_old(j, w) decreases by 1 if i \in N_old(w).
        
        # We need to check if j is neighbor of w in old_color.
        # This is (neighbors[old_color][w] >> j) & 1
        
        for w in range(n):
            if w == i or w == j:
                continue
            # Update k for old_color
            # Check if j is in N_old(w)
            if (neighbors[old_color][w] >> j) & 1:
                # j is a neighbor of w in old_color.
                # So j \in N_old(w).
                # Since j \in N_old(i) (before removal), j \in N_old(i) \cap N_old(w).
                # So k_old(i, w) decreases by 1.
                old_k = k[old_color][i][w]
                new_k = old_k - 1
                delta -= (old_k * (old_k - 1) // 2) - (new_k * (new_k - 1) // 2)
                k[old_color][i][w] = new_k
                k[old_color][w][i] = new_k
                
            if (neighbors[old_color][w] >> i) & 1:
                # i is a neighbor of w in old_color.
                # So i \in N_old(w).
                # Since i \in N_old(j) (before removal), i \in N_old(j) \cap N_old(w).
                # So k_old(j, w) decreases by 1.
                old_k = k[old_color][j][w]
                new_k = old_k - 1
                delta -= (old_k * (old_k - 1) // 2) - (new_k * (new_k - 1) // 2)
                k[old_color][j][w] = new_k
                k[old_color][w][j] = new_k
                
        # Effect on new_color
        # N_new(i) gains j. N_new(j) gains i.
        for w in range(n):
            if w == i or w == j:
                continue
            if (neighbors[new_color][w] >> j) & 1:
                # j is neighbor of w in new_color.
                # j \in N_new(w).
                # j \in N_new(i) (after addition).
                # So k_new(i, w) increases by 1.
                old_k = k[new_color][i][w]
                new_k = old_k + 1
                delta += (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
                k[new_color][i][w] = new_k
                k[new_color][w][i] = new_k
                
            if (neighbors[new_color][w] >> i) & 1:
                # i is neighbor of w in new_color.
                # i \in N_new(w).
                # i \in N_new(j) (after addition).
                # So k_new(j, w) increases by 1.
                old_k = k[new_color][j][w]
                new_k = old_k + 1
                delta += (new_k * (new_k - 1) // 2) - (old_k * (old_k - 1) // 2)
                k[new_color][j][w] = new_k
                k[new_color][w][j] = new_k
                
        # Apply change if accepted
        if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
            current_violations += delta
            colors[e_idx] = new_color
            update_neighbors(e_idx, old_color, new_color)
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Revert k changes
            # We need to undo the k updates.
            # It's easier to just recalculate k for the affected pairs if we don't store history.
            # But reverting is symmetric.
            # Let's just recalculate k for old_color and new_color for pairs involving i or j.
            # This is O(n) per revert, which is acceptable.
            for w in range(n):
                if w == i or w == j:
                    continue
                # Recalculate k[old][i][w]
                common = neighbors[old_color][i] & neighbors[old_color][w]
                k[old_color][i][w] = common.bit_count()
                k[old_color][w][i] = common.bit_count()
                
                common = neighbors[old_color][j] & neighbors[old_color][w]
                k[old_color][j][w] = common.bit_count()
                k[old_color][w][j] = common.bit_count()
                
                common = neighbors[new_color][i] & neighbors[new_color][w]
                k[new_color][i][w] = common.bit_count()
                k[new_color][w][i] = common.bit_count()
                
                common = neighbors[new_color][j] & neighbors[new_color][w]
                k[new_color][j][w] = common.bit_count()
                k[new_color][w][j] = common.bit_count()
            
            # Revert colors and neighbors
            colors[e_idx] = old_color
            update_neighbors(e_idx, new_color, old_color)
            
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    return None
