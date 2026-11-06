import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try n=26, 25, etc.
    # We prioritize finding a valid n=26 as it is the known max.
    # If we find it, we try to see if we can get 27 with remaining time.
    
    best_result = None
    best_n = 0
    
    # Try to find a valid coloring for n=26
    # We use a fast local search that maintains k-matrix and only updates on acceptance.
    
    for n in [26, 25, 24]:
        if time.time() > deadline - 20:
            break
            
        # Try multiple restarts for this n
        found = False
        while time.time() < deadline - 10 and not found:
            result = local_search_fast(n, time_limit=5.0)
            if result is not None:
                found = True
                if n > best_n:
                    best_n = n
                    best_result = result
                    
        if best_n >= 26:
            # We have a valid 26. Try to improve to 27?
            # The task says 27 or 28 would be new.
            # If we have time, try n=27.
            if time.time() < deadline - 5:
                # Try to extend or find a 27
                # This is unlikely to succeed quickly, but we can try.
                # For now, just return 26 if we have it.
                pass
            break

    if best_result:
        return best_result
        
    # Fallback to seed
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def local_search_fast(n, time_limit=5.0):
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
    
    # k[c][u][v] = number of common neighbors of u and v in color c
    # We store this to calculate deltas quickly.
    k = [[[0] * n for _ in range(n)] for _ in range(5)]
    
    def calc_k_for_color(c):
        for u in range(n):
            for v in range(u + 1, n):
                common = neighbors[c][u] & neighbors[c][v]
                kc = common.bit_count()
                k[c][u][v] = kc
                k[c][v][u] = kc
                
    def update_neighbors(idx, old_color, new_color):
        i, j = edges[idx]
        neighbors[old_color][i] &= ~(1 << j)
        neighbors[old_color][j] &= ~(1 << i)
        neighbors[new_color][i] |= (1 << j)
        neighbors[new_color][j] |= (1 << i)
        
    # Initial setup
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
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
        
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < start_time + time_limit:
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        i, j = edges[e_idx]
        
        # Calculate delta violations
        # We do NOT update k or neighbors yet.
        delta = 0
        
        # Effect on old_color: Removing (i, j)
        # k[old][i][w] decreases by 1 if j is in N_old(w)
        # k[old][j][w] decreases by 1 if i is in N_old(w)
        
        for w in range(n):
            if w == i or w == j:
                continue
            
            # Check i-w pair
            if (neighbors[old_color][w] >> j) & 1:
                # j is in N_old(w), so k_old(i, w) decreases by 1
                old_k = k[old_color][i][w]
                # delta -= (old_k choose 2) - ((old_k-1) choose 2)
                # (k choose 2) - (k-1 choose 2) = k-1
                delta -= (old_k - 1)
                
            if (neighbors[old_color][w] >> i) & 1:
                # i is in N_old(w), so k_old(j, w) decreases by 1
                old_k = k[old_color][j][w]
                delta -= (old_k - 1)
                
        # Effect on new_color: Adding (i, j)
        # k[new][i][w] increases by 1 if j is in N_new(w)
        # k[new][j][w] increases by 1 if i is in N_new(w)
        
        for w in range(n):
            if w == i or w == j:
                continue
            
            if (neighbors[new_color][w] >> j) & 1:
                # j is in N_new(w), so k_new(i, w) increases by 1
                old_k = k[new_color][i][w]
                # delta += ((k+1) choose 2) - (k choose 2) = k
                delta += old_k
                
            if (neighbors[new_color][w] >> i) & 1:
                # i is in N_new(w), so k_new(j, w) increases by 1
                old_k = k[new_color][j][w]
                delta += old_k
                
        # Acceptance criterion
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            # Accept move
            current_violations += delta
            colors[e_idx] = new_color
            update_neighbors(e_idx, old_color, new_color)
            
            # Update k matrix for accepted move
            # We need to update k[old][i][w], k[old][j][w], k[new][i][w], k[new][j][w]
            # Note: k[u][v] == k[v][u]
            
            for w in range(n):
                if w == i or w == j:
                    continue
                
                # Update old_color k values
                if (neighbors[old_color][w] >> j) & 1:
                    # This condition was true BEFORE the move?
                    # Wait. neighbors[old_color] has NOT been updated yet in this block?
                    # No, update_neighbors was called.
                    # So neighbors[old_color][w] no longer has j.
                    # We need to use the PRE-move state or logic.
                    # Actually, the condition "j is in N_old(w)" is equivalent to "w is in N_old(j)".
                    # Since we removed j from N_old(i) and i from N_old(j),
                    # and w is not i or j,
                    # The set of w such that j in N_old(w) is the same as the set of w such that w in N_old(j) BEFORE removal.
                    # But we just updated neighbors.
                    # So we can check: if (neighbors[old_color][j] >> w) & 1 was true?
                    # No, neighbors[old_color][j] lost i. It still has w if w was neighbor.
                    # So (neighbors[old_color][j] >> w) & 1 is still valid for checking if w was neighbor of j?
                    # Yes, because we only removed i from j's neighbors.
                    # And we only removed j from i's neighbors.
                    # So for w != i, j, the neighbor relationship between w and i, and w and j, is unchanged in terms of "was it there?".
                    # Wait.
                    # Move: remove (i,j) from old_color.
                    # N_old(i) loses j. N_old(j) loses i.
                    # For any w != i,j:
                    # If w was neighbor of i in old_color, it still is.
                    # If w was neighbor of j in old_color, it still is.
                    # So to check if j was in N_old(w), we check if w is in N_old(j) (current state, since w!=i, j's neighbor list only lost i, not w).
                    # So `if (neighbors[old_color][j] >> w) & 1:` is correct.
                    
                    # Update k[old][i][w]
                    k[old_color][i][w] -= 1
                    k[old_color][w][i] -= 1
                    
                if (neighbors[old_color][i] >> w) & 1:
                    # Check if i was in N_old(w) -> w in N_old(i)
                    k[old_color][j][w] -= 1
                    k[old_color][w][j] -= 1
                    
                # Update new_color k values
                # Move: add (i,j) to new_color.
                # N_new(i) gains j. N_new(j) gains i.
                # For w != i,j:
                # If w is neighbor of j in new_color (current state, before adding i to j? No, we added i to j).
                # Wait, update_neighbors added i to N_new(j) and j to N_new(i).
                # So N_new(j) now contains i.
                # To check if j was in N_new(w) BEFORE addition:
                # This is equivalent to w being in N_new(j) BEFORE addition.
                # Since we just added i to N_new(j), and w != i, the bit for w is unchanged.
                # So `if (neighbors[new_color][j] >> w) & 1:` checks if w was neighbor of j.
                
                if (neighbors[new_color][j] >> w) & 1:
                    k[new_color][i][w] += 1
                    k[new_color][w][i] += 1
                    
                if (neighbors[new_color][i] >> w) & 1:
                    k[new_color][j][w] += 1
                    k[new_color][w][j] += 1
            
            if current_violations == 0:
                return {'n': n, 'colors': colors}
                
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01
            
    return None
