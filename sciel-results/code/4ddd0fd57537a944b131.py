import random
import time
import itertools

def construct():
    n = 16
    # Precompute all edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): idx for idx, (i, j) in enumerate(edges)}
    num_edges = len(edges)
    
    # Precompute K4s and their edge indices
    k4s = []
    k4_edge_indices = []
    for comb in itertools.combinations(range(n), 4):
        v0, v1, v2, v3 = comb
        e_list = [
            edge_index[(v0, v1)],
            edge_index[(v0, v2)],
            edge_index[(v0, v3)],
            edge_index[(v1, v2)],
            edge_index[(v1, v3)],
            edge_index[(v2, v3)]
        ]
        k4s.append(comb)
        k4_edge_indices.append(e_list)
        
    num_k4s = len(k4s)
    
    # Function to check if a K4 is valid
    def is_k4_valid(e_list, colors):
        c_set = set()
        for e in e_list:
            c_set.add(colors[e])
        return len(c_set) >= 5

    # Function to find a duplicated edge in an invalid K4
    # Returns (edge_idx, color) of an edge that can be changed
    def find_repair_edge(e_list, colors):
        c_counts = {}
        for e in e_list:
            c = colors[e]
            c_counts[c] = c_counts.get(c, 0) + 1
        
        # Find a color with count > 1
        for e in e_list:
            c = colors[e]
            if c_counts[c] > 1:
                return e, c
        return None, None

    # Function to find a new color not in the K4
    def find_new_color(e_list, colors, K):
        current_colors = set(colors[e] for e in e_list)
        for c in range(K):
            if c not in current_colors:
                return c
        # If all colors are present (impossible for K=15 and <5 colors in K4)
        return random.randint(0, K - 1)

    def local_search(K, time_limit):
        # Initialize with 1-factorization if K >= 15
        if K >= 15:
            colors = [0] * num_edges
            # 1-factorization of K16
            # Vertices 0..14, 15 is infinity
            # Mod 15
            for i in range(15):
                color = i
                # Edge (15, i)
                e_idx = edge_index[(15, i)]
                colors[e_idx] = color
                
                for j in range(1, 8):
                    u = (i + j) % 15
                    v = (i - j) % 15
                    if u > v:
                        u, v = v, u
                    e_idx = edge_index[(u, v)]
                    colors[e_idx] = color
        else:
            # Random init for smaller K
            colors = [random.randint(0, K - 1) for _ in range(num_edges)]

        start_time = time.time()
        
        # Count violations
        def count_violations():
            count = 0
            for e_list in k4_edge_indices:
                if not is_k4_valid(e_list, colors):
                    count += 1
            return count

        # Initial check
        violations = count_violations()
        if violations == 0:
            return colors

        # Local Search Loop
        while time.time() - start_time < time_limit:
            # Find a random invalid K4
            # To be efficient, we can keep a list of invalid K4s
            # But for simplicity, we pick a random K4 and check it
            # If valid, try another. If invalid, repair.
            
            # Better: iterate through a shuffled list of K4s
            # But rebuilding the list is expensive.
            # Let's just pick random K4s until we find an invalid one.
            
            found_invalid = False
            for _ in range(100): # Try 100 random K4s
                idx = random.randint(0, num_k4s - 1)
                e_list = k4_edge_indices[idx]
                if not is_k4_valid(e_list, colors):
                    found_invalid = True
                    break
            
            if not found_invalid:
                # We might have fixed all, or we are stuck in a cycle of checking valid K4s
                # Let's do a full check
                if count_violations() == 0:
                    return colors
                # If still violations, force pick one by scanning
                for idx in range(num_k4s):
                    e_list = k4_edge_indices[idx]
                    if not is_k4_valid(e_list, colors):
                        found_invalid = True
                        break
                if not found_invalid:
                    return colors

            if found_invalid:
                # Repair
                e_to_change, old_color = find_repair_edge(e_list, colors)
                if e_to_change is not None:
                    new_color = find_new_color(e_list, colors, K)
                    colors[e_to_change] = new_color
                else:
                    # Should not happen for invalid K4 with K>=5
                    # Fallback: random change
                    e_to_change = random.choice(e_list)
                    colors[e_to_change] = random.randint(0, K - 1)
            
            # Occasionally, if stuck, try a random kick
            # (Not implemented here to keep it simple, but the repair strategy is monotonic in terms of fixing specific K4s)
            
        # Time limit reached
        # Check if valid
        if count_violations() == 0:
            return colors
        return None

    # Try K=15
    res = local_search(15, time_limit=100)
    if res:
        return res

    # Try K=16
    res = local_search(16, time_limit=10)
    if res:
        return res

    # Fallback to 17-color seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
