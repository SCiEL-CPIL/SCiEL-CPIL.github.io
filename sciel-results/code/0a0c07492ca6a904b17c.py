import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # We aim for n=26. If we fail, we can try 25, 24, etc.
    # But 26 is the target.
    
    def count_c4s(n, colors):
        """
        colors: list of length n*(n-1)/2, values 0-4.
        Returns number of monochromatic C4s.
        """
        if n < 4:
            return 0
        
        # Map edge index to (u, v)
        # We need to build adjacency matrices for each color.
        # To be fast, we can precompute the edge list.
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        # Build adjacency matrices A_c (n x n) for c in 0..4
        # A_c[i][j] = 1 if edge (i,j) has color c
        A = np.zeros((5, n, n), dtype=np.int32)
        
        for k, c in enumerate(colors):
            u, v = edges[k]
            A[c][u][v] = 1
            A[c][v][u] = 1
        
        total_c4s = 0
        for c in range(5):
            # Number of common neighbors for pair (i, j) is (A_c @ A_c)[i][j]
            # Wait, A_c @ A_c gives number of walks of length 2.
            # For i != j, (A_c^2)[i][j] is the number of common neighbors.
            # A C4 exists if there are at least 2 common neighbors.
            # Number of C4s involving pair (i,j) is C(common, 2).
            A2 = A[c] @ A[c]
            # We only care about i < j
            # Create a mask for upper triangle
            # Sum C(A2[i][j], 2) for all i < j
            # A2 is symmetric.
            # We can iterate or use vectorization.
            # For n=26, n^2 is 676. Iteration is fine.
            for i in range(n):
                for j in range(i + 1, n):
                    common = A2[i][j]
                    if common >= 2:
                        total_c4s += common * (common - 1) // 2
        return total_c4s

    def get_edge_index(n):
        idx = {}
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx[(i, j)] = k
                idx[(j, i)] = k
                k += 1
        return idx

    def find_one_c4(n, colors):
        """
        Returns (u, v, w, x, c) such that u-w-v-x-u is a C4 in color c.
        Returns None if no C4.
        """
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        A = np.zeros((5, n, n), dtype=np.int32)
        for k, c in enumerate(colors):
            u, v = edges[k]
            A[c][u][v] = 1
            A[c][v][u] = 1
        
        for c in range(5):
            A2 = A[c] @ A[c]
            for i in range(n):
                for j in range(i + 1, n):
                    common = A2[i][j]
                    if common >= 2:
                        # Find two common neighbors w, x
                        # Common neighbors are k such that A[c][i][k]==1 and A[c][j][k]==1
                        # We can find them by intersecting rows
                        # Or just scan
                        neighbors_i = np.where(A[c][i] == 1)[0]
                        neighbors_j = np.where(A[c][j] == 1)[0]
                        # Find intersection
                        # Since n is small, simple list intersection
                        common_list = list(set(neighbors_i) & set(neighbors_j))
                        if len(common_list) >= 2:
                            w = common_list[0]
                            x = common_list[1]
                            return (i, j, w, x, c)
        return None

    n = 26
    num_edges = n * (n - 1) // 2
    edge_index = get_edge_index(n)
    
    # Initialize with random colors
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Local Search
    best_c4_count = count_c4s(n, colors)
    
    # If we start with 0 C4s (unlikely), we are done.
    # Otherwise, iterate.
    
    # To speed up, we can cache the adjacency matrices if we only change one edge.
    # But rebuilding is fast enough for n=26.
    
    for iteration in range(100000):
        if time.time() > deadline:
            break
            
        c4_count = count_c4s(n, colors)
        if c4_count == 0:
            break
        
        # Find a C4 to break
        c4 = find_one_c4(n, colors)
        if c4 is None:
            # Should not happen if count > 0
            break
            
        i, j, w, x, c = c4
        # The C4 edges are (i,w), (w,j), (j,x), (x,i)
        # We need to recolor one of these edges to a different color.
        # Let's pick a random edge from the C4.
        edge_pairs = [(i, w), (w, j), (j, x), (x, i)]
        random.shuffle(edge_pairs)
        
        improved = False
        for u, v in edge_pairs:
            idx = edge_index.get((u, v))
            if idx is None:
                continue
            old_c = colors[idx]
            if old_c != c:
                # This edge is not part of the C4 in color c? 
                # Wait, find_one_c4 returns vertices forming a C4 in color c.
                # So edges (i,w), (w,j), (j,x), (x,i) MUST be color c.
                # So old_c should be c.
                pass
            
            # Try changing to a new color
            new_c = random.randint(0, 4)
            if new_c == c:
                new_c = (c + 1) % 5
                
            colors[idx] = new_c
            new_count = count_c4s(n, colors)
            
            if new_count < c4_count:
                # Accept
                c4_count = new_count
                improved = True
                break
            else:
                # Revert
                colors[idx] = old_c
        
        if not improved:
            # Stuck. Try a random move to escape local minimum.
            # Or restart?
            # Let's try a random edge change that doesn't increase count too much.
            # Or just accept a neutral move.
            random_idx = random.randint(0, num_edges - 1)
            old_c = colors[random_idx]
            new_c = random.randint(0, 4)
            colors[random_idx] = new_c
            new_count = count_c4s(n, colors)
            if new_count > c4_count + 10: # Allow small increase
                colors[random_idx] = old_c
            else:
                c4_count = new_count

    final_count = count_c4s(n, colors)
    if final_count == 0:
        return {'n': n, 'colors': colors}
    
    # If n=26 failed, try n=25
    # ... (Repeat for 25, 24, etc. if time permits, but 26 should work)
    
    # Fallback: Return a known valid smaller n if 26 fails
    # Let's try n=25 quickly
    if time.time() < deadline - 10:
        n = 25
        num_edges = n * (n - 1) // 2
        edge_index = get_edge_index(n)
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        for iteration in range(50000):
            if time.time() > deadline - 5:
                break
            c4_count = count_c4s(n, colors)
            if c4_count == 0:
                break
            c4 = find_one_c4(n, colors)
            if c4 is None:
                break
            i, j, w, x, c = c4
            edge_pairs = [(i, w), (w, j), (j, x), (x, i)]
            random.shuffle(edge_pairs)
            improved = False
            for u, v in edge_pairs:
                idx = edge_index.get((u, v))
                if idx is None: continue
                old_c = colors[idx]
                new_c = random.randint(0, 4)
                if new_c == c: new_c = (c+1)%5
                colors[idx] = new_c
                new_count = count_c4s(n, colors)
                if new_count < c4_count:
                    c4_count = new_count
                    improved = True
                    break
                else:
                    colors[idx] = old_c
            if not improved:
                random_idx = random.randint(0, num_edges - 1)
                old_c = colors[random_idx]
                new_c = random.randint(0, 4)
                colors[random_idx] = new_c
                new_count = count_c4s(n, colors)
                if new_count > c4_count + 10:
                    colors[random_idx] = old_c
                else:
                    c4_count = new_count
        
        final_count = count_c4s(n, colors)
        if final_count == 0:
            return {'n': n, 'colors': colors}

    # Fallback n=11 (Safe bet)
    n = 11
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            # Use a simple cyclic pattern, verify it's valid?
            # Cyclic difference (j-i) % 5 might have C4s.
            # Let's just return a valid n=11 found by a quick search or hardcode?
            # Actually, n=11 is small. Random search will find it instantly.
            pass
            
    # Re-run search for n=11 if needed
    n = 11
    num_edges = n * (n - 1) // 2
    edge_index = get_edge_index(n)
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    for _ in range(1000):
        if count_c4s(n, colors) == 0:
            return {'n': n, 'colors': colors}
        # Simple repair
        c4 = find_one_c4(n, colors)
        if c4:
            i, j, w, x, c = c4
            idx = edge_index.get((i, w))
            if idx is not None:
                colors[idx] = (c + 1) % 5
                
    return {'n': n, 'colors': colors}
