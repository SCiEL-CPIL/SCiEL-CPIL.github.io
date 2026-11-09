import time
import random
import math

def construct():
    start_time = time.time()
    timeout = 100  # Safety margin
    
    # Precompute indices for edges
    def get_edge_indices(n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                edges.append((i, j))
        return edges

    def calculate_score(n, colors):
        # colors is a list of length n*(n-1)/2
        # Map edge (i,j) to index
        # idx = i*n - i*(i+1)/2 + (j-i-1) ... simpler to use dict or formula
        # Formula for 0-indexed triangle:
        # k = 0
        # for i in range(n):
        #   for j in range(i+1, n):
        #     idx[i][j] = k
        #     k += 1
        
        # Build adjacency bitmasks
        # adj[color][u] is a bitmask of neighbors
        adj = [[0] * n for _ in range(5)]
        
        idx_map = {}
        k = 0
        for i in range(n):
            for j in range(i+1, n):
                idx_map[(i, j)] = k
                idx_map[(j, i)] = k
                c = colors[k]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                k += 1
        
        score = 0
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    common = adj[c][u] & adj[c][v]
                    # Count bits
                    cnt = bin(common).count('1')
                    if cnt >= 2:
                        score += (cnt * (cnt - 1)) // 2
        return score

    def sa_solve(n, time_limit):
        edges = get_edge_indices(n)
        m = len(edges)
        
        # Initialize random
        colors = [random.randint(0, 4) for _ in range(m)]
        
        # Build initial adjacency and score
        # Reuse logic from calculate_score but keep adj for updates
        adj = [[0] * n for _ in range(5)]
        k = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[k]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                k += 1
                
        current_score = 0
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    common = adj[c][u] & adj[c][v]
                    cnt = bin(common).count('1')
                    if cnt >= 2:
                        current_score += (cnt * (cnt - 1)) // 2
        
        if current_score == 0:
            return colors
        
        T = 1.0
        cooling = 0.9999
        best_colors = colors[:]
        best_score = current_score
        
        # Precompute edge to index map for quick access
        # edge_to_idx
        e2i = {}
        k = 0
        for i in range(n):
            for j in range(i+1, n):
                e2i[(i, j)] = k
                k += 1
        
        while time.time() - start_time < time_limit:
            # Pick random edge
            idx = random.randint(0, m - 1)
            i, j = edges[idx]
            old_c = colors[idx]
            
            if random.random() < 0.1:
                new_c = random.randint(0, 4)
            else:
                new_c = (old_c + 1 + random.randint(0, 3)) % 5
            
            if old_c == new_c:
                continue
                
            # Calculate delta
            # 1. Remove (i,j) from old_c
            delta = 0
            
            # Neighbors in old_c
            A_old = adj[old_c][i]
            B_old = adj[old_c][j]
            
            # For each w in B_old (neighbors of j), N(i,w) decreases by 1
            # k = N(i,w) before
            # w is in A_old? No, w is neighbor of j.
            # We need N(i,w) = popcount(A_old & adj[old_c][w])
            # Since w is in B_old, (j,w) is edge.
            # We iterate w in B_old.
            mask = B_old
            while mask:
                lsb = mask & (-mask)
                w = lsb.bit_length() - 1
                mask ^= lsb
                if w == i: continue
                # k = popcount(A_old & adj[old_c][w])
                # Note: j is in A_old? No, (i,j) is edge, so j in A_old.
                # w is in B_old? Yes.
                # Is w in A_old? If (i,w) is edge in old_c.
                # Common neighbors of i and w in old_c:
                # Includes j if j is in adj[old_c][w] (which is true since w in B_old means (j,w) edge? No, B_old is neighbors of j. So (j,w) edge. So w in adj[old_c][j].)
                # We need common neighbors of i and w.
                # j is a common neighbor of i and w IF j is neighbor of i (true) AND j is neighbor of w (true, since w in B_old).
                # So j is one of the common neighbors.
                # Let K = A_old & adj[old_c][w].
                # k = popcount(K).
                # Since j in A_old and j in adj[old_c][w], j is in K.
                # So k >= 1.
                # After removing (i,j), j is no longer in adj[old_c][i].
                # So new k = k - 1.
                # Delta for pair (i,w) is - (k - 1).
                k = bin(A_old & adj[old_c][w]).count('1')
                delta -= (k - 1)
                
            # For each w in A_old (neighbors of i), N(j,w) decreases by 1
            mask = A_old
            while mask:
                lsb = mask & (-mask)
                w = lsb.bit_length() - 1
                mask ^= lsb
                if w == j: continue
                k = bin(B_old & adj[old_c][w]).count('1')
                delta -= (k - 1)
                
            # 2. Add (i,j) to new_c
            A_new = adj[new_c][i]
            B_new = adj[new_c][j]
            
            # For each w in B_new (neighbors of j in new_c), N(i,w) increases by 1
            mask = B_new
            while mask:
                lsb = mask & (-mask)
                w = lsb.bit_length() - 1
                mask ^= lsb
                if w == i: continue
                # k = popcount(A_new & adj[new_c][w])
                # j is in A_new? No, (i,j) not yet added.
                # But we are adding it.
                # Before adding, j is NOT in adj[new_c][i].
                # Is j in adj[new_c][w]? w in B_new means (j,w) edge in new_c. So yes.
                # So j is NOT a common neighbor of i and w before adding.
                # After adding, j is in adj[new_c][i]. And j in adj[new_c][w].
                # So j becomes a common neighbor.
                # k_new = k_old + 1.
                # Delta for pair (i,w) is + k_old.
                k = bin(A_new & adj[new_c][w]).count('1')
                delta += k
                
            # For each w in A_new (neighbors of i in new_c), N(j,w) increases by 1
            mask = A_new
            while mask:
                lsb = mask & (-mask)
                w = lsb.bit_length() - 1
                mask ^= lsb
                if w == j: continue
                k = bin(B_new & adj[new_c][w]).count('1')
                delta += k
                
            new_score = current_score + delta
            
            if new_score < current_score or random.random() < math.exp(-delta / T):
                # Accept
                current_score = new_score
                colors[idx] = new_c
                
                # Update adjacency
                adj[old_c][i] &= ~(1 << j)
                adj[old_c][j] &= ~(1 << i)
                adj[new_c][i] |= (1 << j)
                adj[new_c][j] |= (1 << i)
                
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
                    if best_score == 0:
                        return best_colors
                
                T *= cooling
                
        return best_colors if best_score == 0 else None

    # Try n from 26 down to 11
    for n in range(26, 10, -1):
        if time.time() - start_time > timeout - 5:
            break
        res = sa_solve(n, time.time() - start_time + 5 if time.time() - start_time < 5 else 5)
        # Adjust time limit logic: give 5s to each n? No, give remaining time.
        # Simpler: give 10s to n=26, 5s to n=25, etc.
        
        # Refined time management
        # I will just run sa_solve with a small time slice and check
        pass

    # Let's implement a simple loop
    # I'll try n=26 for 60s, then n=25 for 30s, etc.
    
    # Re-doing the loop properly
    start_time = time.time()
    for n in [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if time.time() - start_time > timeout:
            break
            
        # Time allocated
        alloc = 10 if n > 22 else 5
        if n == 26:
            alloc = 80 # Try harder for 26
            
        colors = sa_solve(n, alloc)
        if colors is not None:
            return {'n': n, 'colors': colors}
            
    # Fallback to seed
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
