"""Program for finding colourings of K_n without monochromatic K2,2,2."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _paley_coloring(p):
    """Generate Paley coloring for prime p ≡ 1 (mod 4)."""
    squares = set(x * x % p for x in range(1, p))
    adj = np.zeros((p, p), dtype=np.int8)
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            c = 0 if (j - i) % p in squares else 1
            colors.append(c)
            adj[i][j] = c
            adj[j][i] = c
    return p, colors, adj

def _construct_30_vertex():
    """
    Try to find a 30-vertex coloring by extending Paley 29 with Simulated Annealing.
    Uses a faster, more targeted cost function.
    """
    p = 29
    squares = set(x * x % p for x in range(1, p))
    adj29 = np.zeros((29, 29), dtype=np.int8)
    for i in range(29):
        for j in range(i + 1, 29):
            c = 0 if (j - i) % p in squares else 1
            adj29[i][j] = c
            adj29[j][i] = c
    
    # Precompute neighbor masks for the 29-vertex graph
    neigh_mask = [[0]*29 for _ in range(2)]
    for i in range(29):
        for j in range(29):
            if i != j:
                neigh_mask[adj29[i][j]][i] |= (1 << j)

    def _count_k222_fast(mask1):
        """
        Count number of monochromatic K2,2,2 involving vertex 29.
        mask1: 29-bit integer. Bit j is 1 if edge (29, j) is color 1.
        """
        count = 0
        all_mask = (1 << 29) - 1
        
        for c in [0, 1]:
            # S_c is the set of vertices v such that (29, v) has color c
            if c == 1:
                S_c = mask1
            else:
                S_c = all_mask ^ mask1
            
            # We need to find K2,2 in the subgraph induced by S_c (in color c)
            # such that one of the parts is {29, a}.
            # This means we need a in S_c, and b1, b2, c1, c2 in S_c
            # such that edges (a,b1), (a,b2), (a,c1), (a,c2) are color c,
            # and (b1,c1), (b1,c2), (b2,c1), (b2,c2) are color c.
            
            # For a fixed a in S_c, let Cand = S_c & neigh_mask[c][a].
            # We need a K2,2 in Cand.
            # A K2,2 exists in Cand iff there exist b1, b2 in Cand
            # such that |neigh_mask[c][b1] & neigh_mask[c][b2] & Cand| >= 2.
            
            # Optimization: Iterate over a in S_c.
            # For each a, compute Cand. If |Cand| < 4, skip.
            # Then check for K2,2 in Cand.
            
            # To speed up, we can precompute for each pair (u, v) in the 29 vertices,
            # the common neighbors in color c.
            # common_c[u][v] = neigh_mask[c][u] & neigh_mask[c][v]
            
            # But that's 29*29*2 precomputation, which is fine.
            # Actually, we can just compute it on the fly or precompute globally.
            # Let's precompute common_masks[c][u][v] = neigh_mask[c][u] & neigh_mask[c][v]
            
            # Wait, we can just iterate.
            # The bottleneck is the inner loop over pairs in Cand.
            
            # Let's try to optimize the K2,2 check in Cand.
            # If we represent Cand as a list of vertices, we can check pairs.
            
            # Alternative:
            # For each pair (b1, b2) in S_c, let Common = neigh_mask[c][b1] & neigh_mask[c][b2] & S_c.
            # If |Common| >= 2, then for every a in Common, we have a K2,2,2 with parts {29,a} and {b1,b2} and {c1,c2}.
            # Wait, no. The K2,2,2 structure is:
            # Part 1: {29, a}
            # Part 2: {b1, b2}
            # Part 3: {c1, c2}
            # Edges:
            # 29-b1, 29-b2, 29-c1, 29-c2 (color c) -> b1,b2,c1,c2 in S_c
            # a-b1, a-b2, a-c1, a-c2 (color c) -> a in S_c, and b1,b2,c1,c2 in neigh_mask[c][a]
            # b1-c1, b1-c2, b2-c1, b2-c2 (color c) -> c1,c2 in neigh_mask[c][b1] & neigh_mask[c][b2]
            
            # So, for a fixed pair (b1, b2) in S_c:
            # Let C = neigh_mask[c][b1] & neigh_mask[c][b2] & S_c.
            # These are the candidates for c1, c2.
            # Let A = neigh_mask[c][b1] & neigh_mask[c][b2] & S_c & (intersection of neigh_mask[c][a] for all a in ...? No)
            
            # Actually, the condition is:
            # We need to choose a, c1, c2 such that:
            # a in S_c
            # c1, c2 in S_c
            # a, c1, c2 in neigh_mask[c][b1]
            # a, c1, c2 in neigh_mask[c][b2]
            # So a, c1, c2 are in Common = neigh_mask[c][b1] & neigh_mask[c][b2] & S_c.
            # We need to choose 3 distinct vertices from Common: a, c1, c2.
            # The number of ways is C(|Common|, 3).
            
            # So the total count for this c is:
            # Sum over all pairs (b1, b2) in S_c of C(|neigh_mask[c][b1] & neigh_mask[c][b2] & S_c|, 3).
            
            # This is much faster!
            # We just iterate over pairs (b1, b2) in S_c, compute the common mask,
            # count the bits, and add C(k, 3).
            
            # Extract list of vertices in S_c
            s_list = [v for v in range(29) if (S_c >> v) & 1]
            
            for i in range(len(s_list)):
                b1 = s_list[i]
                for j in range(i + 1, len(s_list)):
                    b2 = s_list[j]
                    common = neigh_mask[c][b1] & neigh_mask[c][b2] & S_c
                    k = bin(common).count('1')
                    if k >= 3:
                        count += k * (k - 1) * (k - 2) // 6
                        
        return count

    # Simulated Annealing
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initialize mask randomly
    mask = 0
    for i in range(29):
        if random.randint(0, 1) == 1:
            mask |= (1 << i)
            
    current_cost = _count_k222_fast(mask)
    best_mask = mask
    best_cost = current_cost
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        if best_cost == 0:
            break
            
        # Flip a random bit
        bit = random.randint(0, 28)
        new_mask = mask ^ (1 << bit)
        new_cost = _count_k222_fast(new_mask)
        
        delta = new_cost - current_cost
        
        if delta < 0 or (T > 0.1 and random.random() < np.exp(-delta / T)):
            mask = new_mask
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_mask = mask
                
        T *= cooling
        
    if best_cost == 0:
        # Construct the 30-vertex coloring
        colors = []
        # First, edges within 0..28
        for i in range(29):
            for j in range(i + 1, 29):
                colors.append(adj29[i][j])
        # Edges involving 29
        for j in range(29):
            if (best_mask >> j) & 1:
                colors.append(1)
            else:
                colors.append(0)
                
        return 30, colors

    return None

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0
    
    # Try to find 30-vertex coloring
    result = _construct_30_vertex()
    if result is not None:
        n, colors = result
        return {'n': n, 'colors': colors}
        
    # Fallback to Paley 29
    n, colors, adj = _paley_coloring(29)
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
