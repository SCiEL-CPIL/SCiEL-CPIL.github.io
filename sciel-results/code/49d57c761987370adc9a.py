import time
import random
import sys

def construct():
    deadline = time.time() + 115
    
    # Helper for bit count
    def popcount(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')

    def local_search_bitmask(n, seed, deadline):
        rng = random.Random(seed)
        
        # Initialize random colors
        num_edges = n * (n - 1) // 2
        colors = [rng.randint(0, 3) for _ in range(num_edges)]
        
        # Build bitmasks: masks[i][c] is a bitmask of neighbors j such that color(i,j) == c
        # masks is a list of lists of ints
        masks = [[0] * 4 for _ in range(n)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                masks[i][c] |= (1 << j)
                masks[j][c] |= (1 << i)
                idx += 1
        
        # Calculate initial cost: number of monochromatic triangles
        # For each color c, count triangles
        cost = 0
        for c in range(4):
            # For each edge (i,j) of color c, count common neighbors in color c
            # This counts each triangle 3 times (once for each edge)
            # So total triangles = sum / 3
            # But we can just sum for all edges and divide by 3 at the end?
            # Actually, let's just sum |N_c(i) & N_c(j)| for all edges (i,j) of color c.
            # This counts each triangle 3 times.
            # We want the total number of triangles.
            # Let's compute sum of intersections for all pairs? No, only edges of color c.
            # For a fixed c, let E_c be the set of edges with color c.
            # Triangles in c are { (i,j,k) : (i,j),(j,k),(k,i) in E_c }.
            # Count = (1/3) * sum_{(i,j) in E_c} |N_c(i) & N_c(j)|.
            # We can maintain this sum.
            
            pass # We'll calculate in the loop below or precompute
            
        # Precompute cost
        # To be efficient, we can just calculate the total number of mono triangles.
        # For each i < j < k, check if all 3 edges have same color.
        # This is O(n^3) which is 50^3 = 125k. Fast enough for init.
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = colors[i * (n - 1) - i * (i + 1) // 2 + j - i - 1]
                for k in range(j + 1, n):
                    c_ik = colors[i * (n - 1) - i * (i + 1) // 2 + k - i - 1]
                    c_jk = colors[j * (n - 1) - j * (j + 1) // 2 + k - j - 1]
                    if c_ij == c_ik == c_jk:
                        cost += 1
                        
        T = 1.0
        T_min = 0.01
        start_time = time.time()
        
        while time.time() < deadline - 1:
            # Pick random edge
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            # Get current color
            # idx = i * (n - 1) - i * (i + 1) // 2 + j - i - 1
            idx = i * (n - 1) - i * (i + 1) // 2 + j - i - 1
            old_c = colors[idx]
            
            # Try other colors
            new_c = rng.randint(0, 3)
            while new_c == old_c:
                new_c = rng.randint(0, 3)
            
            # Calculate delta cost
            # Triangles removed (color old_c):
            # Count w such that w is neighbor of i in old_c AND w is neighbor of j in old_c
            # Note: w cannot be i or j. Since masks[i][old_c] contains j, and masks[j][old_c] contains i.
            # The intersection masks[i][old_c] & masks[j][old_c] will contain:
            # - w such that (i,w) is old_c and (j,w) is old_c.
            # - Does it contain i? No, masks[i] doesn't have i.
            # - Does it contain j? No, masks[j] doesn't have j.
            # Wait, masks[i][old_c] has bit j set. masks[j][old_c] has bit i set.
            # So (masks[i][old_c] & masks[j][old_c]) will NOT have bit i or j set.
            # It will have bits for all w != i,j such that both edges are old_c.
            # So popcount of intersection is exactly the number of mono-triangles of color old_c containing edge (i,j).
            removed = popcount(masks[i][old_c] & masks[j][old_c])
            
            # Triangles added (color new_c):
            # We are about to change (i,j) to new_c.
            # The new triangles will be w such that (i,w) is new_c and (j,w) is new_c.
            # Current masks[i][new_c] and masks[j][new_c] do NOT include the edge (i,j) yet.
            # So the intersection gives the number of w such that (i,w) and (j,w) are already new_c.
            # These will form triangles with the new edge (i,j).
            added = popcount(masks[i][new_c] & masks[j][new_c])
            
            delta = added - removed
            
            # Acceptance
            if delta <= 0 or rng.random() < pow(2.71828, -delta / T):
                # Accept move
                # Update colors
                colors[idx] = new_c
                
                # Update masks
                # Remove j from i's old_c mask, i from j's old_c mask
                masks[i][old_c] &= ~(1 << j)
                masks[j][old_c] &= ~(1 << i)
                
                # Add j to i's new_c mask, i to j's new_c mask
                masks[i][new_c] |= (1 << j)
                masks[j][new_c] |= (1 << i)
                
                cost += delta
                
                if cost == 0:
                    return colors
                
            # Cool down
            T *= 0.999
            if T < T_min:
                T = T_min
        
        return None

    # Try to find a valid coloring for n=50 down to 32
    for n in range(50, 31, -1):
        if time.time() > deadline - 5:
            break
        
        # Try a few seeds
        # For larger n, we might need more tries, but time is limited.
        # 50 is the target.
        num_tries = 10 if n == 50 else 5
        for seed in range(num_tries):
            if time.time() > deadline - 2:
                break
            
            result = local_search_bitmask(n, seed, deadline)
            if result is not None:
                # Verify (optional but safe)
                # The local search returns when cost==0, so it should be valid.
                return {'n': n, 'colors': result}
    
    # Fallback: Deterministic n=32 construction
    def _mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log = {}
    x = 1
    for e in range(15):
        log[x] = e
        x = _mul(x, 2)
        
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
                
    return {'n': 32, 'colors': colors}
