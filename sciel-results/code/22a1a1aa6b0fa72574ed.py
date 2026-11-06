import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try to find a valid coloring for n=17 using algebraic partition optimization
    # If successful, return it.
    # If not, fall back to smaller n.
    
    # The algebraic approach:
    # Vertices 0..16. Edge (u,v) color depends on (v-u) mod 17.
    # We partition the 16 non-zero residues into 4 sets of 4.
    
    n = 17
    elements = list(range(1, 17))
    
    # Initial partition: Quadratic residues split, non-residues split
    # Residues: 1, 2, 4, 8, 9, 13, 15, 16
    # Non-res: 3, 5, 6, 7, 10, 11, 12, 14
    
    # Let's start with a random partition or the one from the failed example
    # and try to optimize.
    
    def count_c6_for_set(S):
        """
        Count the number of distinct simple C6 cycles that can be formed
        using differences from the set S in Z_17.
        A cycle is a sequence of 6 differences d1..d6 in S such that
        sum(di) = 0 mod 17, and the partial sums are distinct and non-zero.
        """
        count = 0
        # Iterate over all 6-tuples from S
        # Since |S|=4, 4^6 = 4096 iterations. Very fast.
        for d1 in S:
            for d2 in S:
                for d3 in S:
                    for d4 in S:
                        for d5 in S:
                            # d6 must be -sum(d1..d5) mod 17
                            s5 = (d1 + d2 + d3 + d4 + d5) % 17
                            d6 = (-s5) % 17
                            if d6 in S:
                                # Check if this forms a simple cycle
                                # Vertices: 0, d1, d1+d2, d1+d2+d3, d1+d2+d3+d4, d1+d2+d3+d4+d5
                                # All must be distinct and non-zero (except 0 at start/end)
                                v0 = 0
                                v1 = d1
                                v2 = (v1 + d2) % 17
                                v3 = (v2 + d3) % 17
                                v4 = (v3 + d4) % 17
                                v5 = (v4 + d5) % 17
                                # v6 should be 0
                                # Check distinctness
                                verts = [v0, v1, v2, v3, v4, v5]
                                if len(set(verts)) == 6:
                                    # It's a simple cycle
                                    # To avoid counting rotations/reflections as distinct in a way that inflates cost,
                                    # we can just count it. The goal is to reach 0.
                                    count += 1
        return count

    def total_cost(partition):
        total = 0
        for S in partition:
            total += count_c6_for_set(S)
        return total

    # Initial partition
    # Try to find a partition with 0 C6s
    # We use a simple local search: swap elements between sets
    
    # Start with a random partition
    random.shuffle(elements)
    partition = [elements[0:4], elements[4:8], elements[8:12], elements[12:16]]
    
    # Ensure sets are sorted for consistency
    for i in range(4):
        partition[i] = sorted(partition[i])
        
    current_cost = total_cost(partition)
    
    # Local search
    best_partition = [list(S) for S in partition]
    best_cost = current_cost
    
    # Time limit for this search: 50 seconds
    search_deadline = start_time + 50.0
    
    while time.time() < search_deadline:
        if best_cost == 0:
            break
            
        # Pick a random set i and a random element x in it
        i = random.randint(0, 3)
        j = random.randint(0, 3)
        if i == j:
            continue
        k = random.randint(0, 3)
        x = partition[i][k]
        
        # Try moving x to set j
        new_partition = [list(S) for S in partition]
        new_partition[i].remove(x)
        new_partition[j].append(x)
        
        # Only recalculate cost for sets i and j
        old_cost_i = count_c6_for_set(partition[i])
        old_cost_j = count_c6_for_set(partition[j])
        new_cost_i = count_c6_for_set(new_partition[i])
        new_cost_j = count_c6_for_set(new_partition[j])
        
        delta = (new_cost_i - old_cost_i) + (new_cost_j - old_cost_j)
        
        # Accept if better or equal (to escape plateaus)
        if delta <= 0:
            partition = new_partition
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_partition = [list(S) for S in partition]
        else:
            # Simulated annealing style acceptance?
            # For simplicity, just reject.
            pass
            
    if best_cost == 0:
        # Construct the coloring
        colors = []
        # Map difference to color
        diff_to_color = {}
        for c, S in enumerate(best_partition):
            for d in S:
                diff_to_color[d] = c
                
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                # d is in 1..16
                c = diff_to_color[d]
                colors.append(c)
                
        return {'n': n, 'colors': colors}
        
    # If n=17 failed, try n=16 with a similar or simpler approach
    # Or just return a known valid smaller n
    # For n=13, we can use a simple greedy or known construction
    
    # Fallback: Return a valid coloring for n=13
    # A simple construction for n=13:
    # Use Z_13?
    # Or just use a random search for n=13 which is fast
    
    def find_small_n(n):
        # Backtracking for n<=13 is fast
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]
        
        def check_c6(u, v, c):
            nu = adj[c][u]
            nv = adj[c][v]
            # Path of length 4 between u and v
            # u-x1-x2-x3-x4-v
            x1_mask = nu
            while x1_mask:
                lsb = x1_mask & -x1_mask
                x1 = lsb.bit_length() - 1
                x1_mask ^= lsb
                if x1 == v: continue
                
                x4_mask = nv
                while x4_mask:
                    lsb4 = x4_mask & -x4_mask
                    x4 = lsb4.bit_length() - 1
                    x4_mask ^= lsb4
                    if x4 == u or x4 == x1: continue
                    
                    forbidden = (1 << u) | (1 << v) | (1 << x1) | (1 << x4)
                    nx1 = adj[c][x1] & ~forbidden
                    nx4 = adj[c][x4] & ~forbidden
                    
                    vx2 = nx1
                    while vx2:
                        lsb2 = vx2 & -vx2
                        x2 = lsb2.bit_length() - 1
                        vx2 ^= lsb2
                        if adj[c][x2] & nx4:
                            return True
            return False

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == num_edges:
                return True
            
            u, v = edges[idx]
            if idx == 0:
                c_vals = [0]
            else:
                c_vals = [0, 1, 2, 3]
                random.shuffle(c_vals)
                
            for c in c_vals:
                if check_c6(u, v, c):
                    continue
                    
                colors[idx] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                if backtrack(idx + 1):
                    return True
                    
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                colors[idx] = -1
                
            return False
            
        for _ in range(100):
            if time.time() > deadline:
                break
            colors = [-1] * num_edges
            adj = [[0] * n for _ in range(4)]
            if backtrack(0):
                return {'n': n, 'colors': colors}
        return None

    # Try n=16, 15, 14, 13
    for n in range(16, 12, -1):
        if time.time() > deadline - 10:
            break
        res = find_small_n(n)
        if res:
            return res
            
    # If all fails, return n=12 (very likely to succeed instantly)
    res = find_small_n(12)
    if res:
        return res
        
    # Absolute fallback: n=9 seed
    seed_n = 9
    seed_colors = [min(j - i, seed_n - j + i) - 1 for i in range(seed_n) for j in range(i + 1, seed_n)]
    return {'n': seed_n, 'colors': seed_colors}

if __name__ == "__main__":
    construct()
