import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Fallback solution (n=11) from seed
    def get_seed_solution(n):
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
    
    # Try to find larger n using Simulated Annealing
    # We know n=26 is possible.
    # We will try n from 26 down to 12.
    
    for n in range(26, 11, -1):
        if time.time() > deadline - 15: # Need at least 15s for one try?
            # If time is short, skip to smaller n or fallback
            if time.time() > deadline - 5:
                break
        
        # Run SA for this n
        solution = find_valid_coloring_sa(n, time_limit=10)
        if solution is not None:
            return solution
            
    # If we didn't find anything better than 11, return seed
    return get_seed_solution(11)

def find_valid_coloring_sa(n, time_limit=10):
    """
    Uses Simulated Annealing to find a 5-coloring of K_n with no C4 in any color.
    Returns dict with 'n' and 'colors' if found, else None.
    """
    num_edges = n * (n - 1) // 2
    if num_edges == 0:
        return {'n': n, 'colors': []}
        
    # Initialize random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Precompute edge indices
    # edge_index[i][j] = index in colors array for pair (i, j) with i < j
    edge_index = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[i][j] = idx
            edge_index[j][i] = idx
            idx += 1
            
    # Adjacency lists for each color
    # adj[c][u] = list of neighbors v such that (u, v) is color c
    adj = [[] for _ in range(5)]
    for c in range(5):
        adj[c] = [[] for _ in range(n)]
        
    # Initialize adj
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[edge_index[i][j]]
            adj[c][i].append(j)
            adj[c][j].append(i)
            
    # Function to calculate total C4 count
    def count_c4s():
        total = 0
        for c in range(5):
            # For each pair u < v, count common neighbors in color c
            for u in range(n):
                # Use set intersection for speed? n is small, list is fine
                # But bitsets are faster if we use them.
                # Let's stick to list for now, n=26 is small.
                nu = adj[c][u]
                for v in range(u + 1, n):
                    nv = adj[c][v]
                    # Count common neighbors
                    # Since n is small, set intersection is okay
                    common = len(set(nu) & set(nv))
                    if common >= 2:
                        total += common * (common - 1) // 2
        return total

    # Optimized count using incremental updates is better, but let's try full first to see if it's fast enough.
    # n=26, pairs = 325. 5 colors. 1625 pairs.
    # Set intersection of lists of size ~5.
    # This might be slow in Python.
    
    # Let's use a bitset representation for adjacency for faster intersection.
    # adj_bit[c][u] is an integer where bit v is set if (u, v) is color c.
    adj_bit = [[0] * n for _ in range(5)]
    
    def rebuild_adj_bit():
        for c in range(5):
            for u in range(n):
                adj_bit[c][u] = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_index[i][j]]
                adj_bit[c][i] |= (1 << j)
                adj_bit[c][j] |= (1 << i)
                
    rebuild_adj_bit()
    
    def count_c4s_fast():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = bin(adj_bit[c][u] & adj_bit[c][v]).count('1')
                    if common >= 2:
                        total += common * (common - 1) // 2
        return total

    # Initial energy
    energy = count_c4s_fast()
    
    if energy == 0:
        return {'n': n, 'colors': colors}

    # SA Parameters
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    start_t = time.time()
    
    # Precompute neighbors for quick access in update?
    # We will do incremental update of energy.
    
    # State: colors array, adj_bit, energy
    # Move: change color of edge (u, v) from c1 to c2
    
    while time.time() < start_t + time_limit:
        if T < min_T:
            T = min_T
            
        # Pick random edge
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        idx = edge_index[i][j]
        c1 = colors[idx]
        c2 = random.randint(0, 4)
        if c1 == c2:
            continue
            
        # Calculate delta energy
        # Removing (i, j) from c1, adding to c2
        
        def delta_energy_remove(i, j, c):
            # i, j are connected in color c
            # This affects common neighbors of (i, k) for k in N_c(j)
            # and common neighbors of (j, k) for k in N_c(i)
            # And common neighbors of (i, j) itself? No, (i, j) edge doesn't contribute to common neighbors of (i, j).
            
            delta = 0
            # Neighbors of j in color c (excluding i)
            Nj = [k for k in range(n) if k != i and (adj_bit[c][j] & (1 << k))]
            # Neighbors of i in color c (excluding j)
            Ni = [k for k in range(n) if k != j and (adj_bit[c][i] & (1 << k))]
            
            # For each k in Nj, the pair (i, k) loses common neighbor j
            for k in Nj:
                old_common = bin(adj_bit[c][i] & adj_bit[c][k]).count('1')
                new_common = old_common - 1
                delta -= (old_common * (old_common - 1) // 2) - (new_common * (new_common - 1) // 2)
                
            # For each k in Ni, the pair (j, k) loses common neighbor i
            for k in Ni:
                old_common = bin(adj_bit[c][j] & adj_bit[c][k]).count('1')
                new_common = old_common - 1
                delta -= (old_common * (old_common - 1) // 2) - (new_common * (new_common - 1) // 2)
                
            # Note: The pair (i, j) itself is not affected in terms of common neighbors count?
            # d(i, j) counts z such that (i, z) and (z, j) are in c.
            # Removing edge (i, j) does not change d(i, j).
            
            return delta

        def delta_energy_add(i, j, c):
            # Adding (i, j) to color c
            # Similar to remove but signs are flipped
            delta = 0
            Nj = [k for k in range(n) if k != i and (adj_bit[c][j] & (1 << k))]
            Ni = [k for k in range(n) if k != j and (adj_bit[c][i] & (1 << k))]
            
            for k in Nj:
                old_common = bin(adj_bit[c][i] & adj_bit[c][k]).count('1')
                new_common = old_common + 1
                delta += (new_common * (new_common - 1) // 2) - (old_common * (old_common - 1) // 2)
                
            for k in Ni:
                old_common = bin(adj_bit[c][j] & adj_bit[c][k]).count('1')
                new_common = old_common + 1
                delta += (new_common * (new_common - 1) // 2) - (old_common * (old_common - 1) // 2)
                
            return delta

        dE = delta_energy_remove(i, j, c1) + delta_energy_add(i, j, c2)
        
        if dE <= 0 or random.random() < math.exp(-dE / T):
            # Accept move
            # Update state
            colors[idx] = c2
            
            # Update adj_bit
            adj_bit[c1][i] &= ~(1 << j)
            adj_bit[c1][j] &= ~(1 << i)
            adj_bit[c2][i] |= (1 << j)
            adj_bit[c2][j] |= (1 << i)
            
            energy += dE
            
            if energy == 0:
                return {'n': n, 'colors': colors}
                
        T *= cooling
        
    return None
# EVOLVE-BLOCK-END
