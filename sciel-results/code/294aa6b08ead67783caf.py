import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    n = 40
    m_target = 401
    
    # We will work with a bitmask representation of the adjacency matrix.
    # adj[i] is an integer where bit j is 1 if edge (i,j) exists.
    
    def compute_max_book(adj):
        max_book = 0
        for i in range(n):
            mask_i = adj[i]
            # Only check edges (i, j) where j > i to avoid duplicates
            # We can iterate through bits
            # To be faster, we can precompute neighbor lists? 
            # No, bitmask AND is very fast.
            # Let's iterate j from i+1 to n-1
            for j in range(i + 1, n):
                if (mask_i >> j) & 1:
                    common = (adj[i] & adj[j]).bit_count()
                    if common > max_book:
                        max_book = common
        return max_book

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            # Only consider j > i
            mask &= ~((1 << (i+1)) - 1) 
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                edges.append([i, j])
                mask ^= lsb
        return edges

    def edge_count(adj):
        # Sum of bits in upper triangle
        count = 0
        for i in range(n):
            mask = adj[i]
            mask &= ~((1 << (i+1)) - 1)
            count += mask.bit_count()
        return count

    # Initialization: Circulant Graph
    # A circulant graph C_n(S) has edges (i, i+s) mod n for s in S.
    # For n=40, S is a subset of {1, ..., 19} (symmetric steps).
    # Degree is 2 * |S| (unless n is even and s=n/2, then degree is |S| + (n/2-1) ... wait)
    # If s != n/2, each s adds 2 to degree (i+s and i-s).
    # If s = n/2 = 20, it adds 1 to degree (self-loop? No, i+20 is distinct from i, but i-20 is same as i+20).
    # So if 20 in S, it contributes 1 to degree.
    # We want degree approx 20.
    # If we don't use 20, we need |S| = 10 for degree 20.
    # If we use 20, we need |S| = 10 including 20? Then degree = 2*9 + 1 = 19.
    # Or |S|=10 not including 20 -> degree 20.
    
    # Let's start with a random set S of size 10 from {1..19}.
    # This gives a 20-regular graph with 400 edges.
    # We need 401 edges. We will add 1 random edge during the search or initialization.
    
    best_adj = None
    best_score = float('inf')
    
    # Simulated Annealing parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.999 # Cooling rate per step? Or based on time?
    
    # Let's define the state as the adjacency matrix.
    # Moves:
    # 1. Swap an edge (u,v) with a non-edge (x,y).
    #    This keeps edge count constant.
    # 2. If edge count is not m_target, add/remove edge.
    
    # Strategy:
    # 1. Generate random circulant base (400 edges).
    # 2. Add 1 random edge to reach 401.
    # 3. Run SA to minimize max_book.
    
    for seed in range(50):
        if time.time() > deadline - 5:
            break
            
        random.seed(seed + 100)
        
        # 1. Construct Circulant Base
        # Choose 10 random shifts from 1..19
        shifts = random.sample(range(1, 20), 10)
        
        adj = [0] * n
        for s in shifts:
            for i in range(n):
                j1 = (i + s) % n
                j2 = (i - s) % n
                adj[i] |= (1 << j1)
                adj[i] |= (1 << j2)
        
        # This creates a 20-regular graph with 400 edges.
        # Add 1 random edge to make it 401.
        # Pick a random non-edge.
        # There are 780 total possible edges, 400 exist. 380 non-edges.
        # Just pick random u, v until we find a non-edge.
        while True:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if not ((adj[u] >> v) & 1):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                break
        
        current_score = compute_max_book(adj)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj.copy()
        
        # Simulated Annealing
        # We will perform swaps.
        # Objective: Minimize max_book.
        # To escape local minima, we accept worse moves with probability exp(-delta/T).
        
        T = T_start
        # Number of steps per restart: let's use time-based stopping
        restart_start = time.time()
        max_steps = 20000 # Approx
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Choose a random edge (u, v) to remove
            # And a random non-edge (x, y) to add
            
            # Efficient way to pick random edge:
            # Precompute list of edges? No, that's slow to update.
            # Just sample random u, v.
            
            attempts = 0
            while attempts < 10:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v and ((adj[u] >> v) & 1):
                    break
                attempts += 1
            else:
                continue # Failed to find edge? Unlikely.
            
            attempts = 0
            while attempts < 10:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x != y and not ((adj[x] >> y) & 1):
                    break
                attempts += 1
            else:
                continue # Failed to find non-edge? Unlikely.
            
            # Ensure we are not swapping the same edge (u,v) with itself (x,y) or (y,x)
            if (u == x and v == y) or (u == y and v == x):
                continue
            
            # Perform Swap
            # Remove (u, v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            # Add (x, y)
            adj[x] |= (1 << y)
            adj[y] |= (1 << x)
            
            new_score = compute_max_book(adj)
            
            # Acceptance Criteria
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj.copy()
            else:
                # Delta = new_score - current_score (positive)
                # Probability to accept
                # Since scores are integers, delta is at least 1.
                # exp(-1/T)
                if T > 0:
                    prob = np.exp(-1.0 / T)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        # Revert
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        adj[x] &= ~(1 << y)
                        adj[y] &= ~(1 << x)
            
            # Cool down
            # Time-based cooling is better
            elapsed_in_restart = time.time() - restart_start
            # Simple linear or exponential decay
            T = T_start * (0.5 ** (elapsed_in_restart / 5.0)) # Half life 5s?
            # Or just step based
            # T *= 0.9995
            
            if current_score == 7:
                # Theoretical minimum
                break

    if best_adj is None:
        # Fallback: K_20,20 + 1 edge
        adj = [0] * n
        for i in range(20):
            for j in range(20, 40):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        adj[0] |= (1 << 1)
        adj[1] |= (1 << 0)
        best_adj = adj

    return get_edge_list(best_adj)
