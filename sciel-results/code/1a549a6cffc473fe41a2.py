import random
import time
from itertools import combinations

def construct():
    CLIQUE = 5
    INDEPENDENT = 5
    MAX_VERTICES = 46
    
    # Precompute 5-sets for a given N? 
    # We will do it inside the loop for each N.
    
    best_n = 0
    best_edges = []
    
    # Try to find a graph for N = 42 down to 38
    # 37 is the seed, so we can fallback to it.
    
    for N in range(42, 36, -1):
        # Precompute all 5-combinations
        all_5_sets = list(combinations(range(N), 5))
        
        # Initialize random graph
        # Use Paley-like structure or just random?
        # Random with p=0.5 is often a good start for local search.
        nb = [0] * N
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < 0.5:
                    nb[i] |= (1 << j)
                    nb[j] |= (1 << i)
                    
        def get_penalty(nb):
            k5 = 0
            i5 = 0
            for S in all_5_sets:
                mask = 0
                for v in S:
                    mask |= (1 << v)
                
                is_k5 = True
                is_i5 = True
                for v in S:
                    others = mask ^ (1 << v)
                    # Check clique: all others must be in nb[v]
                    if (nb[v] & others) != others:
                        is_k5 = False
                    # Check independent: no others in nb[v]
                    if (nb[v] & others) != 0:
                        is_i5 = False
                        
                if is_k5:
                    k5 += 1
                if is_i5:
                    i5 += 1
            return k5 + i5, k5, i5
            
        def find_bad_set(nb):
            # Returns a bad 5-set or None
            # We can shuffle the list to avoid bias
            # But shuffling 850k items is expensive.
            # Just iterate in order? Or random index?
            # Let's just iterate. If penalty is high, we find one fast.
            for S in all_5_sets:
                mask = 0
                for v in S:
                    mask |= (1 << v)
                
                is_k5 = True
                is_i5 = True
                for v in S:
                    others = mask ^ (1 << v)
                    if (nb[v] & others) != others:
                        is_k5 = False
                    if (nb[v] & others) != 0:
                        is_i5 = False
                        
                if is_k5 or is_i5:
                    return S
            return None

        current_penalty, _, _ = get_penalty(nb)
        
        # Local Search
        max_iters = 5000
        no_improve = 0
        start_time = time.time()
        
        while no_improve < 100 and max_iters > 0 and (time.time() - start_time) < 20:
            max_iters -= 1
            
            bad_set = find_bad_set(nb)
            if bad_set is None:
                break # Found valid graph
                
            # Pick a random edge in this bad set to flip
            # There are 10 edges.
            # Generate all pairs in bad_set
            pairs = list(combinations(bad_set, 2))
            u, v = random.choice(pairs)
            
            # Flip edge (u, v)
            # Save state
            old_u = nb[u]
            old_v = nb[v]
            
            if nb[u] & (1 << v):
                # Remove edge
                nb[u] ^= (1 << v)
                nb[v] ^= (1 << u)
            else:
                # Add edge
                nb[u] |= (1 << v)
                nb[v] |= (1 << u)
                
            new_penalty, _, _ = get_penalty(nb)
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                no_improve = 0
            else:
                # Revert
                nb[u] = old_u
                nb[v] = old_v
                no_improve += 1
        
        if current_penalty == 0:
            # Construct edges list
            edges = []
            for i in range(N):
                for j in range(i + 1, N):
                    if nb[i] & (1 << j):
                        edges.append((i, j))
            best_n = N
            best_edges = edges
            break
            
    # Fallback
    if best_n == 0:
        # Paley 37
        N = 37
        edges = []
        Q = set()
        for i in range(1, N):
            Q.add((i * i) % N)
        for i in range(N):
            for j in range(i + 1, N):
                if (j - i) % N in Q:
                    edges.append((i, j))
        best_n = N
        best_edges = edges
        
    return best_n, best_edges
