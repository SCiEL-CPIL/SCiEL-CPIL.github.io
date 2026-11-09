import time
import random
import math
from collections import defaultdict

def _compute_book_sizes(edges, n=40):
    """Compute the largest book size for the given edges."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    max_book = 0
    for u, v in edges:
        common = (adj[u] & adj[v]).bit_count()
        if common > max_book:
            max_book = common
    return max_book

def _build_structural_init(a_size, seed, target_edges=401):
    """
    Build a graph with a bipartite core A-B and a bipartite subgraph in B.
    A is independent. B is split into B1, B2.
    Uses a structured A-B connection to minimize intersections.
    """
    n = 40
    A = list(range(a_size))
    B = list(range(a_size, n))
    b_size = len(B)
    
    edges = []
    edge_set = set()
    
    # 1. Construct A-B edges
    # We want a bipartite graph with low max common neighbor count.
    # Use a "cyclic" construction.
    # Map B to 0..b_size-1 for indexing
    # We want to add roughly target_edges - max_possible_B_B edges in A-B?
    # Actually, we need to decide split.
    # Max edges in B-B (if bipartite) is (b_size//2) * (b_size - b_size//2).
    # Let's aim for a balanced split.
    
    # Heuristic: Start with a regular bipartite graph A-B.
    # Degree d for A and B.
    # Total edges E_AB = a_size * d_A = b_size * d_B.
    # Let's try to make E_AB large, and fill the rest with B-B.
    # But B-B edges increase book sizes of B-B edges.
    # A-B edges increase book sizes of B-B edges (via common neighbors in A).
    # A-B edges do NOT form triangles (A is independent).
    # So triangles are ONLY formed by edges in B-B.
    # Book size of edge (u,v) in B-B is |N_A(u) \cap N_A(v)|.
    # We want to minimize max |N_A(u) \cap N_A(v)| for edges (u,v) in B-B.
    
    # Strategy:
    # 1. Generate A-B edges such that intersections are small.
    # 2. Select B-B edges with the smallest intersections.
    
    # Step 1: A-B edges
    # Use random but structured.
    random.seed(seed)
    
    # Target number of A-B edges.
    # Let's say we use most of the budget for A-B.
    # Max B-B edges in a bipartite graph on B is roughly b_size^2 / 4.
    # If b_size=20, max B-B = 100.
    # So we need at least 401 - 100 = 301 A-B edges.
    # Max A-B edges is 20*20 = 400.
    # So we can have 300-390 A-B edges.
    
    # Let's pick a target A-B count.
    # Try to make it regular.
    target_ab = 350 # Example
    # Adjust target_ab to be feasible
    # We need to add (401 - target_ab) edges in B-B.
    # Max B-B edges is ~100. So target_ab must be >= 301.
    # Min B-B edges is 0. So target_ab <= 400.
    
    # Let's try a few values of target_ab
    best_init_edges = None
    best_init_score = float('inf')
    
    for target_ab in [320, 340, 360, 380]:
        needed_bb = target_edges - target_ab
        if needed_bb < 0 or needed_bb > (b_size * (b_size - 1)) // 2:
            continue
        
        # Check if target_ab is feasible for the bipartite graph A-B
        if target_ab > a_size * b_size:
            continue
            
        # Construct A-B
        # Shuffle B and assign
        # To minimize intersections, use a "spread" assignment.
        # For each a in A, connect to d random B's.
        d_a = target_ab // a_size
        remainder = target_ab % a_size
        
        ab_edges = []
        # To ensure low intersections, we can use a random permutation for each A node, 
        # but offset them.
        
        # Simple random might work if we verify.
        # Let's use a "greedy" random:
        # Initialize degrees of B to 0.
        deg_b = [0] * b_size
        # We want to keep deg_b balanced to minimize max intersection?
        # Actually, balanced degrees help.
        
        for i, a in enumerate(A):
            # Choose d_a or d_a+1 neighbors in B
            count = d_a + (1 if i < remainder else 0)
            # Randomly pick from B, but try to pick those with lower current degree
            # This balances degrees in B.
            # Sort B by degree, pick from top 'count' with some randomness
            # Or just random if we want simplicity.
            # Let's use random but weighted by inverse degree.
            
            # For speed, just random shuffle and pick
            b_indices = list(range(b_size))
            random.shuffle(b_indices)
            
            # Greedy pick: pick the one with min degree
            # This is O(b_size^2) which is fine for 20.
            selected = []
            for _ in range(count):
                # Find min degree in remaining
                min_idx = min(b_indices, key=lambda idx: deg_b[idx])
                selected.append(min_idx)
                deg_b[min_idx] += 1
                b_indices.remove(min_idx)
                
            for b_idx in selected:
                b_node = B[b_idx]
                ab_edges.append((a, b_node))
                edge_set.add((a, b_node))
                
        # Now we have A-B edges.
        # Calculate common neighbors for all pairs in B.
        adj_a = [0] * n
        for a, b in ab_edges:
            adj_a[b] |= (1 << a)
            
        common_neighbors = {}
        for i in range(b_size):
            for j in range(i+1, b_size):
                b_i = B[i]
                b_j = B[j]
                common = (adj_a[b_i] & adj_a[b_j]).bit_count()
                common_neighbors[(b_i, b_j)] = common
                
        # Select B-B edges
        # We need 'needed_bb' edges with smallest common_neighbors.
        # Sort pairs by common_neighbors
        pairs = sorted(common_neighbors.items(), key=lambda x: x[1])
        
        bb_edges = []
        for (u, v), count in pairs[:needed_bb]:
            bb_edges.append((u, v))
            edge_set.add((u, v))
            
        all_edges = [list(e) for e in ab_edges] + [list(e) for e in bb_edges]
        
        if len(all_edges) != target_edges:
            continue
            
        score = _compute_book_sizes(all_edges, n)
        if score < best_init_score:
            best_init_score = score
            best_init_edges = all_edges
            
    return best_init_edges

def _simulate_annealing(n=40, target_edges=401, time_limit=10.0, seed=42, initial_edges=None):
    """Simulated annealing with efficient bitmask updates."""
    start_time = time.time()
    random.seed(seed)
    
    if initial_edges is None:
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        initial_edges = [list(e) for e in all_possible[:target_edges]]
        
    current_edges = [list(e) for e in initial_edges]
    edge_set = set((min(u,v), max(u,v)) for u,v in current_edges)
    
    adj = [0] * n
    for u, v in current_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    book_sizes = {}
    current_max = 0
    for u, v in current_edges:
        key = (min(u,v), max(u,v))
        bs = (adj[u] & adj[v]).bit_count()
        book_sizes[key] = bs
        if bs > current_max:
            current_max = bs
            
    best_edges = [list(e) for e in current_edges]
    best_max = current_max
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    T0 = 5.0
    Tend = 0.01
    alpha = 0.9995
    
    T = T0
    while T > Tend and time.time() - start_time < time_limit:
        remove_idx = random.randint(0, len(current_edges) - 1)
        ru, rv = current_edges[remove_idx]
        rkey = (min(ru, rv), max(ru, rv))
        
        ckey = None
        for _ in range(10):
            cu, cv = random.choice(all_possible)
            ckey = (min(cu, cv), max(cu, cv))
            if ckey not in edge_set:
                break
        if ckey is None or ckey in edge_set:
            continue
            
        cu, cv = ckey
        
        affected = set()
        affected.add(rkey)
        affected.add(ckey)
        
        for v in range(n):
            if v == ru or v == rv or v == cu or v == cv:
                continue
            k = (min(v, ru), max(v, ru))
            if k in edge_set:
                affected.add(k)
            k = (min(v, rv), max(v, rv))
            if k in edge_set:
                affected.add(k)
            k = (min(v, cu), max(v, cu))
            if k in edge_set:
                affected.add(k)
            k = (min(v, cv), max(v, cv))
            if k in edge_set:
                affected.add(k)
                
        old_books = {k: book_sizes.get(k, 0) for k in affected}
        
        del book_sizes[rkey]
        edge_set.remove(rkey)
        adj[ru] &= ~(1 << rv)
        adj[rv] &= ~(1 << ru)
        
        edge_set.add(ckey)
        adj[cu] |= (1 << cv)
        adj[cv] |= (1 << cu)
        current_edges[remove_idx] = [cu, cv]
        
        new_max_affected = 0
        for k in affected:
            if k in edge_set:
                u, v = k
                bs = (adj[u] & adj[v]).bit_count()
                book_sizes[k] = bs
                if bs > new_max_affected:
                    new_max_affected = bs
            else:
                if k in book_sizes:
                    del book_sizes[k]
        
        was_max_edge = old_books.get(rkey, 0) == current_max
        if was_max_edge:
            new_max = max(book_sizes.values()) if book_sizes else 0
        else:
            new_max = max(current_max, new_max_affected)
            
        delta = new_max - current_max
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_edges = [list(e) for e in current_edges]
        else:
            edge_set.remove(ckey)
            adj[cu] &= ~(1 << cv)
            adj[cv] &= ~(1 << cu)
            if ckey in book_sizes:
                del book_sizes[ckey]
            
            edge_set.add(rkey)
            adj[ru] |= (1 << rv)
            adj[rv] |= (1 << ru)
            
            for k in affected:
                if k in old_books:
                    if k in edge_set:
                        book_sizes[k] = old_books[k]
                    else:
                        if k in book_sizes:
                            del book_sizes[k]
                else:
                    if k in book_sizes:
                        del book_sizes[k]
                        
            current_edges[remove_idx] = [ru, rv]
        
        T *= alpha
        
    return best_edges, best_max

def construct():
    start_time = time.time()
    n = 40
    target_edges = 401
    deadline = start_time + 145.0
    
    best_edges = None
    best_score = float('inf')
    
    # Try different A sizes
    # A=20 is balanced
    # A=15, 25 etc
    a_sizes = [15, 18, 20, 22, 25]
    seeds = [42, 123, 456, 789, 1024]
    
    for a_size in a_sizes:
        if time.time() > deadline - 5.0:
            break
            
        for seed in seeds:
            if time.time() > deadline - 5.0:
                break
                
            init_edges = _build_structural_init(a_size, seed)
            
            if init_edges is None:
                continue

            init_score = _compute_book_sizes(init_edges, n)
            
            if init_score < best_score:
                best_score = init_score
                best_edges = init_edges
                
            # Run SA with short time limit to polish
            time_limit = min(15.0, deadline - time.time() - 2.0)
            if time_limit < 1.0:
                continue
                
            edges, score = _simulate_annealing(n, target_edges, time_limit=time_limit, seed=seed, initial_edges=init_edges)
            
            if score < best_score:
                best_score = score
                best_edges = edges
                
    if best_edges is None:
        # Fallback: Random
        random.seed(0)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        best_edges = [list(e) for e in all_possible[:target_edges]]
        
    return best_edges
