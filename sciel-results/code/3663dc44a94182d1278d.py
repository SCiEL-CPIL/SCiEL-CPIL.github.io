"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _compute_book_sizes(edges, n=40):
    """Compute the book size for each edge in the graph."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
        
    book_sizes = {}
    max_book = 0
    for u, v in edges:
        key = (min(u,v), max(u,v))
        common = adj[u] & adj[v]
        book_sizes[key] = len(common)
        if book_sizes[key] > max_book:
            max_book = book_sizes[key]
            
    return max_book, book_sizes, adj, edge_set

def _structural_init(n=40, target_edges=401, seed=42):
    """
    Create a structurally good initial graph using a 9/31 partition
    with degree-constrained intra-partition edges.
    
    Strategy:
    - Partition: A = 0..8 (9 vertices), B = 9..39 (31 vertices)
    - Add all A-B edges: 9*31 = 279 edges
    - Add 122 edges within B, split B into B1 (15) and B2 (16)
    - Use degree constraints to keep book sizes low
    """
    random.seed(seed)
    
    A = list(range(9))
    B = list(range(9, 40))
    
    edges = set()
    
    # Add all edges between A and B (279 edges)
    for a in A:
        for b in B:
            edges.add((a, b))
    
    # Split B into B1 and B2
    B1 = list(range(9, 24))   # 15 vertices
    B2 = list(range(24, 40))  # 16 vertices
    
    needed = target_edges - len(edges)  # 401 - 279 = 122
    added = 0
    
    deg_b1 = {v: 0 for v in B1}
    deg_b2 = {v: 0 for v in B2}
    
    # Generate all B1-B2 pairs and shuffle
    pairs = [(u, v) for u in B1 for v in B2]
    random.shuffle(pairs)
    
    # Add edges with degree constraints (max degree ~9)
    max_deg = 9
    for u, v in pairs:
        if added >= needed:
            break
        if deg_b1[u] < max_deg and deg_b2[v] < max_deg:
            edges.add((u, v))
            deg_b1[u] += 1
            deg_b2[v] += 1
            added += 1
    
    # If we didn't reach the target, fill remaining without constraints
    if added < needed:
        for u, v in pairs:
            if added >= needed:
                break
            key = (min(u, v), max(u, v))
            if key not in edges:
                edges.add(key)
                added += 1
    
    # If still not enough (shouldn't happen), add random intra-B edges
    if added < needed:
        all_b_pairs = [(i, j) for i in range(9, 40) for j in range(i+1, 40)]
        random.shuffle(all_b_pairs)
        for u, v in all_b_pairs:
            if added >= needed:
                break
            key = (min(u, v), max(u, v))
            if key not in edges:
                edges.add(key)
                added += 1
    
    # Convert to list of [u, v] pairs
    final_edges = []
    for u, v in edges:
        if u < v:
            final_edges.append([u, v])
        else:
            final_edges.append([v, u])
    
    # Ensure exactly target_edges
    if len(final_edges) > target_edges:
        final_edges = final_edges[:target_edges]
    elif len(final_edges) < target_edges:
        # Add more random edges
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        existing = set((min(u,v), max(u,v)) for u, v in final_edges)
        random.shuffle(all_possible)
        for u, v in all_possible:
            if len(final_edges) >= target_edges:
                break
            key = (u, v)
            if key not in existing:
                final_edges.append([u, v])
                existing.add(key)
    
    return final_edges

def construct():
    start_time = time.time()
    n = 40
    target_edges = 401
    time_limit = 115.0  # Safety margin
    
    # Use structural initialization with 9/31 partition
    initial_edges = _structural_init(n, target_edges)
    
    max_book, book_sizes, adj, edge_set = _compute_book_sizes(initial_edges, n)
    
    best_edges = list(initial_edges)
    best_score = max_book
    
    # Simulated Annealing parameters
    T = 3.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    current_edges = list(initial_edges)
    current_max = max_book
    current_book_sizes = dict(book_sizes)
    current_adj = {k: set(v) for k, v in adj.items()}
    current_edge_set = set(edge_set)
    
    # Build index from edge key to position in current_edges
    edge_index = {}
    for i, e in enumerate(current_edges):
        key = (min(e[0], e[1]), max(e[0], e[1]))
        edge_index[key] = i
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Track stagnation
    no_improve_count = 0
    stagnation_limit = 3000
    
    iterations = 0
    while time.time() - start_time < time_limit:
        iterations += 1
        
        # Check time more frequently
        if time.time() - start_time > time_limit - 5:
            break
            
        # Stagnation handling
        if no_improve_count > stagnation_limit:
            # Perturbation: make 5 random swaps
            for _ in range(5):
                idx = random.randint(0, len(current_edges) - 1)
                u1, v1 = current_edges[idx]
                rem_key = (min(u1, v1), max(u1, v1))
                
                while True:
                    add_u, add_v = random.choice(all_possible_edges)
                    add_key = (min(add_u, add_v), max(add_u, add_v))
                    if add_key not in current_edge_set:
                        break
                
                # Apply swap without checking acceptance
                current_edge_set.remove(rem_key)
                current_edge_set.add(add_key)
                current_adj[u1].discard(v1)
                current_adj[v1].discard(u1)
                current_adj[add_u].add(add_v)
                current_adj[add_v].add(add_u)
                current_edges[idx] = [add_u, add_v]
                
                # Update edge index
                del edge_index[rem_key]
                edge_index[add_key] = idx
                
                # Update book sizes
                affected = {rem_key, add_key}
                for w in list(current_adj[u1]):
                    k = (min(u1, w), max(u1, w))
                    if k in current_edge_set:
                        affected.add(k)
                for w in list(current_adj[v1]):
                    k = (min(v1, w), max(v1, w))
                    if k in current_edge_set:
                        affected.add(k)
                for w in list(current_adj[add_u]):
                    k = (min(add_u, w), max(add_u, w))
                    if k in current_edge_set:
                        affected.add(k)
                for w in list(current_adj[add_v]):
                    k = (min(add_v, w), max(add_v, w))
                    if k in current_edge_set:
                        affected.add(k)
                
                for k in affected:
                    if k in current_edge_set:
                        u, v = k
                        current_book_sizes[k] = len(current_adj[u] & current_adj[v])
                    elif k in current_book_sizes:
                        del current_book_sizes[k]
                
                current_max = max(current_book_sizes.values()) if current_book_sizes else 0
            
            no_improve_count = 0
            T = max(T * 2, 3.0)  # Reheat
        
        # Pick an edge to remove - bias toward high-book edges
        if random.random() < 0.7 and current_book_sizes:
            # Pick from edges with above-average book size
            avg_book = sum(current_book_sizes.values()) / len(current_book_sizes)
            high_book_edges = [k for k, v in current_book_sizes.items() if v > avg_book]
            if high_book_edges:
                rem_key = random.choice(high_book_edges)
            else:
                rem_key = random.choice(list(current_book_sizes.keys()))
        else:
            rem_key = random.choice(list(current_book_sizes.keys()))
        
        u1, v1 = rem_key
        # Find index using the index
        idx = edge_index.get(rem_key, -1)
        if idx == -1:
            continue
        
        # Pick a non-edge to add - bias toward edges that won't create large books
        candidates = []
        for _ in range(15):  # Sample 15 candidates for speed
            add_u, add_v = random.choice(all_possible_edges)
            add_key = (min(add_u, add_v), max(add_u, add_v))
            if add_key not in current_edge_set:
                # Estimate book size if added
                common = current_adj[add_u] & current_adj[add_v]
                est_book = len(common)
                candidates.append((est_book, add_key))
        
        if not candidates:
            continue
        
        # Pick the candidate with minimum estimated book size
        candidates.sort()
        _, add_key = candidates[0]
        add_u, add_v = add_key
        
        # Calculate affected edges
        affected_keys = set()
        affected_keys.add(rem_key)
        affected_keys.add(add_key)
        
        for w in list(current_adj[u1]):
            if w != v1:
                k = (min(u1, w), max(u1, w))
                if k in current_edge_set:
                    affected_keys.add(k)
        for w in list(current_adj[v1]):
            if w != u1:
                k = (min(v1, w), max(v1, w))
                if k in current_edge_set:
                    affected_keys.add(k)
        for w in list(current_adj[add_u]):
            if w != add_v:
                k = (min(add_u, w), max(add_u, w))
                if k in current_edge_set:
                    affected_keys.add(k)
        for w in list(current_adj[add_v]):
            if w != add_u:
                k = (min(add_v, w), max(add_v, w))
                if k in current_edge_set:
                    affected_keys.add(k)
        
        old_values = {k: current_book_sizes.get(k, 0) for k in affected_keys}
        old_max = current_max
        
        # Apply swap
        current_edge_set.remove(rem_key)
        current_adj[u1].remove(v1)
        current_adj[v1].remove(u1)
        current_edges[idx] = [add_u, add_v]
        
        current_edge_set.add(add_key)
        current_adj[add_u].add(add_v)
        current_adj[add_v].add(add_u)
        
        # Update edge index
        del edge_index[rem_key]
        edge_index[add_key] = idx
        
        # Update book sizes
        for k in affected_keys:
            if k in current_edge_set:
                u, v = k
                current_book_sizes[k] = len(current_adj[u] & current_adj[v])
            else:
                if k in current_book_sizes:
                    del current_book_sizes[k]
        
        new_max = max(current_book_sizes.values()) if current_book_sizes else 0
        
        # Acceptance
        delta = new_max - current_max
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = True
        else:
            accept = random.random() < pow(2.718, -delta / T)
        
        if accept:
            current_max = new_max
            if new_max < best_score:
                best_score = new_max
                best_edges = list(current_edges)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            current_edge_set.remove(add_key)
            current_edge_set.add(rem_key)
            
            current_adj[add_u].remove(add_v)
            current_adj[add_v].remove(add_u)
            current_adj[u1].add(v1)
            current_adj[v1].add(u1)
            
            current_edges[idx] = [u1, v1]
            
            # Update edge index
            del edge_index[add_key]
            edge_index[rem_key] = idx
            
            for k, val in old_values.items():
                if k in current_edge_set:
                    current_book_sizes[k] = val
                else:
                    if k in current_book_sizes:
                        del current_book_sizes[k]
            
            current_max = old_max
        
        T = max(T * cooling_rate, T_min)
        
    return best_edges

# EVOLVE-BLOCK-END
