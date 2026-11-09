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
    """Create a structurally good initial graph using bipartite split."""
    random.seed(seed)
    
    # Split into A and B
    A = list(range(20))
    B = list(range(20, 40))
    
    edges = []
    edge_set = set()
    
    # Start with most bipartite edges
    bipartite_edges = [(a, b) for a in A for b in B]
    random.shuffle(bipartite_edges)
    
    # We need 401 edges total
    # Use 400 bipartite edges + 1 intra-partition edge? 
    # But that gives book size 20 for the intra edge.
    # Better: use fewer bipartite edges and more intra edges, but strategically.
    
    # Actually, let's try: use all 400 bipartite edges, then add 1 intra edge.
    # But to reduce book size, we can remove some bipartite edges.
    
    # Strategy: Start with K_{20,20}, then remove some edges to create "gaps"
    # that reduce the common neighbors of intra-partition edges.
    
    # Let's use a different approach:
    # 1. Create a random bipartite graph with ~395 edges
    # 2. Add ~6 intra-partition edges
    # 3. Optimize with local search
    
    # Actually, the cell finding says: "split A/B with degree-constrained intra-B edges"
    # So maybe: keep A-B edges mostly complete, but add intra-B edges with constraints.
    
    # Let me try: 
    # - Keep all 400 A-B edges
    # - Add 1 A-A edge
    # - Remove some A-B edges to reduce the book size of that A-A edge
    
    # For an edge (u,v) in A, book size = |N(u) ∩ N(v)|
    # In K_{20,20}, N(u) = B, N(v) = B, so |N(u) ∩ N(v)| = 20
    # To reduce this, we need to remove edges so that not all B vertices are connected to both u and v.
    
    # If we remove edge (u, b) for some b in B, then b is no longer in N(u), so it's not in the intersection.
    # So removing 1 edge (u, b) reduces book size by 1 (if v is still connected to b).
    
    # To get book size 7, we need to remove at least 13 edges incident to u or v (in B).
    
    # But we only have 1 extra edge (401 - 400 = 1), so we can't remove many edges.
    
    # Alternative: Don't start with complete bipartite.
    # Start with a sparser bipartite graph and add more intra edges.
    
    # Let's try: 
    # - Bipartite graph with 380 edges (random)
    # - 21 intra-partition edges (10 in A, 11 in B)
    # Total: 401
    
    # This might give lower max book size.
    
    # Actually, let me just use a random initialization but with a bias toward bipartite structure.
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Separate into bipartite and intra
    bipartite = [(i, j) for i in A for j in B]
    intra_a = [(i, j) for i in range(20) for j in range(i+1, 20)]
    intra_b = [(i, j) for i in range(20, 40) for j in range(i+1, 40)]
    
    # Start with most bipartite edges
    random.shuffle(bipartite)
    random.shuffle(intra_a)
    random.shuffle(intra_b)
    
    # Use 395 bipartite edges
    selected = bipartite[:395]
    edge_set = set()
    for u, v in selected:
        key = (min(u,v), max(u,v))
        edge_set.add(key)
    
    # Add 6 intra edges (3 from A, 3 from B)
    selected += intra_a[:3]
    selected += intra_b[:3]
    
    # Ensure we have exactly 401 distinct edges
    final_edges = []
    seen = set()
    for u, v in selected:
        key = (min(u,v), max(u,v))
        if key not in seen:
            seen.add(key)
            final_edges.append([u, v])
    
    # If we don't have enough, add more
    while len(final_edges) < target_edges:
        # Pick random edge not in seen
        while True:
            u, v = random.choice(all_possible)
            key = (min(u,v), max(u,v))
            if key not in seen:
                seen.add(key)
                final_edges.append([u, v])
                break
    
    return final_edges

def construct():
    start_time = time.time()
    n = 40
    target_edges = 401
    time_limit = 115.0
    
    # Use structural initialization
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
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Track stagnation
    no_improve_count = 0
    stagnation_limit = 5000
    
    iterations = 0
    while time.time() - start_time < time_limit:
        iterations += 1
        
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
        # Find index
        idx = -1
        for i, e in enumerate(current_edges):
            k = (min(e[0], e[1]), max(e[0], e[1]))
            if k == rem_key:
                idx = i
                break
        if idx == -1:
            continue
        
        # Pick a non-edge to add - bias toward edges that won't create large books
        candidates = []
        for _ in range(20):  # Sample 20 candidates
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
