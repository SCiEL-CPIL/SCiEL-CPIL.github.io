import time
import random
import numpy as np
from itertools import combinations

# Deadline guard
DEADLINE = 140.0  # seconds, safety margin from 150s limit

def _compute_books(edge_set, n=40):
    """Compute the size of the largest book (max triangles containing an edge).
    edge_set: set of (u, v) with u < v
    Returns: (max_book, book_dict) where book_dict maps edge -> count of triangles
    """
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    # For each edge, count common neighbors
    book_dict = {}
    max_book = 0
    for u, v in edge_set:
        common = len(adj[u] & adj[v])
        book_dict[(u, v)] = common
        if common > max_book:
            max_book = common
    return max_book, book_dict

def _random_graph(n=40, m=401):
    """Generate a random graph with exactly m edges."""
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    return set(all_edges[:m])

def _improve(graph, max_iters=500000, time_limit=None):
    """Local search via edge swaps to minimize the largest book."""
    n = 40
    m = 401
    current = set(graph)
    max_book, book_dict = _compute_books(current)
    best_book = max_book
    best_graph = set(current)
    
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    edge_set = set(all_edges)
    
    # Build adjacency for quick lookup
    adj = [set() for _ in range(n)]
    for u, v in current:
        adj[u].add(v)
        adj[v].add(u)
    
    iters = 0
    start_time = time.time()
    
    while iters < max_iters:
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        if time.time() > DEADLINE:
            break
        
        iters += 1
        
        # Pick a random edge to remove
        edge_to_remove = random.choice(list(current))
        u_rm, v_rm = edge_to_remove
        
        # Pick a random non-edge to add
        # Sample random pairs until we find a non-edge
        attempts = 0
        while attempts < 20:
            u_add, v_add = random.sample(range(n), 2)
            if u_add > v_add:
                u_add, v_add = v_add, u_add
            if (u_add, v_add) not in current:
                break
            attempts += 1
        else:
            continue
        
        # Compute change in book sizes
        # Removing edge (u_rm, v_rm): 
        #   - The edge itself is removed, so its book count no longer matters
        #   - For each common neighbor w of u_rm and v_rm, the edge (u_rm,w) loses a triangle,
        #     and edge (v_rm,w) loses a triangle
        # Adding edge (u_add, v_add):
        #   - New edge gets book count = |N(u_add) ∩ N(v_add)|
        #   - For each common neighbor w, edges (u_add,w) and (v_add,w) gain a triangle
        
        # Compute new book for the added edge
        common_add = adj[u_add] & adj[v_add]
        new_book_add = len(common_add)
        
        # Check if removing u_rm-v_rm reduces the max
        # The edges affected by removal: (u_rm, v_rm) itself (removed), and all (u_rm, w) and (v_rm, w) for w in common_rm
        common_rm = adj[u_rm] & adj[v_rm]
        
        # Simulate: temporarily remove edge, compute affected books
        # Edges whose book count changes:
        #   (u_rm, v_rm): removed
        #   (u_rm, w) for w in common_rm: book decreases by 1
        #   (v_rm, w) for w in common_rm: book decreases by 1
        #   (u_add, v_add): added with book = new_book_add
        #   (u_add, w) for w in common_add: book increases by 1
        #   (v_add, w) for w in common_add: book increases by 1
        
        # We need to check if the new max book is <= current max_book
        # If new_book_add > max_book, this swap is bad (unless we also reduce max_book edges)
        
        # Quick rejection: if the new edge's book is already > max_book, skip
        if new_book_add > max_book:
            continue
        
        # Compute the new max book after the swap
        # Start with current max_book
        # Edges that had book == max_book and are affected by removal might drop
        # Edges that had book == max_book and are affected by addition might rise (but we already checked new_book_add <= max_book)
        
        # Let's compute the actual new max by checking affected edges
        # Affected edges:
        affected = set()
        # Edges losing a triangle
        for w in common_rm:
            e1 = (min(u_rm, w), max(u_rm, w))
            e2 = (min(v_rm, w), max(v_rm, w))
            affected.add(e1)
            affected.add(e2)
        # The removed edge
        affected.add(edge_to_remove)
        # Edges gaining a triangle
        for w in common_add:
            e1 = (min(u_add, w), max(u_add, w))
            e2 = (min(v_add, w), max(v_add, w))
            affected.add(e1)
            affected.add(e2)
        # The added edge
        affected.add((u_add, v_add))
        
        # Compute new books for affected edges
        new_max = 0
        # First, find the max among unaffected edges that had book == max_book
        # If no unaffected edge has book == max_book, the new max is max of affected edges' new books
        # Otherwise, new max is at least max_book (from unaffected)
        
        # Check if any edge with book == max_book is NOT in affected and not removed
        has_unaffected_max = False
        for e, b in book_dict.items():
            if b == max_book and e != edge_to_remove and e not in affected:
                has_unaffected_max = True
                break
        
        if has_unaffected_max:
            new_max = max_book
        else:
            # New max is determined by affected edges' new book values
            new_max = 0
        
        # Compute new book values for affected edges
        for e in affected:
            if e == edge_to_remove:
                continue  # removed
            if e == (u_add, v_add):
                new_b = new_book_add
            else:
                # Recompute book for this edge
                eu, ev = e
                new_b = len(adj[eu] & adj[ev])
                # Adjust for the swap:
                # If e involves u_rm or v_rm, we need to account for the removed edge
                # If e involves u_add or v_add, we need to account for the added edge
                
                # Actually, let's just recompute properly
                # The adjacency changes: remove (u_rm, v_rm), add (u_add, v_add)
                # For edge e = (a, b), its book = |N(a) ∩ N(b)| in the new graph
                
                # New N(a) = old N(a) - {v_rm if a==u_rm or a==v_rm} + {u_add if a==u_add or a==v_add}
                # This is getting complex. Let's just do a full recompute for affected edges.
                
                # Simpler: just compute from scratch for these few edges
                # Build temporary adjacency
                # Actually, for efficiency, let's compute:
                # New common neighbors of a and b = 
                #   (old common) - {neighbors removed from a or b} + {neighbors added to a or b}
                
                # Neighbors removed from a: v_rm if a in {u_rm, v_rm}
                # Neighbors added to a: u_add if a in {u_add, v_add}
                
                old_common = adj[a] & adj[b] if False else None  # can't easily get this
                
                # Let's just brute force for affected edges (there are at most ~2*|common_rm| + 2*|common_add| + 2 edges)
                # For each affected edge (a,b), compute new common neighbors:
                # w is a common neighbor in new graph iff:
                #   w adjacent to a in new graph AND w adjacent to b in new graph
                #   w != a, w != b
                #   w is not the edge being removed/added in a way that breaks adjacency
                
                # w adjacent to a in new graph:
                #   (w in adj[a]) XOR (w == v_rm and a in {u_rm, v_rm}) XOR (w == u_add and a in {u_add, v_add})
                #   More precisely:
                #   new_adj_a = adj[a] - ({v_rm} if a in {u_rm,v_rm} else {}) + ({u_add} if a in {u_add,v_add} else {})
                #   But we need to be careful: if a == u_rm, then v_rm was in adj[a], now removed
                #   if a == u_add, then u_add... wait, u_add is the other endpoint
            
                # This is getting too complex for inline. Let's just do a full recompute of books
                # for the whole graph after the swap, but only if the swap looks promising.
            
            if new_b > new_max:
                new_max = new_b
        
        # Hmm, this approach is getting messy. Let me simplify:
        # Just do the swap, recompute books, and revert if worse.
        
        # Perform swap
        current.remove(edge_to_remove)
        adj[u_rm].remove(v_rm)
        adj[v_rm].remove(u_rm)
        current.add((u_add, v_add))
        adj[u_add].add(v_add)
        adj[v_add].add(u_add)
        
        # Recompute books
        new_max_book, new_book_dict = _compute_books(current)
        
        if new_max_book < max_book:
            # Accept
            max_book = new_max_book
            book_dict = new_book_dict
            if max_book < best_book:
                best_book = max_book
                best_graph = set(current)
        else:
            # Revert
            current.remove((u_add, v_add))
            adj[u_add].remove(v_add)
            adj[v_add].remove(u_add)
            current.add(edge_to_remove)
            adj[u_rm].add(v_rm)
            adj[v_rm].add(u_rm)
    
    return best_graph, best_book

def construct():
    start_time = time.time()
    
    # Start with the known good solution: K20,20 + one edge
    # This gives a specific book size
    base_edges = [(a, b) for a in range(20) for b in range(20, 40)]
    base_edges.append((0, 1))
    initial_graph = set(base_edges)
    
    initial_max, _ = _compute_books(initial_graph)
    best_graph = set(initial_graph)
    best_book = initial_max
    
    # Try multiple random restarts with local search
    rng = random.Random(42)
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > DEADLINE:
            break
        
        # Create a perturbed version of the best graph
        if r == 0:
            current_graph = set(best_graph)
        else:
            # Perturb: swap k random edges
            current_graph = set(best_graph)
            adj = [set() for _ in range(40)]
            for u, v in current_graph:
                adj[u].add(v)
                adj[v].add(u)
            
            k = rng.randint(5, 20)
            for _ in range(k):
                if time.time() > DEADLINE:
                    break
                # Remove a random edge
                e_rm = rng.choice(list(current_graph))
                current_graph.remove(e_rm)
                u, v = e_rm
                adj[u].discard(v)
                adj[v].discard(u)
                
                # Add a random non-edge
                attempts = 0
                while attempts < 50:
                    u2, v2 = rng.sample(range(40), 2)
                    if u2 > v2:
                        u2, v2 = v2, u2
                    if (u2, v2) not in current_graph:
                        break
                    attempts += 1
                else:
                    # Revert
                    current_graph.add(e_rm)
                    adj[u].add(v)
                    adj[v].add(u)
                    continue
                current_graph.add((u2, v2))
                adj[u2].add(v2)
                adj[v2].add(u2)
        
        # Run local search with time budget
        time_remaining = DEADLINE - time.time() - 5  # 5s safety margin
        if time_remaining <= 0:
            break
        
        # Limit iterations based on time
        max_iters = min(200000, int(time_remaining * 2000))
        
        improved_graph, improved_book = _improve(
            current_graph, 
            max_iters=max_iters,
            time_limit=time_remaining
        )
        
        if improved_book < best_book:
            best_book = improved_book
            best_graph = improved_graph
        
        if best_book <= 6:
            break  # Can't do better than 6 (lower bound is 40/6 ≈ 6.67, so 7 is the integer bound)
    
    # Convert to list of [u, v]
    result = [[u, v] for u, v in best_graph]
    return result

# Test
if __name__ == "__main__":
    edges = construct()
    assert len(edges) == 401
    assert len(set(tuple(sorted(e)) for e in edges)) == 401
    for e in edges:
        assert 0 <= e[0] < 40 and 0 <= e[1] < 40
        assert e[0] != e[1]
    print(f"Graph constructed with {len(edges)} edges")
    max_book, _ = _compute_books(set(tuple(sorted(e)) for e in edges))
    print(f"Max book size: {max_book}")
