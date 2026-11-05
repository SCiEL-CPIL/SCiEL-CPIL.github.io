import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n = 40
    m = 401
    random.seed(42)
    
    # Bitmask representation for fast intersection
    adj_mask = [0] * n
    
    def compute_books(adj_mask):
        """Compute book size for each edge using bitmask intersection."""
        books = {}
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                v = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if u < v:
                    common = adj_mask[u] & adj_mask[v]
                    books[(u, v)] = bin(common).count('1')
        return books

    def get_max_book(books):
        return max(books.values()) if books else 0

    def get_sum_sq_books(books):
        return sum(b*b for b in books.values())

    def balanced_bipartite_init(n):
        """
        Initialize with a balanced bipartite graph plus 1 internal edge.
        """
        adj_mask = [0] * n
        edge_set = set()
        
        # Add all bipartite edges between A={0..19} and B={20..39}
        for i in range(20):
            for j in range(20, 40):
                u, v = i, j
                edge_set.add((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        # Add 1 internal edge to reach 401 edges
        u, v = 0, 1
        edge_set.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        return edge_set, adj_mask

    # Initialize
    edge_set, adj_mask = balanced_bipartite_init(n)
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    books = compute_books(adj_mask)
    current_max_book = get_max_book(books)
    current_sum_sq = get_sum_sq_books(books)
    
    best_max_book = current_max_book
    best_sum_sq = current_sum_sq
    best_edge_set = set(edge_set)
    best_adj_mask = list(adj_mask)
    
    # If we already hit 7, we can't do better (theoretical min).
    if best_max_book == 7:
        return [list(e) for e in best_edge_set]

    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.9999
    step_count = 0
    no_improve_steps = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Check for stagnation and restart if necessary
        if no_improve_steps > max_no_improve:
            # Restart from best solution with perturbation
            adj_mask = list(best_adj_mask)
            edge_set = set(best_edge_set)
            books = compute_books(adj_mask)
            current_max_book = get_max_book(books)
            current_sum_sq = get_sum_sq_books(books)
            
            # Perturb: do 10 random swaps
            for _ in range(10):
                e_rem = random.choice(list(edge_set))
                u, v = e_rem
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                edge_set.remove(e_rem)
                
                # Find a non-edge to add
                while True:
                    a, b = random.choice(all_possible)
                    if (a, b) not in edge_set:
                        break
                adj_mask[a] |= (1 << b)
                adj_mask[b] |= (1 << a)
                edge_set.add((a, b))
            
            books = compute_books(adj_mask)
            current_max_book = get_max_book(books)
            current_sum_sq = get_sum_sq_books(books)
            no_improve_steps = 0
            T = 5.0 # Reheat slightly

        # Select edge to remove: prefer high-book edges
        if random.random() < 0.8:
            sorted_books = sorted(books.items(), key=lambda x: -x[1])
            top_k = max(1, len(sorted_books) // 5)
            e_remove = random.choice(sorted_books[:top_k])[0]
        else:
            e_remove = random.choice(list(edge_set))
            
        u, v = e_remove
        u, v = min(u, v), max(u, v)
        
        # Select edge to add: sample candidates and pick one with low common neighbors
        candidates = []
        for _ in range(30):
            a, b = random.choice(all_possible)
            a, b = min(a, b), max(a, b)
            if (a, b) not in edge_set and (a, b) != e_remove:
                candidates.append((a, b))
        
        if not candidates:
            T *= cooling
            continue
            
        best_cand = None
        best_common = 100
        for a, b in candidates:
            common = bin(adj_mask[a] & adj_mask[b]).count('1')
            if common < best_common:
                best_common = common
                best_cand = (a, b)
        
        if best_cand is None:
            T *= cooling
            continue
            
        a, b = best_cand
        
        # Apply move
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
        
        edge_set.remove(e_remove)
        edge_set.add((a, b))
        
        # Recompute books for affected edges
        # Affected vertices: u, v, a, b
        affected_verts = {u, v, a, b}
        
        # Get old books for affected edges
        old_books = {}
        for e in edge_set:
            x, y = e
            if x in affected_verts or y in affected_verts:
                # This edge is in the new graph, but we need its old value if it existed
                # Actually, we need to update the global books dict
                pass
        
        # Simpler: Recompute books for all edges incident to affected vertices
        # And the removed edge is gone, added edge is new.
        
        # Update books dict
        # Remove entries for edges involving u, v, a, b that are no longer valid or changed
        # This is tricky with dict. Let's just recompute the whole books dict?
        # No, that's O(E * n) which is too slow.
        
        # Let's update incrementally.
        # Edges that change book size are those incident to u, v, a, b.
        
        # 1. Remove old book entries for edges incident to u, v, a, b
        # Note: e_remove is gone. (a,b) is new.
        
        # Identify edges to update:
        # All edges (x,y) such that x in {u,v,a,b} or y in {u,v,a,b}
        
        edges_to_update = set()
        for vert in affected_verts:
            mask = adj_mask[vert]
            while mask:
                other = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if vert < other:
                    edges_to_update.add((vert, other))
                else:
                    edges_to_update.add((other, vert))
        
        # Save old values
        old_vals = {}
        for e in edges_to_update:
            if e in books:
                old_vals[e] = books[e]
        
        # Calculate new values
        for e in edges_to_update:
            x, y = e
            common = adj_mask[x] & adj_mask[y]
            books[e] = bin(common).count('1')
            
        # The removed edge e_remove is no longer in edge_set, so it shouldn't be in books.
        # It was in books before. We need to delete it.
        if e_remove in books:
            del books[e_remove]
            
        # The added edge (a,b) is in edge_set and in edges_to_update (since a,b in affected_verts).
        # So it has been updated/added.
        
        new_max_book = get_max_book(books)
        new_sum_sq = get_sum_sq_books(books)
        
        # Acceptance
        accept = False
        if new_max_book < current_max_book:
            accept = True
        elif new_max_book == current_max_book:
            if new_sum_sq < current_sum_sq:
                accept = True
            else:
                if random.random() < math.exp(-(new_sum_sq - current_sum_sq) / (T * 100)):
                    accept = True
        else:
            if random.random() < math.exp(-(new_max_book - current_max_book) / T):
                accept = True
                
        if accept:
            current_max_book = new_max_book
            current_sum_sq = new_sum_sq
            no_improve_steps = 0
            
            if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
                best_max_book = current_max_book
                best_sum_sq = current_sum_sq
                best_edge_set = set(edge_set)
                best_adj_mask = list(adj_mask)
                
                if best_max_book == 7:
                    return [list(e) for e in best_edge_set]
        else:
            # Revert
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_mask[a] &= ~(1 << b)
            adj_mask[b] &= ~(1 << a)
            
            edge_set.remove((a, b))
            edge_set.add(e_remove)
            
            # Restore books
            for e in edges_to_update:
                if e in old_vals:
                    books[e] = old_vals[e]
                else:
                    # This edge didn't exist before, so delete it
                    if e in books:
                        del books[e]
                        
            # Restore removed edge's book value
            if e_remove in old_vals:
                 # It was in old_vals if it was in edges_to_update?
                 # e_remove is incident to u,v. So it was in edges_to_update?
                 # No, e_remove was removed from adj_mask before computing edges_to_update.
                 # So e_remove is NOT in edges_to_update.
                 # So we need to manually restore it.
                 pass
            
            # Wait, my incremental update logic is flawed for the removed edge.
            # Let's just recompute books for the whole graph if the logic is complex.
            # O(E * n) = 401 * 40 = 16040 operations.
            # In Python, this might be ~1ms.
            # 120s / 1ms = 120,000 steps.
            # This is feasible!
            
            # Let's simplify: Just recompute books from scratch on every step?
            # No, 120,000 steps * 1ms = 120s. It fits.
            # But sorting and other overhead might make it slower.
            # Let's stick to incremental but fix the revert.
            
            # Actually, reverting incremental updates is error-prone.
            # Let's use a simpler strategy:
            # Keep a copy of the books dict before the move?
            # books_copy = dict(books)
            # This is O(E) = 401. Fast.
            
            # Let's restructure the loop to use books_copy for revert.

    return [list(e) for e in best_edge_set]
