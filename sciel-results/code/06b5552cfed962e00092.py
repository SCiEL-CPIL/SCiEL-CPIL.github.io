import random
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as sparse_laplacian

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 40
    m = 401
    
    def build_bitmasks(edge_set):
        adj_masks = [0] * n
        for (u, v) in edge_set:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return adj_masks

    def calc_book_bitmask(u, v, adj_masks):
        common = adj_masks[u] & adj_masks[v]
        return bin(common).count('1')

    def calc_all_books(edge_set, adj_masks):
        books = {}
        for (u, v) in edge_set:
            books[(u, v)] = calc_book_bitmask(u, v, adj_masks)
        return books

    def get_max_book(books):
        return max(books.values()) if books else 0

    def find_worst_edge(books):
        worst_edge = None
        worst_val = -1
        for e, val in books.items():
            if val > worst_val:
                worst_val = val
                worst_edge = e
        return worst_edge, worst_val

    def spectral_score(adj_masks):
        """Compute a spectral proxy for book size concentration."""
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.int32)
        for u in range(n):
            mask = adj_masks[u]
            while mask:
                v = (mask & -mask).bit_length() - 1
                A[u, v] = 1
                mask &= mask - 1
        
        # Compute A^2 (number of 2-walks)
        A2 = A @ A
        # The diagonal of A^2 is degrees, off-diagonal is book size
        # We want to minimize the maximum off-diagonal entry
        max_off_diag = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    if A2[i, j] > max_off_diag:
                        max_off_diag = A2[i, j]
        return max_off_diag

    def find_best_swap(edge_set, adj_masks, books, current_score, max_candidates=200):
        """Find the best edge swap to reduce max book size."""
        worst_edge, worst_val = find_worst_edge(books)
        if worst_val < 6:
            return None, None, None, None, None
        
        u, v = worst_edge
        
        # Collect all non-edges as candidates
        candidates = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edge_set and (i, j) != (u, v):
                    # Estimate triangle count for this potential edge
                    tri = calc_book_bitmask(i, j, adj_masks)
                    candidates.append((tri, i, j))
        
        # Sort by triangle count
        candidates.sort(key=lambda x: x[0])
        
        best_swap = None
        best_new_score = current_score + 1
        
        for tri, ru, rv in candidates[:max_candidates]:
            # Try the swap
            edge_set.remove((u, v))
            edge_set.add((ru, rv))
            
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            adj_masks[ru] |= (1 << rv)
            adj_masks[rv] |= (1 << ru)
            
            # Recalculate books for affected edges
            affected = {u, v, ru, rv}
            new_books = {}
            for e, val in books.items():
                e_u, e_v = e
                if e_u in affected or e_v in affected:
                    new_books[e] = calc_book_bitmask(e_u, e_v, adj_masks)
                else:
                    new_books[e] = val
            new_books[(ru, rv)] = calc_book_bitmask(ru, rv, adj_masks)
            
            new_max = get_max_book(new_books)
            
            if new_max < best_new_score:
                best_new_score = new_max
                best_swap = (u, v, ru, rv, new_max)
            
            # Revert
            edge_set.remove((ru, rv))
            edge_set.add((u, v))
            adj_masks[ru] &= ~(1 << rv)
            adj_masks[rv] &= ~(1 << ru)
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        
        if best_swap is None:
            return None, None, None, None, None
        
        return best_swap[0], best_swap[1], best_swap[2], best_swap[3], best_swap[4]

    def initialize_balanced():
        """Initialize with a balanced degree sequence."""
        edges = []
        edge_set = set()
        deg = [0] * n
        
        # Start with a cycle
        for i in range(n):
            j = (i + 1) % n
            if i < j:
                edge_set.add((i, j))
                edges.append([i, j])
                deg[i] += 1
                deg[j] += 1
        
        # Add remaining edges with degree balancing
        remaining = m - len(edges)
        attempts = 0
        while len(edges) < m and attempts < 50000:
            attempts += 1
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                continue
            # Prefer lower degree vertices
            if deg[u] > deg[v]:
                u, v = v, u
            edge_set.add((u, v))
            edges.append([u, v])
            deg[u] += 1
            deg[v] += 1
        
        return edges, edge_set, deg

    best_score = 999
    best_edges = None
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        edges, edge_set, deg = initialize_balanced()
        adj_masks = build_bitmasks(edge_set)
        books = calc_all_books(edge_set, adj_masks)
        current_score = get_max_book(books)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = [[u, v] for (u, v) in edge_set]
        
        if current_score <= 6:
            break
        
        # Local search
        stuck = 0
        for step in range(3000):
            if time.time() > deadline - 5:
                return best_edges
            
            worst_edge, worst_val = find_worst_edge(books)
            if worst_val <= 6:
                break
            
            # Try to find a beneficial swap
            su, sv, ru, rv, new_score = find_best_swap(edge_set, adj_masks, books, current_score, max_candidates=100)
            
            if su is not None and new_score < current_score:
                # Apply swap
                edge_set.remove((su, sv))
                edge_set.add((ru, rv))
                
                adj_masks[su] &= ~(1 << sv)
                adj_masks[sv] &= ~(1 << su)
                adj_masks[ru] |= (1 << rv)
                adj_masks[rv] |= (1 << ru)
                
                affected = {su, sv, ru, rv}
                new_books = {}
                for e, val in books.items():
                    e_u, e_v = e
                    if e_u in affected or e_v in affected:
                        new_books[e] = calc_book_bitmask(e_u, e_v, adj_masks)
                    else:
                        new_books[e] = val
                new_books[(ru, rv)] = calc_book_bitmask(ru, rv, adj_masks)
                
                current_score = new_score
                books = new_books
                
                # Update edges list
                if [su, sv] in edges:
                    edges.remove([su, sv])
                elif [sv, su] in edges:
                    edges.remove([sv, su])
                edges.append([ru, rv])
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges = [[a, b] for (a, b) in edge_set]
                
                stuck = 0
            else:
                stuck += 1
                if stuck > 100:
                    break
    
    return best_edges
