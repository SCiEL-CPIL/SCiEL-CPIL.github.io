"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill
    
    N = 40
    E = 401
    
    def compute_books(A):
        """Compute book sizes using matrix multiplication.
        Book size of edge (u,v) is A2[u,v] where A2 = A @ A (off-diagonal)."""
        A2 = A @ A
        np.fill_diagonal(A2, 0)
        return A2
    
    def get_max_book(A):
        """Get maximum book size from adjacency matrix."""
        A2 = compute_books(A)
        return int(np.max(A2))
    
    def get_books_dict(A):
        """Get dictionary of book sizes for all edges."""
        A2 = compute_books(A)
        books = {}
        for u in range(N):
            for v in range(u+1, N):
                if A[u, v] == 1:
                    books[(u, v)] = int(A2[u, v])
        return books
    
    def random_graph():
        """Generate a random graph with exactly E edges."""
        all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(all_edges)
        return all_edges[:E]
    
    def build_adjacency(edges):
        """Build adjacency matrix from edge list."""
        A = np.zeros((N, N), dtype=np.int32)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        return A
    
    def local_search_sa(edges, A, max_book_val, time_limit=30.0):
        """Simulated annealing local search with edge swaps."""
        T = 5.0
        T_min = 0.01
        no_improve_count = 0
        max_no_improve = 2000
        current_max = max_book_val
        
        # Precompute non-edges for faster sampling
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))
        
        non_edges = []
        for i in range(N):
            for j in range(i+1, N):
                if (i, j) not in edge_set:
                    non_edges.append((i, j))
        
        edge_list = list(edge_set)
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            if no_improve_count > max_no_improve:
                break
            
            # Pick random edge to remove
            rem_idx = random.randint(0, len(edge_list) - 1)
            rem_u, rem_v = edge_list[rem_idx]
            
            # Pick random non-edge to add
            if not non_edges:
                break
            add_idx = random.randint(0, len(non_edges) - 1)
            add_u, add_v = non_edges[add_idx]
            
            # Perform swap
            A[rem_u, rem_v] = 0
            A[rem_v, rem_u] = 0
            A[add_u, add_v] = 1
            A[add_v, add_u] = 1
            
            new_A2 = compute_books(A)
            new_max = int(np.max(new_A2))
            new_sum_sq = float(np.sum(new_A2 ** 2))
            
            old_A2 = A @ A
            np.fill_diagonal(old_A2, 0)
            old_max = current_max
            old_sum_sq = float(np.sum(old_A2 ** 2))
            
            accept = False
            if new_max < old_max:
                accept = True
                current_max = new_max
                no_improve_count = 0
            elif new_max == old_max and new_sum_sq < old_sum_sq:
                accept = True
                no_improve_count = 0
            else:
                # Calculate penalty
                if new_max > old_max:
                    delta = 100.0 + (new_sum_sq - old_sum_sq)
                else:
                    delta = new_sum_sq - old_sum_sq
                
                if delta < 0:
                    accept = True
                    no_improve_count = 0
                else:
                    if random.random() < np.exp(-delta / T):
                        accept = True
                    else:
                        no_improve_count += 1
            
            if accept:
                # Update edge lists
                edge_list[rem_idx] = (add_u, add_v)
                edge_set.discard((rem_u, rem_v))
                edge_set.add((add_u, add_v))
                non_edges[add_idx] = (rem_u, rem_v)
            else:
                # Revert
                A[rem_u, rem_v] = 1
                A[rem_v, rem_u] = 1
                A[add_u, add_v] = 0
                A[add_v, add_u] = 0
                no_improve_count += 1
            
            # Cool down
            T = max(T_min, T * 0.999)
        
        # Reconstruct edges from A
        final_edges = []
        for i in range(N):
            for j in range(i+1, N):
                if A[i, j] == 1:
                    final_edges.append((i, j))
        
        return final_edges, current_max
    
    def triangle_breaking_search(edges, A, max_book_val, time_limit=20.0):
        """Targeted search: prioritize breaking edges in high-book triangles."""
        current_max = max_book_val
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))
        
        non_edges = []
        for i in range(N):
            for j in range(i+1, N):
                if (i, j) not in edge_set:
                    non_edges.append((i, j))
        
        end_time = time.time() + time_limit
        no_improve = 0
        max_no_improve = 1500
        
        while time.time() < end_time:
            if no_improve > max_no_improve:
                break
            
            # Find the edge with maximum book
            A2 = compute_books(A)
            max_val = int(np.max(A2))
            if max_val < current_max:
                current_max = max_val
            
            # Find all edges with book == max_val
            high_book_edges = []
            for i in range(N):
                for j in range(i+1, N):
                    if A[i, j] == 1 and A2[i, j] == max_val:
                        high_book_edges.append((i, j))
            
            if not high_book_edges:
                break
            
            # Pick a random high-book edge
            rem_u, rem_v = random.choice(high_book_edges)
            
            # Find non-edges that would minimize new book size
            best_add = None
            best_new_book = float('inf')
            candidates = random.sample(non_edges, min(50, len(non_edges)))
            
            for add_u, add_v in candidates:
                # Estimate book size if we add this edge
                # Book size = number of common neighbors of add_u and add_v
                common = int(np.sum(A[add_u, :] * A[add_v, :]))
                if common < best_new_book:
                    best_new_book = common
                    best_add = (add_u, add_v)
            
            if best_add is None or best_new_book >= max_val:
                # Try random non-edge instead
                add_u, add_v = random.choice(non_edges)
                common = int(np.sum(A[add_u, :] * A[add_v, :]))
                if common >= max_val:
                    no_improve += 1
                    continue
            else:
                add_u, add_v = best_add
            
            # Perform swap
            A[rem_u, rem_v] = 0
            A[rem_v, rem_u] = 0
            A[add_u, add_v] = 1
            A[add_v, add_u] = 1
            
            new_A2 = compute_books(A)
            new_max = int(np.max(new_A2))
            
            if new_max < current_max:
                # Accept
                current_max = new_max
                no_improve = 0
                edge_set.discard((rem_u, rem_v))
                edge_set.add((add_u, add_v))
                non_edges.remove((add_u, add_v))
                non_edges.append((rem_u, rem_v))
            else:
                # Revert
                A[rem_u, rem_v] = 1
                A[rem_v, rem_u] = 1
                A[add_u, add_v] = 0
                A[add_v, add_u] = 0
                no_improve += 1
        
        # Reconstruct edges from A
        final_edges = []
        for i in range(N):
            for j in range(i+1, N):
                if A[i, j] == 1:
                    final_edges.append((i, j))
        
        return final_edges, current_max
    
    # Main search loop
    best_edges = None
    best_max = float('inf')
    
    num_restarts = 15
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Start with random graph
        edges = random_graph()
        A = build_adjacency(edges)
        max_book_val = get_max_book(A)
        
        # First apply triangle-breaking search (fast, targeted)
        edges, max_book_val = triangle_breaking_search(edges, A, max_book_val, time_limit=15.0)
        
        if max_book_val < best_max:
            best_max = max_book_val
            best_edges = edges.copy()
        
        # Then apply SA for further refinement
        A = build_adjacency(edges)
        edges, max_book_val = local_search_sa(edges, A, max_book_val, time_limit=40.0)
        
        if max_book_val < best_max:
            best_max = max_book_val
            best_edges = edges.copy()
        
        # If we hit theoretical lower bound, stop
        if best_max <= 7:
            break
    
    # Final refinement on best solution
    if best_edges is not None and time.time() < deadline - 5:
        A = build_adjacency(best_edges)
        max_book_val = get_max_book(A)
        best_edges, max_book_val = local_search_sa(best_edges, A, max_book_val, time_limit=20.0)
        if max_book_val < best_max:
            best_max = max_book_val
    
    if best_edges is None:
        best_edges = random_graph()
    
    # Convert to list of [u, v] pairs
    result = [[u, v] for u, v in best_edges]
    return result

# EVOLVE-BLOCK-END
