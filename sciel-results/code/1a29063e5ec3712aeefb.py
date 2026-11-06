import numpy as np
import time
import random

def construct():
    n = 40
    m = 401
    deadline = time.time() + 115.0
    
    # State
    A = np.zeros((n, n), dtype=np.int8)
    book = np.zeros((n, n), dtype=np.int32)
    deg = np.zeros(n, dtype=np.int32)
    edges = set()
    
    def update_add(u, v):
        # Add edge (u, v)
        if A[u, v]: return
        A[u, v] = 1
        A[v, u] = 1
        deg[u] += 1
        deg[v] += 1
        edges.add((u, v))
        
        # Compute common neighbors
        # common = np.where(A[u] & A[v] == 1)[0]
        # book[u, v] = len(common)
        # book[v, u] = len(common)
        
        # Faster:
        common_mask = A[u] & A[v]
        common_count = np.sum(common_mask)
        book[u, v] = common_count
        book[v, u] = common_count
        
        # Update book sizes for neighbors
        # For each w in common, book[u, w] += 1 and book[v, w] += 1
        common_indices = np.where(common_mask)[0]
        for w in common_indices:
            book[u, w] += 1
            book[w, u] += 1
            book[v, w] += 1
            book[w, v] += 1

    def update_remove(u, v):
        # Remove edge (u, v)
        if not A[u, v]: return
        A[u, v] = 0
        A[v, u] = 0
        deg[u] -= 1
        deg[v] -= 1
        edges.discard((u, v))
        
        common_mask = A[u] & A[v]
        common_indices = np.where(common_mask)[0]
        
        # Revert book sizes
        for w in common_indices:
            book[u, w] -= 1
            book[w, u] -= 1
            book[v, w] -= 1
            book[w, v] -= 1
            
        book[u, v] = 0
        book[v, u] = 0

    def get_max_book():
        # Max over i<j where A[i,j]==1
        # We can just take max of book, since book[i,j]=0 if not edge?
        # Wait, book[i,j] is only non-zero if edge exists?
        # In update_add, we set book[u,v].
        # In update_remove, we set to 0.
        # So max(book) is the max book size.
        return int(np.max(book))

    def get_sum_sq():
        return int(np.sum(book * book))

    # Initialization
    # Try a few random starts
    best_max_b = float('inf')
    best_edges = None
    best_sum_sq = float('inf')
    
    all_possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        A.fill(0)
        book.fill(0)
        deg.fill(0)
        edges.clear()
        
        # Random initialization
        random.shuffle(all_possible_edges)
        for u, v in all_possible_edges[:m]:
            update_add(u, v)
            
        current_max_b = get_max_book()
        current_sum_sq = get_sum_sq()
        
        if current_max_b < best_max_b:
            best_max_b = current_max_b
            best_sum_sq = current_sum_sq
            best_edges = set(edges)
            
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        
        while time.time() < deadline - 2:
            # Pick edge to remove
            # Bias towards edges with high book size
            # Get list of current edges
            edge_list = list(edges)
            if not edge_list:
                break
            
            # Weighted selection
            # We can just pick random for simplicity, or high book
            # Let's pick random first, if stuck, try high book
            rem_u, rem_v = random.choice(edge_list)
            
            # Pick non-edge to add
            # Sample random non-edge
            add_u, add_v = 0, 0
            added = False
            for _ in range(10):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                if (u, v) not in edges:
                    add_u, add_v = u, v
                    added = True
                    break
            
            if not added:
                continue
                
            # Calculate potential impact
            # Remove rem
            update_remove(rem_u, rem_v)
            max_b_after_rem = get_max_book()
            
            # Add add
            update_add(add_u, add_v)
            max_b_after_add = get_max_book()
            sum_sq_after = get_sum_sq()
            
            delta_max = max_b_after_add - current_max_b
            delta_sq = sum_sq_after - current_sum_sq
            
            # Acceptance
            accept = False
            if delta_max < 0:
                accept = True
            elif delta_max == 0:
                if delta_sq < 0:
                    accept = True
                else:
                    # Accept with prob exp(-delta_sq / T)
                    if T > T_min and random.random() < np.exp(-delta_sq / T):
                        accept = True
            else:
                # delta_max > 0
                if T > T_min and random.random() < np.exp(-(delta_max * 100 + delta_sq) / T):
                    accept = True
            
            if accept:
                current_max_b = max_b_after_add
                current_sum_sq = sum_sq_after
                T = max(T_min, T * cooling)
                if current_max_b < best_max_b or (current_max_b == best_max_b and current_sum_sq < best_sum_sq):
                    best_max_b = current_max_b
                    best_sum_sq = current_sum_sq
                    best_edges = set(edges)
            else:
                # Revert
                update_remove(add_u, add_v)
                update_add(rem_u, rem_v)
                T = max(T_min, T * cooling)

    return [[u, v] for u, v in best_edges]
