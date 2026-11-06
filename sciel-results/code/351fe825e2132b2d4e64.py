"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 40
    target_edges = 401
    total_possible_edges = n * (n - 1) // 2
    
    def get_book_sizes(A):
        """Compute book size for every edge."""
        books = {}
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    books[(i, j)] = np.sum(A[i] & A[j])
        return books

    def get_max_book(A):
        """Compute max book size efficiently."""
        max_b = 0
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    b = np.sum(A[i] & A[j])
                    if b > max_b:
                        max_b = b
        return max_b

    # Initialize with a Turan-like structure: complete bipartite + one extra edge
    # K(20,20) has 400 edges, book size = 19 for each edge
    # We need 401 edges, so add one more edge within a part
    A = np.zeros((n, n), dtype=bool)
    
    # Build K(20,20) between {0..19} and {20..39}
    for i in range(20):
        for j in range(20, 40):
            A[i, j] = True
            A[j, i] = True
    
    # Add one extra edge: (0, 1) within left part
    A[0, 1] = True
    A[1, 0] = True
    
    current_max = get_max_book(A)
    best_A = A.copy()
    best_max = current_max
    
    # Maintain edge and non-edge lists
    def build_lists(A):
        edge_list = []
        non_edge_list = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    edge_list.append((i, j))
                else:
                    non_edge_list.append((i, j))
        return edge_list, non_edge_list
    
    edge_list, non_edge_list = build_lists(A)
    
    # Strategy: Focus on breaking edges with high book sizes
    # For each high-book edge, try to remove one of its triangle-forming edges
    # and add a random non-edge elsewhere
    
    T = 2.0
    cooling_rate = 0.9995
    min_T = 0.01
    
    steps = 0
    no_improve_steps = 0
    stagnation_limit = 500
    
    while time.time() < deadline:
        steps += 1
        
        # Get current book sizes to identify critical edges
        books = get_book_sizes(A)
        
        # Find edges with book size >= best_max
        critical_edges = [(e, b) for e, b in books.items() if b >= best_max]
        
        if critical_edges:
            # Pick a critical edge to break
            crit_edge, crit_book = random.choice(critical_edges)
            u, v = crit_edge
            
            # Find common neighbors (triangle partners)
            common = np.where(A[u] & A[v])[0]
            common = [x for x in common if x != u and x != v]
            
            if common:
                # Try removing an edge from one of these triangles
                # Pick a random common neighbor w, and try to remove (u,w) or (v,w)
                w = random.choice(common)
                
                # Decide which edge to remove: (u,w) or (v,w)
                if random.random() < 0.5:
                    remove_edge = (min(u, w), max(u, w))
                else:
                    remove_edge = (min(v, w), max(v, w))
                
                # Pick a random non-edge to add
                if non_edge_list:
                    add_edge = random.choice(non_edge_list)
                else:
                    continue
                
                # Perform the swap
                A[remove_edge[0], remove_edge[1]] = False
                A[remove_edge[1], remove_edge[0]] = False
                A[add_edge[0], add_edge[1]] = True
                A[add_edge[1], add_edge[0]] = True
                
                new_max = get_max_book(A)
                delta = new_max - current_max
                
                if delta < 0 or (T > min_T and random.random() < math.exp(-delta / T)):
                    # Accept move
                    current_max = new_max
                    
                    # Update lists
                    try:
                        idx = edge_list.index(remove_edge)
                        edge_list[idx] = add_edge
                    except ValueError:
                        edge_list.append(add_edge)
                    
                    try:
                        idx = non_edge_list.index(add_edge)
                        non_edge_list[idx] = remove_edge
                    except ValueError:
                        non_edge_list.append(remove_edge)
                    
                    if current_max < best_max:
                        best_max = current_max
                        best_A = A.copy()
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    # Reject move
                    A[remove_edge[0], remove_edge[1]] = True
                    A[remove_edge[1], remove_edge[0]] = False
                    A[add_edge[0], add_edge[1]] = False
                    A[add_edge[1], add_edge[0]] = True
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
        else:
            # No critical edges, do random swap
            if edge_list and non_edge_list:
                u, v = random.choice(edge_list)
                a, b = random.choice(non_edge_list)
                
                A[u, v] = False
                A[v, u] = False
                A[a, b] = True
                A[b, a] = True
                
                new_max = get_max_book(A)
                delta = new_max - current_max
                
                if delta < 0 or (T > min_T and random.random() < math.exp(-delta / T)):
                    current_max = new_max
                    
                    idx = edge_list.index((u, v))
                    edge_list[idx] = (a, b)
                    
                    idx = non_edge_list.index((a, b))
                    non_edge_list[idx] = (u, v)
                    
                    if current_max < best_max:
                        best_max = current_max
                        best_A = A.copy()
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    A[u, v] = True
                    A[v, u] = True
                    A[a, b] = False
                    A[b, a] = False
                    no_improve_steps += 1
        
        T *= cooling_rate
        
        # Reheat if stuck
        if no_improve_steps > stagnation_limit:
            T = 1.5
            no_improve_steps = 0
        
        if time.time() > deadline:
            break

    # Convert best_A to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j]:
                result.append([i, j])
    
    # Verify count
    if len(result) != 401:
        # Fallback: rebuild from scratch
        A = np.zeros((n, n), dtype=bool)
        for i in range(20):
            for j in range(20, 40):
                A[i, j] = True
                A[j, i] = True
        A[0, 1] = True
        A[1, 0] = True
        result = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    result.append([i, j])
        
    return result
# EVOLVE-BLOCK-END
