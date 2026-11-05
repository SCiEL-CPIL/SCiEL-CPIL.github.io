import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n = 40
    target_edges = 401
    
    # 1. Construct T_3(40)
    # Parts: 14, 13, 13
    part_sizes = [14, 13, 13]
    parts = []
    idx = 0
    for size in part_sizes:
        parts.append(list(range(idx, idx + size)))
        idx += size
        
    A = np.zeros((n, n), dtype=bool)
    edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    A[u, v] = True
                    A[v, u] = True
                    edges.append((u, v))
    
    # Initial book sizes
    # book[u][v] is the number of triangles containing edge (u,v)
    # For T_3, book(u,v) = size of the third part
    book = np.zeros((n, n), dtype=int)
    for i in range(3):
        for j in range(i + 1, 3):
            k = 3 - i - j
            size_k = part_sizes[k]
            for u in parts[i]:
                for v in parts[j]:
                    book[u, v] = size_k
                    book[v, u] = size_k
    
    def get_max_book():
        return np.max(book)
    
    def remove_edge(u, v):
        # Update book sizes for edges sharing a triangle with (u,v)
        # Triangles are (u, v, w) where w is common neighbor
        common = np.logical_and(A[u], A[v])
        # w indices
        ws = np.where(common)[0]
        for w in ws:
            book[u, w] -= 1
            book[w, u] -= 1
            book[v, w] -= 1
            book[w, v] -= 1
        
        # Remove edge (u,v)
        A[u, v] = False
        A[v, u] = False
        book[u, v] = 0
        book[v, u] = 0
        
        # Remove from edges list (efficiently? We can just filter later or use a set)
        # For now, we'll maintain a list and remove by value. N=400, so O(N) is fine.
        edges.remove((u, v))
        edges.remove((v, u)) if (v, u) in edges else None # Edges are stored as (u,v) with u<v? 
        # Let's enforce u < v in edges list to simplify removal.
        # In T3 construction above, I didn't enforce u<v.
        # Let's refactor to store edges as a set of frozensets or tuples with u<v.
        
    # Refactor to use set of tuples (u,v) with u < v for easy management
    # Re-initialize properly
    
    A = np.zeros((n, n), dtype=bool)
    edge_set = set()
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    A[u, v] = True
                    A[v, u] = True
                    if u < v:
                        edge_set.add((u, v))
                    else:
                        edge_set.add((v, u))
                        
    book = np.zeros((n, n), dtype=int)
    for i in range(3):
        for j in range(i + 1, 3):
            k = 3 - i - j
            size_k = part_sizes[k]
            for u in parts[i]:
                for v in parts[j]:
                    book[u, v] = size_k
                    book[v, u] = size_k

    def remove_edge(u, v):
        if u > v: u, v = v, u
        if (u, v) not in edge_set: return
        common = np.logical_and(A[u], A[v])
        ws = np.where(common)[0]
        for w in ws:
            book[u, w] -= 1
            book[w, u] -= 1
            book[v, w] -= 1
            book[w, v] -= 1
        
        A[u, v] = False
        A[v, u] = False
        book[u, v] = 0
        book[v, u] = 0
        edge_set.remove((u, v))

    def add_edge(u, v):
        if u > v: u, v = v, u
        if (u, v) in edge_set: return
        common = np.logical_and(A[u], A[v])
        ws = np.where(common)[0]
        for w in ws:
            book[u, w] += 1
            book[w, u] += 1
            book[v, w] += 1
            book[w, v] += 1
        
        A[u, v] = True
        A[v, u] = True
        book[u, v] = len(ws)
        book[v, u] = len(ws)
        edge_set.add((u, v))

    # Phase 1: Greedy Removal
    while len(edge_set) > target_edges:
        # Find edge with max book size
        max_val = get_max_book()
        # Find an edge with this value
        candidates = []
        for u, v in edge_set:
            if book[u, v] == max_val:
                candidates.append((u, v))
        
        if not candidates:
            break
            
        # Pick one (randomly if multiple)
        u, v = random.choice(candidates)
        remove_edge(u, v)
        
        if time.time() > deadline:
            break

    # Phase 2: Local Search (Edge Swaps)
    # We want to minimize max_book.
    current_max = get_max_book()
    
    # Precompute non-edges for faster sampling
    # Total possible edges: 40*39/2 = 780
    # Current edges: 401
    # Non-edges: 379
    
    def get_non_edges():
        ne = []
        for u in range(n):
            for v in range(u + 1, n):
                if not A[u, v]:
                    ne.append((u, v))
        return ne
    
    non_edges = get_non_edges()
    
    # To speed up, we can just iterate.
    # Strategy: 
    # 1. Identify edges with book size == current_max.
    # 2. For each such edge, try removing it and adding a random non-edge.
    # 3. If new max < current_max, accept.
    # 4. If new max == current_max, accept with low probability to explore.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_max = get_max_book()
        
        # Get edges with max book size
        max_edges = [(u, v) for u, v in edge_set if book[u, v] == current_max]
        
        if not max_edges:
            break
            
        # Shuffle max_edges to avoid bias
        random.shuffle(max_edges)
        
        for u_out, v_out in max_edges:
            if time.time() > deadline: break
            
            # Try swapping with a few random non-edges
            # Or iterate through all non-edges if fast enough?
            # 379 non-edges * 401 edges * cost of update.
            # Cost of update is O(N) ~ 40.
            # 379 * 401 * 40 ~ 6 million ops per full pass.
            # Python is slow. 6M ops might take 1-2 seconds.
            # So we can do a few full passes.
            
            # Let's sample non-edges to be safe on time
            # Or just iterate all, it's likely fast enough for N=40
            for u_in, v_in in non_edges:
                # Skip if same edge (shouldn't happen as it's non-edge)
                
                # Simulate removal of (u_out, v_out)
                # We need to calculate the change in max book size.
                # This is tricky to do in O(1).
                # We must apply, check, revert.
                
                # Apply removal
                remove_edge(u_out, v_out)
                # Apply addition
                add_edge(u_in, v_in)
                
                new_max = get_max_book()
                
                if new_max < current_max:
                    # Accept and keep
                    improved = True
                    # Update non_edges list since we added an edge
                    non_edges.remove((u_in, v_in)) if (u_in, v_in) in non_edges else None
                    # Note: non_edges is a list, removal is O(N). 
                    # We can just rebuild non_edges occasionally or keep a set.
                    # Let's use a set for non_edges for O(1) removal/addition.
                    break # Move to next max edge or restart? 
                    # If we accept, the set of max edges changes.
                    # It's safer to restart the outer loop logic or just continue and re-evaluate max.
                    # Given the small improvement, let's just break and let the while loop re-evaluate.
                else:
                    # Revert
                    remove_edge(u_in, v_in)
                    add_edge(u_out, v_out)
                    
                if time.time() > deadline: break
        
        if time.time() > deadline: break

    # Ensure we have exactly 401 edges
    # If we broke early, we might have 401 edges from Phase 1.
    # Local search maintains count.
    
    # Format output
    result = []
    for u, v in edge_set:
        result.append([u, v])
        
    return result

# Refinement: Use a set for non_edges to make updates O(1)
# And optimize the inner loop.

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 40
    target_edges = 401
    
    part_sizes = [14, 13, 13]
    parts = []
    idx = 0
    for size in part_sizes:
        parts.append(list(range(idx, idx + size)))
        idx += size
        
    A = np.zeros((n, n), dtype=bool)
    edge_set = set()
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    A[u, v] = True
                    A[v, u] = True
                    if u < v:
                        edge_set.add((u, v))
                    else:
                        edge_set.add((v, u))
                        
    book = np.zeros((n, n), dtype=int)
    for i in range(3):
        for j in range(i + 1, 3):
            k = 3 - i - j
            size_k = part_sizes[k]
            for u in parts[i]:
                for v in parts[j]:
                    book[u, v] = size_k
                    book[v, u] = size_k

    def remove_edge(u, v):
        if u > v: u, v = v, u
        if (u, v) not in edge_set: return
        common = np.logical_and(A[u], A[v])
        ws = np.where(common)[0]
        for w in ws:
            book[u, w] -= 1
            book[w, u] -= 1
            book[v, w] -= 1
            book[w, v] -= 1
        
        A[u, v] = False
        A[v, u] = False
        book[u, v] = 0
        book[v, u] = 0
        edge_set.remove((u, v))

    def add_edge(u, v):
        if u > v: u, v = v, u
        if (u, v) in edge_set: return
        common = np.logical_and(A[u], A[v])
        ws = np.where(common)[0]
        for w in ws:
            book[u, w] += 1
            book[w, u] += 1
            book[v, w] += 1
            book[w, v] += 1
        
        A[u, v] = True
        A[v, u] = True
        book[u, v] = len(ws)
        book[v, u] = len(ws)
        edge_set.add((u, v))

    def get_max_book():
        return np.max(book)

    # Phase 1: Greedy Removal
    while len(edge_set) > target_edges:
        max_val = get_max_book()
        candidates = []
        for u, v in edge_set:
            if book[u, v] == max_val:
                candidates.append((u, v))
        
        if not candidates:
            break
            
        u, v = random.choice(candidates)
        remove_edge(u, v)
        
        if time.time() > deadline:
            break

    # Phase 2: Local Search
    # Use a set for non-edges
    non_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if not A[u, v]:
                non_edges.add((u, v))
                
    current_max = get_max_book()
    
    while time.time() < deadline:
        current_max = get_max_book()
        max_edges = [(u, v) for u, v in edge_set if book[u, v] == current_max]
        
        if not max_edges:
            break
            
        random.shuffle(max_edges)
        found_improvement = False
        
        for u_out, v_out in max_edges:
            if time.time() > deadline: break
            if found_improvement: break
            
            # Iterate through non-edges
            # To avoid bias, shuffle non_edges? Or just iterate.
            # Iterating all is 379 items.
            for u_in, v_in in non_edges:
                # Remove out
                remove_edge(u_out, v_out)
                # Add in
                add_edge(u_in, v_in)
                
                new_max = get_max_book()
                
                if new_max < current_max:
                    # Accept
                    non_edges.remove((u_in, v_in))
                    # Note: (u_out, v_out) is now a non-edge
                    non_edges.add((u_out, v_out))
                    found_improvement = True
                    break
                else:
                    # Revert
                    remove_edge(u_in, v_in)
                    add_edge(u_out, v_out)
                    
                if time.time() > deadline: break
            
            if time.time() > deadline: break

    result = []
    for u, v in edge_set:
        result.append([u, v])
        
    return result

# Final check: The problem asks for [u, v] distinct edges.
# My edge_set stores tuples (u,v) with u<v.
# The output should be a list of lists.

