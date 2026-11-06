import numpy as np
import random
import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 40
    target_edges = 401
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.seed(42)
    
    # Initialize
    edges = set(random.sample(all_edges, target_edges))
    neighbors = [set() for _ in range(n)]
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
        
    # Initialize book sizes
    book_sizes = {}
    for u, v in edges:
        book_sizes[(u,v)] = len(neighbors[u].intersection(neighbors[v]))
        
    # Max heap for book sizes
    # Heap items: (-size, u, v)
    heap = [(-size, u, v) for (u, v), size in book_sizes.items()]
    heapq.heapify(heap)
    
    def get_current_max():
        while heap:
            neg_size, u, v = heap[0]
            size = -neg_size
            # Check if this is still the current book size for this edge
            if (u, v) in book_sizes and book_sizes[(u, v)] == size:
                return size
            else:
                heapq.heappop(heap)
        return 0
        
    current_max = get_current_max()
    best_max = current_max
    best_edges = set(edges)
    
    T = 10.0
    cooling = 0.999
    
    # Helper to update book sizes for a node's incident edges
    # When a neighbor is added/removed from node 'node', 
    # the book size of edge (node, z) changes by +/- 1 if the neighbor is in N(z).
    
    while time.time() < deadline:
        # Pick edge to remove
        if not edges: break
        u, v = random.choice(list(edges))
        
        # Pick non-edge to add
        while True:
            x, y = random.sample(range(n), 2)
            if x > y: x, y = y, x
            if (x, y) not in edges:
                break
                
        # --- APPLY CHANGE ---
        
        # 1. Remove (u, v)
        # Update book sizes for edges incident to u and v
        # For each z in N(u) (excluding v):
        #   If v in N(z), book_size[(u,z)] -= 1
        for z in list(neighbors[u]):
            if z == v: continue
            edge_key = (u, z) if u < z else (z, u)
            if v in neighbors[z]:
                book_sizes[edge_key] -= 1
                heapq.heappush(heap, (-book_sizes[edge_key], edge_key[0], edge_key[1]))
                
        for z in list(neighbors[v]):
            if z == u: continue
            edge_key = (v, z) if v < z else (z, v)
            if u in neighbors[z]:
                book_sizes[edge_key] -= 1
                heapq.heappush(heap, (-book_sizes[edge_key], edge_key[0], edge_key[1]))
                
        # Remove (u,v) from structures
        del book_sizes[(u,v) if u<v else (v,u)]
        edges.remove((u,v))
        neighbors[u].remove(v)
        neighbors[v].remove(u)
        
        # 2. Add (x, y)
        # Calculate book size of new edge (x,y)
        # It is |N(x) intersection N(y)|
        new_book_size_xy = len(neighbors[x].intersection(neighbors[y]))
        edge_key_xy = (x, y) if x < y else (y, x)
        book_sizes[edge_key_xy] = new_book_size_xy
        heapq.heappush(heap, (-new_book_size_xy, edge_key_xy[0], edge_key_xy[1]))
        
        # Update book sizes for edges incident to x and y
        # For each z in N(x):
        #   If y in N(z), book_size[(x,z)] += 1
        for z in list(neighbors[x]):
            edge_key = (x, z) if x < z else (z, x)
            if y in neighbors[z]:
                book_sizes[edge_key] += 1
                heapq.heappush(heap, (-book_sizes[edge_key], edge_key[0], edge_key[1]))
                
        for z in list(neighbors[y]):
            edge_key = (y, z) if y < z else (z, y)
            if x in neighbors[z]:
                book_sizes[edge_key] += 1
                heapq.heappush(heap, (-book_sizes[edge_key], edge_key[0], edge_key[1]))
                
        # Add (x,y) to structures
        edges.add((x,y))
        neighbors[x].add(y)
        neighbors[y].add(x)
        
        # --- EVALUATE ---
        new_max = get_current_max()
        
        if new_max < best_max:
            best_max = new_max
            best_edges = set(edges)
            
        # SA Acceptance
        delta = new_max - current_max
        if delta < 0:
            current_max = new_max
        else:
            if random.random() < np.exp(-delta / T):
                current_max = new_max
            else:
                # REVERT
                # This is complex to revert incrementally correctly if we made mistakes.
                # Given the low N, reverting by re-applying the inverse operations is safe.
                
                # Inverse of Add (x,y) is Remove (x,y)
                # Inverse of Remove (u,v) is Add (u,v)
                
                # Remove (x,y)
                for z in list(neighbors[x]):
                    edge_key = (x, z) if x < z else (z, x)
                    if y in neighbors[z]:
                        book_sizes[edge_key] -= 1
                        heapq.heappush(heap, (-book_sizes[edge_key], edge_key[0], edge_key[1]))
                for z in list(neighbors[y]):
                    edge_key = (y, z) if y < z else (z, y)
                    if x in neighbors[z]:
                        book_sizes[edge_key] -= 1
                        heapq.heappush(heap, (-book_sizes[edge_key], edge_key[0], edge_key[1]))
                        
                del book_sizes[edge_key_xy]
                edges.remove((x,y))
                neighbors[x].remove(y)
                neighbors[y].remove(x)
                
                # Add (u,v)
                # Book size of (u,v) is |N(u) intersection N(v)|
                # Note: N(u) and N(v) are currently missing each other.
                # So intersection is |N(u) \ {v} \cap N(v) \ {u}| which is correct.
                new_book_size_uv = len(neighbors[u].intersection(neighbors[v]))
                edge_key_uv = (u, v) if u < v else (v, u)
                book_sizes[edge_key_uv] = new_book_size_uv
                heapq.heappush(heap, (-new_book_size_uv, edge_key_uv[0], edge_key_uv[1]))
                
                for z in list(neighbors[u]):
                    edge_key = (u, z) if u < z else (z, u)
                    if v in neighbors[z]:
                        book_sizes[edge_key] += 1
                        heapq.heappush(heap, (-book_sizes[edge_key], edge_key[0], edge_key[1]))
                for z in list(neighbors[v]):
                    edge_key = (v, z) if v < z else (z, v)
                    if u in neighbors[z]:
                        book_sizes[edge_key] += 1
                        heapq.heappush(heap, (-book_sizes[edge_key], edge_key[0], edge_key[1]))
                        
                edges.add((u,v))
                neighbors[u].add(v)
                neighbors[v].add(u)
                
                # Update current_max to reflect revert
                current_max = get_current_max()

        T *= cooling
        
    return [[u, v] for u, v in best_edges]
