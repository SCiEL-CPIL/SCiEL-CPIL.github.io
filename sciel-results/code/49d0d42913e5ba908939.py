import time
import random
import numpy as np

def construct():
    deadline = time.time() + 145  # Safety margin before 150s limit
    n = 40
    m = 401
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_set = set(all_edges)
    
    # 1. Spectral Initialization: 
    # We want a graph with m=401 edges. 
    # K_{20,20} has 400 edges. Adding 1 edge creates a book of size 20 (bad).
    # Instead, we start with a regular-like graph.
    # Average degree d = 2*401/40 = 20.05.
    # A 20-regular graph on 40 vertices is possible (since 40*20 is even).
    # We can use a circulant graph or random regular graph.
    # Let's try to construct a 20-regular graph first, then add 1 edge.
    # Or better, construct a graph with degrees close to 20.
    
    # Strategy: Start with a random 20-regular graph, then add the "least harmful" edge.
    # Or, start with K_{20,20} and perform edge swaps to reduce max book.
    # K_{20,20} has max book 20 (for edges in the added part? No, K_{20,20} has no triangles).
    # Wait, K_{20,20} is bipartite, so it has 0 triangles. 
    # If we add 1 edge to K_{20,20}, say (u,v) within a part, it creates 20 triangles.
    # So max book is 20.
    # If we swap an edge from K_{20,20} to create a more mixed structure, we might lower the max book.
    
    # Let's use a random 20-regular graph as the base.
    # Generating a random regular graph is non-trivial but doable.
    # Alternatively, use a circulant graph with connection set S.
    # For n=40, a 20-regular circulant graph has connection set of size 10 (symmetric) or 20 (if not symmetric).
    # Let's try a circulant graph with steps {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}.
    # This is the complement of the cycle graph? No.
    # The complete graph K40 has degree 39.
    # A 20-regular circulant: connect i to i+1, ..., i+10 and i-1, ..., i-10.
    # This graph has high clustering?
    
    # Let's try a different initialization: Random regular graph.
    # We can approximate by taking a random permutation and adding edges, then fixing degrees.
    # Simpler: Start with a random graph with p=0.5 (approx 396 edges) and adjust.
    
    # Let's use a bitmask representation for speed.
    # adj_mask[i] is an integer where bit j is 1 if edge (i,j) exists.
    
    def compute_max_book(adj_mask):
        max_book = 0
        for i in range(n):
            for j in range(i+1, n):
                if (adj_mask[i] >> j) & 1:
                    common = (adj_mask[i] & adj_mask[j]).bit_count()
                    if common > max_book:
                        max_book = common
        return max_book

    def get_edge_list(adj_mask):
        edges = []
        for i in range(n):
            mask = adj_mask[i]
            # Only consider j > i to avoid duplicates
            mask &= ~((1 << (i+1)) - 1) 
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                edges.append([i, j])
                mask ^= lsb
        return edges

    best_score = float('inf')
    best_adj_mask = None

    # Initialize with a few different structures
    
    # Structure 1: Random 20-regular graph (approx)
    # We can generate a random permutation and create a "shift" graph
    # Let's just use random restarts with a good local search.
    # The key is the local search:
    # 1. Find edge (u,v) with max book size.
    # 2. Try to swap it: remove (u,v), add (x,y) where x,y are non-adjacent.
    #    Choose (x,y) such that it minimizes the increase in book sizes for edges incident to x,y.
    #    Specifically, we want to avoid creating new high-book edges.
    
    # To make local search fast, we need to update book sizes incrementally or recompute efficiently.
    # Recomputing max book for 401 edges takes ~400 * bit_count operations.
    # bit_count is very fast. 400 * 40 operations per evaluation is ~16000 ops.
    # If we can do 1000 swaps/sec, that's 16M ops/sec, which is fine in Python with bitmasks?
    # Actually, Python loops are slow. Let's see.
    # 401 edges. For each edge, bit_count of 40-bit int.
    # This is very fast.
    
    # Let's implement a local search that performs many swaps.
    
    def local_search(adj_mask, time_limit):
        start_time = time.time()
        current_score = compute_max_book(adj_mask)
        
        # Precompute adjacency lists for faster neighbor iteration?
        # Bitmask is enough.
        
        while time.time() < time_limit:
            # Find the edge with the maximum book size
            max_book_edge = None
            max_book_val = -1
            for i in range(n):
                mask_i = adj_mask[i]
                # Iterate over neighbors j > i
                mask_j = mask_i & ~((1 << (i+1)) - 1)
                while mask_j:
                    lsb = mask_j & -mask_j
                    j = lsb.bit_length() - 1
                    common = (adj_mask[i] & adj_mask[j]).bit_count()
                    if common > max_book_val:
                        max_book_val = common
                        max_book_edge = (i, j)
                    mask_j ^= lsb
            
            if max_book_val < 5: # Theoretical min is 7? If we get below 7, we are great.
                break
                
            u, v = max_book_edge
            
            # Try to remove (u,v) and add a new edge (x,y)
            # We want to choose (x,y) such that the new book sizes are small.
            # Ideally, x and y should have low common neighbors with each other and with their neighbors.
            
            # Candidates for (x,y):
            # 1. x, y are not adjacent.
            # 2. x != u, x != v, y != u, y != v (to keep edge count same? No, we swap, so count stays same).
            # Actually, if we remove (u,v) and add (x,y), the degree of u,v drops by 1, x,y rises by 1.
            # We should prefer x,y that are not too connected.
            
            # Let's sample a few random non-edges and check if the swap improves the max book.
            # To be faster, we can check the potential book size of the new edge (x,y)
            # and the impact on neighbors.
            
            improved = False
            # Try a limited number of random swaps to save time
            for _ in range(20):
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y: continue
                if x > y: x, y = y, x
                
                # Check if (x,y) is an edge
                if (adj_mask[x] >> y) & 1: continue
                
                # Check if (u,v) is an edge (it is)
                
                # Perform swap
                old_mask_u = adj_mask[u]
                old_mask_v = adj_mask[v]
                old_mask_x = adj_mask[x]
                old_mask_y = adj_mask[y]
                
                # Remove (u,v)
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                
                # Add (x,y)
                adj_mask[x] |= (1 << y)
                adj_mask[y] |= (1 << x)
                
                new_score = compute_max_book(adj_mask)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    # Revert
                    adj_mask[u] = old_mask_u
                    adj_mask[v] = old_mask_v
                    adj_mask[x] = old_mask_x
                    adj_mask[y] = old_mask_y
            
            if not improved:
                # Try to find a better swap more systematically?
                # Or just break to avoid getting stuck?
                # Let's try a few more random restarts within this local search?
                # No, let's just break and start a new global restart.
                break

        return current_score, adj_mask

    # Global restarts
    for seed in range(100):
        if time.time() > deadline - 5:
            break
            
        random.seed(seed)
        
        # Initialize with a random graph with 401 edges
        # To avoid high initial book sizes, we can start with a sparse random graph
        # and add edges, or start with a random regular-like graph.
        
        # Let's try to construct a 20-regular graph approximately.
        # Generate a random permutation of vertices.
        perm = list(range(n))
        random.shuffle(perm)
        
        # Create a circulant-like graph based on this permutation?
        # Simpler: Random graph with p=0.5, then adjust to 401 edges.
        
        adj_mask = [0] * n
        edge_count = 0
        
        # Add random edges until we have 401
        # To make it more regular, we can prioritize vertices with low degree.
        # Let's just sample 401 edges randomly.
        # This is the same as the seed program, but we will apply local search.
        
        # Better initialization: 
        # Start with K_{20,20} (0 triangles) and add 1 edge (max book 20).
        # Then use local search to reduce the book size.
        # This might be a good starting point because the "bad" structure is localized.
        
        # Let's try two initializations:
        # 1. Random 401 edges
        # 2. K_{20,20} + 1 random edge
        
        init_type = seed % 2
        
        if init_type == 0:
            # Random
            edges = random.sample(all_edges, m)
            for u, v in edges:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        else:
            # K_{20,20} + 1 edge
            for i in range(20):
                for j in range(20, 40):
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
            # Add one edge within a part
            u = random.randint(0, 19)
            v = random.randint(0, 19)
            if u != v:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
            else:
                # If u==v, pick another
                v = (u + 1) % 20
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                
        # Local search
        score, adj_mask = local_search(adj_mask, deadline - 2)
        
        if score < best_score:
            best_score = score
            best_adj_mask = adj_mask.copy()
            
        if best_score <= 7:
            # We found a very good solution. Can we do better?
            # The theorem says >= 40/6 = 6.66, so 7 is the integer minimum.
            # If we get 7, we are likely optimal.
            # Let's try to push for 6? No, 6 < 6.66, so 7 is the minimum possible integer.
            # Wait, 40/6 = 6.66. So score >= 7.
            # If we find 7, we can stop.
            break

    if best_adj_mask is None:
        # Fallback
        adj_mask = [0] * n
        for i in range(20):
            for j in range(20, 40):
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
        adj_mask[0] |= (1 << 1)
        adj_mask[1] |= (1 << 0)
        best_adj_mask = adj_mask

    return get_edge_list(best_adj_mask)
