import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    n = 40
    target_edges = 401
    
    # Helper to normalize edge
    def norm(u, v):
        return (u, v) if u < v else (v, u)
    
    # Helper to compute score (max book)
    def get_score(adj_matrix, edges_set):
        max_book = 0
        for u, v in edges_set:
            # Common neighbors of u and v
            # adj_matrix is n x n boolean (or 0/1)
            # common = sum(adj_matrix[u] & adj_matrix[v])
            # Since adj_matrix[u][u] is 0, this is exactly the number of triangles containing (u,v)
            count = np.sum(adj_matrix[u] & adj_matrix[v])
            if count > max_book:
                max_book = count
        return max_book

    # 1. Spectral Initialization
    # We want to start with a graph close to a bipartite graph K_{20,20} + 1 edge.
    # K_{20,20} has 400 edges and max book 0.
    # Adding 1 edge creates a few triangles.
    # Spectral partitioning of a random graph or just using index parity is a good start.
    # Let's try to construct a near-bipartite graph directly using spectral methods on a random start?
    # Actually, simpler: Start with K_{20,20} + 1 random edge.
    # Then refine.
    
    # Let's try a slightly more robust initialization:
    # Random graph G(n, 401/780)
    # Then run a few steps of "bipartite projection" or just use the random graph and let local search do the work.
    # However, the current program uses random init. Let's try a structured init.
    
    # Strategy: 
    # 1. Start with a random graph.
    # 2. Compute spectral embedding of the graph to find a bipartition.
    # 3. Rearrange vertices to minimize edges within parts (making it more bipartite).
    # 4. Then do local search.
    
    # Actually, for n=40, we can afford to be smarter.
    # Let's initialize with a random graph, but then immediately try to "bipartize" it.
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Initial random graph
    random.shuffle(all_pairs)
    current_edges = set(all_pairs[:target_edges])
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.uint8)
    for u, v in current_edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    current_score = get_score(adj, current_edges)
    best_edges = set(current_edges)
    best_score = current_score
    
    # Phase 1: Spectral Bipartization
    # Goal: Move towards a complete bipartite graph structure.
    # We can iteratively find the best cut (bipartition) of the current graph and keep the edges crossing the cut.
    # If we keep only crossing edges, we get a bipartite graph.
    # We need exactly 401 edges.
    
    # Let's try a few rounds of "spectral flip"
    # 1. Get eigenvector of smallest eigenvalue (or second smallest if connected)
    # 2. Partition vertices into two sets based on sign
    # 3. Identify edges within sets (bad) and across sets (good)
    # 4. Swap "bad" edges with missing "good" edges.
    
    for _ in range(20): # 20 rounds of spectral improvement
        if time.time() > deadline:
            break
            
        # Compute Laplacian
        L = np.diag(np.sum(adj, axis=1)) - adj.astype(float)
        
        # Eigen decomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
        except:
            break
            
        # Use the eigenvector corresponding to the smallest non-zero eigenvalue (Fiedler vector)
        # For a connected graph, index 1. If disconnected, might be 0.
        # Let's just take the one with the second smallest eigenvalue, assuming connectedness or close to it.
        # If the graph is bipartite, the smallest eigenvalue is 0, and the second is small.
        # Actually, for bipartite graphs, the spectrum is symmetric around 0.
        # The vector that separates the parts is the one for eigenvalue 0 (if disconnected) or near 0.
        # Let's pick the eigenvector for the smallest eigenvalue > 0.
        
        # Find index of smallest positive eigenvalue
        pos_eigs = eigenvalues[eigenvalues > 1e-6]
        if len(pos_eigs) == 0:
            idx = 0
        else:
            idx = np.argmin(eigenvalues) + 1 # Skip the 0 eigenvalue if present, or just take the smallest
        
        # More robust: take the eigenvector that minimizes the cut
        # Usually the Fiedler vector (2nd smallest eigenvalue) is used for bisection.
        idx = 1
        
        vec = eigenvectors[:, idx]
        
        # Partition
        # Mean centering
        vec = vec - np.mean(vec)
        # Split at 0
        part_a = np.where(vec < 0)[0]
        part_b = np.where(vec >= 0)[0]
        
        if len(part_a) == 0 or len(part_b) == 0:
            continue
            
        # Identify edges to swap
        # Bad edges: both endpoints in part_a or both in part_b
        # Good missing edges: one in part_a, one in part_b
        
        bad_edges = []
        for u, v in current_edges:
            u_in_a = u in part_a
            v_in_a = v in part_a
            if u_in_a == v_in_a: # Both in same part
                bad_edges.append((u, v))
                
        # Shuffle bad edges to break ties
        random.shuffle(bad_edges)
        
        # Find missing good edges
        # We need to add edges between A and B
        # Iterate through all possible pairs in A x B and check if missing
        
        # To be efficient, we can just pick random candidates
        # But let's find a few specific ones
        
        swaps_done = 0
        max_swaps = len(bad_edges)
        
        for u, v in bad_edges:
            if time.time() > deadline:
                break
            if swaps_done >= max_swaps:
                break
                
            # Try to find a missing edge (x, y) where x in A, y in B
            # And ideally, this swap reduces the score.
            # For now, just do random valid swaps to push towards bipartite
            
            # Pick random x in A, y in B
            x = random.choice(part_a)
            y = random.choice(part_b)
            
            new_edge = norm(x, y)
            
            if new_edge in current_edges:
                continue
                
            # Perform swap
            # Remove (u,v)
            current_edges.remove((u, v))
            adj[u, v] = 0
            adj[v, u] = 0
            
            # Add (x,y)
            current_edges.add(new_edge)
            adj[x, y] = 1
            adj[y, x] = 1
            
            swaps_done += 1
            
        # Re-evaluate score
        new_score = get_score(adj, current_edges)
        if new_score < best_score:
            best_score = new_score
            best_edges = set(current_edges)
        current_score = new_score

    # Phase 2: Greedy Local Search
    # Now we are likely close to a bipartite graph.
    # Max book should be small.
    # We want to minimize the max book.
    
    # Precompute degrees for faster neighbor iteration?
    # For n=40, iterating over all w is fine.
    
    while time.time() < deadline:
        # Find the edge with the maximum book
        max_book = -1
        max_edge = None
        edge_list = list(current_edges)
        
        # We can optimize this by only checking edges that might be max
        # But for n=40, checking all 401 edges is fast.
        
        # Actually, let's maintain a list of "hot" edges?
        # No, just recompute.
        
        for u, v in edge_list:
            count = np.sum(adj[u] & adj[v])
            if count > max_book:
                max_book = count
                max_edge = (u, v)
                
        if max_book == 0:
            break # Optimal reached
        
        # Try to improve the max_edge
        # Options:
        # 1. Remove max_edge and add a different edge.
        # 2. Keep max_edge but modify neighbors to reduce triangles.
        
        # Strategy: Remove max_edge (u,v).
        # This reduces the book of (u,v) to 0 (it's gone).
        # But it might increase the book of other edges? No, removing an edge cannot increase the number of triangles for any other edge.
        # Wait, the score is the MAX book.
        # If we remove the edge with the max book, the new max book will be <= old max book.
        # It might be strictly less if no other edge had the same max book.
        # If multiple edges had the max book, removing one might not change the max.
        
        # So, removing the max book edge is always a non-increasing move for the score.
        # However, we must add an edge back to maintain count 401.
        # Adding an edge might INCREASE the book of some other edge.
        
        # So the move is:
        # 1. Remove (u,v).
        # 2. Add (x,y) such that the new max book is minimized.
        
        # Let's evaluate candidate edges to add.
        # We want to add an edge (x,y) that creates as few triangles as possible.
        # Ideally, an edge with 0 common neighbors.
        
        # Candidate edges: all non-edges.
        # There are 780 - 400 = 380 non-edges.
        # Checking all 380 non-edges for each step might be slow if we do it every step.
        # 380 * (cost of checking common neighbors)
        # Cost of checking common neighbors for (x,y) is O(1) with bitsets or O(n) with loops.
        # With numpy, np.sum(adj[x] & adj[y]) is very fast.
        
        # Let's try all non-edges?
        # 380 numpy sums per step.
        # If we can do ~1000 steps in 100s, that's 380k numpy ops. Should be fine.
        
        # But we can be smarter.
        # We only care about edges that don't increase the max book too much.
        
        # Let's sample a subset of non-edges if time is tight, or just iterate all.
        
        # First, remove (u,v)
        u, v = max_edge
        current_edges.remove((u, v))
        adj[u, v] = 0
        adj[v, u] = 0
        
        # Calculate the current max book WITHOUT (u,v)
        # This is the "base" max book.
        # We need to find (x,y) to add that minimizes the resulting max book.
        
        # Let's pre-compute the books of all remaining edges?
        # That's 400 edges.
        # Books = {edge: count}
        
        # Actually, we can just find the second largest book in the current graph (without u,v).
        # Let second_max_book be the max book among all edges except (u,v).
        
        # If we add (x,y), the new max book is max(second_max_book, book(x,y)).
        # So we want to minimize book(x,y).
        # Ideally book(x,y) <= second_max_book.
        
        # Let's find the min possible book(x,y) among non-edges.
        
        # Generate list of non-edges
        # This is expensive to generate from scratch every time.
        # Maintain a list of non-edges?
        
        # For n=40, let's just iterate all pairs and check if in current_edges.
        
        best_new_edge = None
        min_book_for_new_edge = n # Upper bound
        
        # Optimization: Only check edges that could potentially have low book.
        # Edges between vertices with low degrees?
        
        # Let's just iterate all 780 pairs.
        for x in range(n):
            for y in range(x+1, n):
                if (x, y) in current_edges:
                    continue
                # Calculate book for (x,y)
                # Note: adj does not contain (u,v) currently
                count = np.sum(adj[x] & adj[y])
                if count < min_book_for_new_edge:
                    min_book_for_new_edge = count
                    best_new_edge = (x, y)
                    
                    if count == 0:
                        break # Can't do better
            if min_book_for_new_edge == 0:
                break
                
        if best_new_edge is None:
            # Should not happen
            current_edges.add((u, v))
            adj[u, v] = 1
            adj[v, u] = 1
            continue
            
        # Add the best new edge
        x, y = best_new_edge
        current_edges.add(best_new_edge)
        adj[x, y] = 1
        adj[y, x] = 1
        
        new_score = get_score(adj, current_edges)
        
        if new_score < best_score:
            best_score = new_score
            best_edges = set(current_edges)
            
        current_score = new_score
        
        # If we didn't improve the global best, we might be stuck in a local minimum
        # where removing the max edge and adding the best possible edge still results in the same max book.
        # In that case, we need a random perturbation.
        
        if new_score >= best_score:
            # Random restart or perturbation
            # Let's do a random swap
            if time.time() > deadline:
                break
                
            # Pick random edge to remove
            rem_edge = random.choice(list(current_edges))
            # Pick random non-edge to add
            add_edge = None
            attempts = 0
            while add_edge is None and attempts < 20:
                a, b = random.sample(range(n), 2)
                e = norm(a, b)
                if e not in current_edges:
                    add_edge = e
                attempts += 1
                
            if add_edge is None:
                continue
                
            current_edges.remove(rem_edge)
            adj[rem_edge[0], rem_edge[1]] = 0
            adj[rem_edge[1], rem_edge[0]] = 0
            
            current_edges.add(add_edge)
            adj[add_edge[0], add_edge[1]] = 1
            adj[add_edge[1], add_edge[0]] = 1
            
            new_score = get_score(adj, current_edges)
            if new_score < best_score:
                best_score = new_score
                best_edges = set(current_edges)
            current_score = new_score

    # Return best found
    return [[u, v] for u, v in best_edges]
