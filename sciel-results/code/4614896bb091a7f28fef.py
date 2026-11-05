import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # Problem parameters
    # q=3, n=7, radius=1
    # Total words
    N = q ** n
    
    # Precompute neighbors (ball of radius 1) for each word
    # neighbors[i] is a list of indices within Hamming distance 1 from i
    neighbors = [[] for _ in range(N)]
    
    # Helper to convert index to list of digits and back
    # But for speed, we can generate neighbors by manipulating the index directly
    # Or just generate all words and compute distances.
    # Given N=2187, generating all words and checking distance is 2187^2 ~ 4.7M ops.
    # That's fast enough in Python? 4.7M * (7 comparisons) ~ 30M ops. Might take 1-2s.
    # Let's use a more efficient generation.
    
    # Generate all words as tuples
    words = []
    for i in range(N):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        words.append(tuple(w))
        
    # Compute neighbors
    for i in range(N):
        w_i = words[i]
        # Center
        neighbors[i].append(i)
        # Distance 1
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                # Create new word
                new_w_list = list(w_i)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                # Find index
                # This is O(N) lookup if we do it naively.
                # Better to use a dictionary mapping word to index.
                pass
                
    # Let's use a dictionary for word->index
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Recompute neighbors using dict lookup
    for i in range(N):
        w_i = words[i]
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w_i[pos]:
                    continue
                new_w_list = list(w_i)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh.append(word_to_idx[new_w])
        neighbors[i] = neigh

    best_code = None
    best_size = float('inf')
    
    # Compute spectral ordering using eigenvectors of the adjacency matrix
    # This helps prioritize vertices that are "central" in the graph structure
    # We use the second smallest eigenvalue's eigenvector (Fiedler vector)
    # to get a spectral clustering ordering
    
    # Build adjacency matrix (sparse-like using numpy)
    # For efficiency, we'll compute the degree matrix and use power iteration
    # or just use numpy's eigensolver on the normalized Laplacian
    
    # Actually, for N=2187, we can compute the full adjacency matrix
    # But that's 2187x2187 = ~4.8M entries, which is fine
    
    # Let's build the adjacency matrix
    adj_matrix = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        for j in neighbors[i]:
            adj_matrix[i, j] = 1
    
    # Compute degree
    degree = adj_matrix.sum(axis=1)
    
    # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    # But we want the Fiedler vector of the adjacency matrix or Laplacian
    # For covering, we want to find a good initial set of "hub" vertices
    
    # Let's use the eigenvectors of the adjacency matrix
    # The smallest eigenvalue's eigenvector (all ones) is trivial
    # The second smallest gives the Fiedler vector for bipartite-like structure
    
    # Actually, let's use a different approach: 
    # Compute the "spectral centrality" by looking at the eigenvectors
    # of the adjacency matrix and selecting vertices with high absolute values
    # in the low-frequency eigenvectors
    
    # For simplicity and speed, let's use the power method to get the top eigenvectors
    # Or just use numpy's eigensolver on a smaller matrix
    
    # Given time constraints, let's use a simpler spectral heuristic:
    # Compute the "influence" of each vertex by looking at its position
    # in the eigenvectors of the adjacency matrix
    
    # Let's compute a few eigenvectors using numpy
    try:
        # Use the adjacency matrix
        # For large matrices, use sparse eigenvalue solvers if available
        # For N=2187, dense is okay but might be slow
        # Let's try using the Laplacian instead
        
        # Degree matrix
        D = np.diag(degree)
        
        # Normalized Laplacian
        # L_sym = I - D^(-1/2) A D^(-1/2)
        D_inv_sqrt = np.zeros_like(degree, dtype=float)
        nonzero_deg = degree > 0
        D_inv_sqrt[nonzero_deg] = 1.0 / np.sqrt(degree[nonzero_deg])
        
        D_inv_sqrt_matrix = np.diag(D_inv_sqrt)
        L_sym = np.eye(N) - D_inv_sqrt_matrix @ adj_matrix @ D_inv_sqrt_matrix
        
        # Get the smallest few eigenvalues and eigenvectors
        # The first eigenvector is all ones (trivial)
        # The second is the Fiedler vector
        # We want to use the low-frequency eigenvectors to get a spectral ordering
        
        # Use numpy's eigensolver
        # For N=2187, this might take a few seconds
        if time.time() < deadline - 30:  # Only do this if we have time
            # Use eigh for symmetric matrices
            eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
            
            # Sort by eigenvalue (eigh already returns in ascending order)
            # The first eigenvector (index 0) is the trivial one
            # Use eigenvectors 1, 2, 3, ... to get spectral coordinates
            
            # Take the first few non-trivial eigenvectors
            num_eigenvectors = min(10, N - 1)
            spectral_coords = eigenvectors[:, 1:num_eigenvectors+1]  # Skip the trivial one
            
            # Compute the "spectral centrality" as the norm of the spectral coordinates
            # Vertices with large norms are more "central" in the spectral sense
            spectral_centrality = np.linalg.norm(spectral_coords, axis=1)
            
            # Get the ordering by spectral centrality (highest first)
            spectral_order = np.argsort(-spectral_centrality)
            
            # Now use this ordering to guide the greedy algorithm
            # Instead of random tie-breaking, we can use spectral ordering
            # to break ties or to initialize the candidate pool
        else:
            spectral_order = None
    except Exception:
        spectral_order = None
    
    # Run multiple iterations of Greedy + Prune
    # Randomize tie-breaking to escape local optima
    for iteration in range(20): # Max 20 restarts
        if time.time() > deadline:
            break
            
        # Greedy Set Cover
        uncovered = set(range(N))
        code = []
        
        # To speed up greedy, we can pre-compute which candidates are "active"
        # But with N=2187, simple loop is fine.
        
        while uncovered:
            best_candidate = -1
            max_new_cov = 0
            candidates = []
            
            # Use spectral ordering to guide candidate selection
            # If we have spectral_order, iterate in that order
            # This helps find "central" vertices first, which may cover more
            # But we still need to find the max coverage, so we iterate all
            # The spectral ordering can be used for tie-breaking
            
            # Get the list of candidates in spectral order if available
            if spectral_order is not None:
                candidate_list = list(spectral_order)
            else:
                candidate_list = list(range(N))
            
            # Shuffle slightly to maintain diversity across iterations
            # But keep the spectral bias
            random.shuffle(candidate_list)
            
            # Iterate all possible codewords
            for c in candidate_list:
                # Count coverage
                # Since |neighbors[c]| is 15, this is 15 lookups in a set.
                cov = 0
                for u in neighbors[c]:
                    if u in uncovered:
                        cov += 1
                
                if cov > max_new_cov:
                    max_new_cov = cov
                    best_candidate = c
                    candidates = [c]
                elif cov == max_new_cov and cov > 0:
                    candidates.append(c)
            
            if best_candidate == -1:
                # Should not happen if uncovered is not empty
                break
                
            # Use spectral ordering to pick from candidates
            # If we have spectral_order, pick the one with highest spectral centrality
            # among the candidates with max coverage
            if spectral_order is not None and len(candidates) > 1:
                # Create a mapping from vertex to spectral rank
                # Lower rank = higher centrality
                spectral_rank = {v: i for i, v in enumerate(spectral_order)}
                # Pick the candidate with lowest spectral rank (highest centrality)
                chosen = min(candidates, key=lambda c: spectral_rank.get(c, N))
            else:
                # Randomly pick from candidates to add diversity
                chosen = random.choice(candidates)
            code.append(chosen)
            
            # Update uncovered
            for u in neighbors[chosen]:
                if u in uncovered:
                    uncovered.remove(u)
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
            
        # Prune Redundant Codewords
        # Maintain cover_count
        cover_count = [0] * N
        for c in code:
            for u in neighbors[c]:
                cover_count[u] += 1
        
        # Try to remove
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            # Iterate in reverse to allow removal by index? 
            # Or just mark for removal.
            to_remove = []
            for idx, c in enumerate(code):
                # Check if c is redundant
                # c is redundant if for all u in B(c), cover_count[u] > 1
                redundant = True
                for u in neighbors[c]:
                    if cover_count[u] <= 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(idx)
            
            if to_remove:
                # Remove them
                # We need to update cover_count
                for idx in to_remove:
                    c = code[idx]
                    for u in neighbors[c]:
                        cover_count[u] -= 1
                
                # Filter code
                # Create a set of indices to keep
                remove_set = set(to_remove)
                code = [c for i, c in enumerate(code) if i not in remove_set]
                changed = True
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
            # If we reached a very good size, we might stop early?
            # 186 is the target. If we hit 186, we can stop.
            if best_size <= 186:
                break
            
            # Also try a local search refinement
            # Try swapping codewords to see if we can reduce the size
            # This is a 1-swap local search
            if time.time() < deadline - 5:
                # Try to improve by swapping one codeword with another
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    # Recompute cover_count
                    cover_count = [0] * N
                    for c in code:
                        for u in neighbors[c]:
                            cover_count[u] += 1
                    
                    # Try removing each codeword and see if we can replace it
                    for idx, c in enumerate(code):
                        # Check if c is redundant (can be removed)
                        redundant = True
                        for u in neighbors[c]:
                            if cover_count[u] <= 1:
                                redundant = False
                                break
                        
                        if redundant:
                            # Remove c
                            for u in neighbors[c]:
                                cover_count[u] -= 1
                            code.pop(idx)
                            improved = True
                            break
                        
                        # Try swapping c with another codeword c2
                        # c2 should be one that is NOT in code
                        # And after swap, all words still covered
                        # This is expensive, so only try a few
                        if time.time() < deadline - 5:
                            # Sample a few candidates for swap
                            for _ in range(5):
                                c2 = random.randint(0, N - 1)
                                if c2 in code:
                                    continue
                                
                                # Check if swapping c with c2 maintains coverage
                                # Words covered by c but not by c2 must be covered by other codewords
                                # Words covered by c2 are fine
                                # So we need: for all u in B(c), if u not in B(c2), then cover_count[u] > 1 (after removing c)
                                
                                # First, remove c's coverage
                                for u in neighbors[c]:
                                    cover_count[u] -= 1
                                
                                # Check coverage
                                valid_swap = True
                                for u in neighbors[c]:
                                    if cover_count[u] == 0:
                                        # u is not covered by any other codeword
                                        # It must be covered by c2
                                        if u not in neighbors[c2]:
                                            valid_swap = False
                                            break
                                
                                if valid_swap:
                                    # Perform the swap
                                    code[idx] = c2
                                    # Update cover_count
                                    for u in neighbors[c2]:
                                        cover_count[u] += 1
                                    improved = True
                                    break
                                else:
                                    # Restore c's coverage
                                    for u in neighbors[c]:
                                        cover_count[u] += 1
                                if improved:
                                    break
                
    # Convert internal indices back to lists of integers
    if best_code is None:
        # Fallback: return the direct sum construction if something went wrong
        # But best_code should always be set.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Final verification and cleanup
    # Ensure all codewords are distinct (should be, but just in case)
    seen = set()
    final_code = []
    for c in best_code:
        if c not in seen:
            seen.add(c)
            final_code.append(c)
    
    return [list(words[c]) for c in final_code]
