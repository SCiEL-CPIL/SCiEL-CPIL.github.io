import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def has_k4(adj_masks, n):
        """Check for K4 using bitmasks."""
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj_masks[i] >> j) & 1:
                    continue
                common = adj_masks[i] & adj_masks[j]
                # Remove i and j from common
                common &= ~(1 << i)
                common &= ~(1 << j)
                # Check if any two vertices in common are adjacent
                while common:
                    c = common & (-common)
                    common ^= c
                    c_idx = c.bit_length() - 1
                    # Check if c_idx is adjacent to any other in common
                    rest = common  # common with c removed
                    if rest & adj_masks[c_idx]:
                        return True
        return False

    def has_ind6(adj_masks, n):
        """Check for independent set of size 6 (K6 in complement)."""
        full_mask = (1 << n) - 1
        for i in range(n):
            # Independent set containing i: need 5 more from Nbar(i) \ {i}
            non_neigh = full_mask & ~(1 << i) & ~adj_masks[i]
            # Need to find K5 in induced subgraph of complement on non_neigh
            # That is, find 5 vertices in non_neigh that are pairwise non-adjacent in G
            # Equivalently, find 5 vertices in non_neigh that form a clique in complement
            # Let's check all 5-subsets... too slow. Use recursive/branch and bound.
            
            # Better: check if the complement of G[non_neigh] has a K5
            # Complement adjacency among non_neigh: u,v non-adjacent in G means adjacent in complement
            # We need 5 vertices in non_neigh pairwise non-adjacent in G
            
            # Use a simpler check: for each pair (u,v) in non_neigh that are non-adjacent in G,
            # check if they share 3 common non-neighbors (in G) that are also non-adjacent to each other
            # This is getting complex. Let's use a direct but slightly optimized approach.
            
            # For each 5-combination check is too slow for n=35.
            # Use: independent set of 6 = set S of 6 vertices with no edges among them.
            # For each edge (i,j) in complement (i.e., non-edge in G), check common complement neighbors.
            
            # Let's do: for each pair (u,v) where u,v are non-adjacent in G (i.e., edge in complement),
            # find common non-neighbors in G (excluding u,v), check if any two of those are also non-adjacent in G
            # and share 2 more common non-neighbors that are non-adjacent...
            
            # Actually, let's just use the same approach as the original but cleaner:
            pass

        # Direct check for independent 6-set
        # For each 6-combination is O(n^6) which is too slow for n=35 (35^6 ≈ 1.8e9)
        # Need a smarter approach.
        
        # Check: for each pair (i,j) non-adjacent in G, common non-neighbors
        # If any two of those common non-neighbors are non-adjacent, and they share 2 more common non-neighbors
        # that are non-adjacent... this is the same as checking K4 in complement for the remaining.
        
        # Let's use: independent set of 6 exists iff there exist 6 vertices with no edges.
        # Equivalently, the complement graph has a K6.
        # Check for K6 in complement: for each pair (i,j) adjacent in complement (non-edge in G),
        # find common neighbors in complement (common non-neighbors in G), check if any 4 of those form K4 in complement.
        
        # For each non-edge (i,j) in G:
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    continue  # i,j are adjacent in G, not an edge in complement
                # i,j form an edge in complement. Common neighbors in complement:
                # vertices k that are non-adjacent to both i and j in G
                common_non = full_mask & ~adj_masks[i] & ~adj_masks[j]
                common_non &= ~(1 << i)
                common_non &= ~(1 << j)
                
                # Need to find K4 in complement among common_non
                # For each pair (a,b) in common_non that are non-adjacent in G:
                common_list = []
                temp = common_non
                while temp:
                    c = temp & (-temp)
                    temp ^= c
                    common_list.append(c.bit_length() - 1)
                
                # Check for K4 in complement among common_list
                # For each pair (a,b) in common_list non-adjacent in G, check common non-neighbors
                for ai in range(len(common_list)):
                    for bi in range(ai + 1, len(common_list)):
                        a, b = common_list[ai], common_list[bi]
                        if (adj_masks[a] >> b) & 1:
                            continue  # a,b adjacent in G, not edge in complement
                        # a,b non-adjacent in G. Common non-neighbors of a and b in G (excluding i,j,a,b):
                        common_ab_non = full_mask & ~adj_masks[a] & ~adj_masks[b]
                        common_ab_non &= ~(1 << a)
                        common_ab_non &= ~(1 << b)
                        common_ab_non &= ~(1 << i)
                        common_ab_non &= ~(1 << j)
                        
                        # Need 2 more vertices in common_ab_non that are non-adjacent in G
                        common_ab_list = []
                        temp2 = common_ab_non
                        while temp2:
                            c = temp2 & (-temp2)
                            temp2 ^= c
                            common_ab_list.append(c.bit_length() - 1)
                        
                        for ci in range(len(common_ab_list)):
                            for di in range(ci + 1, len(common_ab_list)):
                                c, d = common_ab_list[ci], common_ab_list[di]
                                if (adj_masks[c] >> d) & 1:
                                    continue
                                # Found independent set of 6: {i, j, a, b, c, d}
                                return True
        return False

    def is_valid(adj_masks, n):
        if has_k4(adj_masks, n):
            return False
        if has_ind6(adj_masks, n):
            return False
        return True

    def edges_from_masks(n, adj_masks):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def random_adj(n, p=0.4):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def spectral_adj(n):
        """Generate adjacency from random symmetric matrix eigenvectors."""
        # Generate random symmetric matrix
        A = np.random.randn(n, n)
        A = (A + A.T) / 2
        # Compute eigendecomposition
        try:
            vals, vecs = np.linalg.eigh(A)
        except np.linalg.LinAlgError:
            return None
        # Use top eigenvectors to define structure
        # Take the largest eigenvalue's eigenvector and use its signs
        # Actually, let's use a different approach: use the matrix itself
        # Binarize A: entry is 1 if A[i,j] > threshold
        # Use median as threshold
        threshold = np.median(A[np.triu_indices(n, k=1)])
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] > threshold:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def local_search(adj, n, max_steps=5000):
        """Improve graph via local search."""
        if not is_valid(adj, n):
            return None
        best_adj = list(adj)
        best_score = 0  # We just want to maintain validity
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            # Toggle edge (u,v)
            old = (adj[u] >> v) & 1
            if old:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            if is_valid(adj, n):
                best_adj = list(adj)
            else:
                # Revert
                if old:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        return best_adj

    # Start with known good structure and try to extend
    best_n = 1
    best_adj = [0]
    
    # Try building up from various starting points
    for target_n in range(2, 36):
        if time.time() > deadline - 15.0:
            break
        
        found = False
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts:
            if time.time() > deadline - 8.0:
                break
            attempts += 1
            
            # Different initialization strategies
            strategy = random.random()
            if strategy < 0.3:
                # Random graph
                adj = random_adj(target_n, p=0.35 + 0.2 * random.random())
            elif strategy < 0.6:
                # Spectral construction
                adj = spectral_adj(target_n)
                if adj is None:
                    continue
            else:
                # Start from best known and add vertex
                if len(best_adj) >= target_n - 1:
                    adj = list(best_adj[:target_n - 1]) + [0]
                    # Connect new vertex to random subset
                    new_v = target_n - 1
                    for i in range(target_n - 1):
                        if random.random() < 0.3:
                            adj[i] |= (1 << new_v)
                            adj[new_v] |= (1 << i)
                else:
                    adj = random_adj(target_n, p=0.4)
            
            # Check validity
            if is_valid(adj, target_n):
                found = True
                best_n = target_n
                best_adj = adj
                break
            else:
                # Try local search repair
                repaired = local_search(adj, target_n, max_steps=200)
                if repaired is not None:
                    found = True
                    best_n = target_n
                    best_adj = repaired
                    break
        
        if not found:
            # Try harder with more attempts if we're making progress
            if target_n <= best_n + 1:
                continue
            else:
                break

    # Final verification and edge extraction
    if not is_valid(best_adj, best_n):
        # Fallback: find a smaller valid graph
        for fallback_n in range(best_n - 1, 0, -1):
            if time.time() > deadline - 2.0:
                break
            adj = random_adj(fallback_n, p=0.4)
            if is_valid(adj, fallback_n):
                best_n = fallback_n
                best_adj = adj
                break
    
    edges = edges_from_masks(best_n, best_adj)
    
    # Final safety check
    if not is_valid(best_adj, best_n):
        # Return a trivially valid graph
        best_n = 1
        best_adj = [0]
        edges = []
    
    return {'n': best_n, 'edges': edges}
