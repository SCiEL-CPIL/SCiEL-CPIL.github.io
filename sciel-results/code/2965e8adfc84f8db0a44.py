import numpy as np
import random
import itertools

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters (not expected)
        return []

    total_vertices = q ** n
    vertices = list(itertools.product(range(q), repeat=n))
    
    # Precompute distances
    # dist[i][j] is Hamming distance between vertices[i] and vertices[j]
    # Using numpy for speed
    verts_np = np.array(vertices)
    # Compute pairwise distances
    # dist = np.sum(verts_np[:, np.newaxis, :] != verts_np[np.newaxis, :, :], axis=2)
    # This creates a 729x729x6 array, then sums. 729*729*6 ~ 3M ops. Fast.
    
    # However, we only need to know if dist >= 2 for the complement graph MIS.
    # And dist <= 1 for the domination check.
    
    # Let's compute the distance matrix once.
    # To save memory/time, we can compute it in chunks or just do it directly.
    # 729 is small.
    
    dist_matrix = np.zeros((total_vertices, total_vertices), dtype=np.int8)
    for i in range(total_vertices):
        for j in range(i + 1, total_vertices):
            d = np.sum(verts_np[i] != verts_np[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
            
    # Precompute balls (self + dist 1) for domination checks
    # ball_mask[i] is a boolean array of length N, True if dist(i, j) <= 1
    ball_masks = (dist_matrix <= 1)
    
    def is_dominating(set_indices):
        if not set_indices:
            return total_vertices == 0
        covered = np.zeros(total_vertices, dtype=bool)
        for idx in set_indices:
            covered |= ball_masks[idx]
        return covered.all()

    def greedy_mis_complement(random_seed=None):
        """
        Finds a Maximal Independent Set in the complement of the distance-1 graph.
        Complement edge (i,j) exists if dist(i,j) >= 2.
        Independent Set in Complement: No two vertices have dist >= 2.
        So all pairs have dist <= 1.
        """
        if random_seed is not None:
            random.seed(random_seed)
            
        candidates = list(range(total_vertices))
        mis = []
        
        while candidates:
            # Pick a random candidate
            v = random.choice(candidates)
            mis.append(v)
            
            # Filter candidates: keep only those u such that dist(v, u) <= 1
            # Because if dist(v, u) >= 2, they are adjacent in Complement, 
            # so u cannot be in the same Independent Set as v.
            # So we require dist(v, u) <= 1.
            
            # Using the precomputed dist_matrix
            valid_mask = dist_matrix[v] <= 1
            
            # Update candidates
            new_candidates = []
            for u in candidates:
                if valid_mask[u]:
                    new_candidates.append(u)
            candidates = new_candidates
            
        return mis

    def complete_dominating_set(initial_set):
        """
        Adds vertices to initial_set until it dominates the space.
        """
        S = list(initial_set)
        covered = np.zeros(total_vertices, dtype=bool)
        for idx in S:
            covered |= ball_masks[idx]
        
        undominated = np.where(~covered)[0]
        
        while len(undominated) > 0:
            # Pick a random undominated vertex to target
            target = random.choice(undominated.tolist())
            
            # Find the vertex in the whole space that covers the most undominated vertices
            # Optimization: Only check vertices that are in the ball of the target?
            # Or just check all? 729 is small.
            best_v = -1
            max_cov = -1
            
            # We can restrict candidates to those that cover 'target'
            # i.e., dist(v, target) <= 1
            candidates = np.where(ball_masks[target])[0]
            
            for v in candidates:
                # Count how many undominated vertices v covers
                # cov_count = np.sum(ball_masks[v] & ~covered)
                # This is fast with numpy
                cov_count = np.sum(ball_masks[v] & ~covered)
                if cov_count > max_cov:
                    max_cov = cov_count
                    best_v = v
            
            if best_v == -1:
                # Should not happen if graph is connected, but safety
                best_v = target
                
            S.append(best_v)
            covered |= ball_masks[best_v]
            undominated = np.where(~covered)[0]
            
        return S

    def minimize_by_deletion(dominating_set):
        """Remove redundant codewords."""
        S = set(dominating_set)
        
        # Count coverage
        counts = np.zeros(total_vertices, dtype=int)
        for c in S:
            counts += ball_masks[c]
            
        changed = True
        while changed:
            changed = False
            # Iterate in random order
            for c in random.sample(list(S), len(S)):
                # If all vertices covered by c are covered by at least 2 codewords, c is redundant
                # i.e., min count in ball(c) > 1
                min_count_in_ball = np.min(counts[ball_masks[c]])
                if min_count_in_ball > 1:
                    S.remove(c)
                    counts -= ball_masks[c]
                    changed = True
                    
        return list(S)

    best_size = float('inf')
    best_set = None
    
    # Run multiple trials
    # The MIS part is the "new" design element.
    for i in range(50):
        # 1. Get MIS of Complement
        mis_set = greedy_mis_complement(random_seed=i)
        
        # 2. Complete to Dominating Set
        ds = complete_dominating_set(mis_set)
        
        # 3. Minimize
        minimized = minimize_by_deletion(ds)
        
        if len(minimized) < best_size:
            best_size = len(minimized)
            best_set = minimized
            
        if best_size <= 73:
            break

    if best_set is None:
        # Fallback
        mis_set = greedy_mis_complement(0)
        ds = complete_dominating_set(mis_set)
        best_set = minimize_by_deletion(ds)

    return [vertices[idx] for idx in best_set]
