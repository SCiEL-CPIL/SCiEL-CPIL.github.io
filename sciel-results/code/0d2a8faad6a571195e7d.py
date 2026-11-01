"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Generate all candidate vectors with norm^2 = 2 (weight 2: two entries ±1)
    # These are the "short" vectors that give the densest kissing configurations
    candidates_list = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    
    candidates = np.array(candidates_list, dtype=np.int64)
    N = len(candidates)
    
    # For norm^2 = 2 vectors, the condition is:
    # dist^2 >= max_norm^2
    # dist^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i·v_j = 2 + 2 - 2*v_i·v_j = 4 - 2*v_i·v_j
    # We need: 4 - 2*v_i·v_j >= 2 (since max_norm^2 = 2 for all these vectors)
    # So: 2*v_i·v_j <= 2, i.e., v_i·v_j <= 1
    # Conflict when v_i·v_j > 1, i.e., v_i·v_j >= 2
    # For weight-2 vectors, dot product is in {-2, -1, 0, 1, 2}
    # Conflict when dot = 2 (same vector, excluded) or... wait, dot=2 means same direction
    # Actually for distinct vectors, dot can be -2, -1, 0, 1, 2
    # dot=2 means they are the same vector (since both have norm^2=2)
    # So for distinct vectors, max dot is 1, meaning no conflicts?
    
    # Wait, let me reconsider. Two distinct weight-2 vectors:
    # v = (1,1,0,...), w = (1,1,0,...) -> same, excluded
    # v = (1,1,0,...), w = (1,-1,0,...) -> dot = 0
    # v = (1,1,0,...), w = (-1,1,0,...) -> dot = 0
    # v = (1,1,0,...), w = (1,0,1,...) -> dot = 1
    # v = (1,1,0,...), w = (0,1,1,...) -> dot = 1
    # v = (1,1,0,...), w = (1,0,0,...) -> not weight 2
    # 
    # So for distinct weight-2 vectors, dot product is in {-2, -1, 0, 1}
    # dot = -2 means v = -w, which is possible! e.g., (1,1,0,...) and (-1,-1,0,...)
    # 
    # dist^2 = 4 - 2*dot. If dot = -2, dist^2 = 4-(-4) = 8 >= 2 ✓
    # If dot = -1, dist^2 = 4-(-2) = 6 >= 2 ✓
    # If dot = 0, dist^2 = 4 >= 2 ✓
    # If dot = 1, dist^2 = 4-2 = 2 >= 2 ✓
    #
    # So ALL distinct weight-2 vectors are valid! The maximum independent set is all of them.
    # Number of weight-2 vectors: C(11,2) * 4 = 55 * 4 = 220
    
    # But wait - we can also mix in other vectors. Let's think about what gives the best score.
    # The score is k (number of vectors). So we want the maximum number of vectors.
    
    # Let's also consider weight-4 vectors (norm^2 = 4):
    # For mixed sets, max_norm^2 = 4 (if any weight-4 vector is present)
    # Then for two weight-2 vectors: dist^2 = 4 - 2*dot >= 4 required
    # 4 - 2*dot >= 4 => dot <= 0
    # So weight-2 vectors with dot > 0 conflict when weight-4 vectors are present.
    
    # For two weight-4 vectors: dist^2 = 8 - 2*dot >= 4 => dot <= 2
    # For weight-2 and weight-4: dist^2 = 2 + 4 - 2*dot = 6 - 2*dot >= 4 => dot <= 1
    
    # Strategy: Try pure weight-2 set (220 vectors) vs mixed sets.
    
    # Pure weight-2: all 220 vectors work. Score = 220.
    
    # Let's verify this and return it immediately since it's a known valid solution.
    # But can we do better by adding weight-4 vectors?
    
    # If we add even one weight-4 vector, max_norm^2 becomes 4.
    # Then weight-2 vectors must have dot <= 0 with each other.
    # This is a much more restrictive condition.
    
    # Let's estimate: for weight-2 vectors, how many have dot <= 0 with a given weight-2 vector?
    # v = (1,1,0,...,0) (positions 0,1)
    # w has dot with v:
    # - If w uses positions {0,1}: dot in {-2,-1,0,1,2} but w≠v so dot in {-2,-1,0,1}
    #   dot <= 0: {-2,-1,0} -> 3 out of 4 (excluding v itself which has dot=2)
    #   Wait, w can be (1,1), (1,-1), (-1,1), (-1,-1) at positions {0,1}
    #   dot with (1,1): 2, 0, 0, -2. So dot<=0: 3 vectors.
    # - If w uses one of {0,1} and one other position, say {0,2}:
    #   w = (±1, 0, ±1, 0,...). dot with (1,1,0,...) = ±1. So dot in {-1, 1}.
    #   dot <= 0: 2 out of 4.
    # - If w uses neither 0 nor 1: dot = 0. All 4 work.
    #
    # This is complex. Let's just try to build a large independent set in the conflict graph.
    
    # Build conflict graph for all candidates (weight 2 and weight 4)
    # But weight-4 candidates: C(11,4) * 16 = 330 * 16 = 5280. Total ~5500.
    # That's manageable.
    
    # Actually, let's first just return the pure weight-2 solution as a baseline,
    # then try to improve with a greedy approach on the mixed set.
    
    # Pure weight-2 solution:
    pure_w2 = candidates.copy()
    best_result = pure_w2
    best_size = N  # 220
    
    # Now try to build a larger set using weight-2 and weight-4 vectors
    # Generate weight-4 candidates
    w4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            w4_list.append(v)
    
    w4_candidates = np.array(w4_list, dtype=np.int64)
    N4 = len(w4_candidates)
    
    # Combined candidate set
    all_cands = np.vstack([candidates, w4_candidates])
    N_all = len(all_cands)
    
    # Norm squared for each
    norms_sq = np.sum(all_cands ** 2, axis=1)
    # First N have norm^2=2, rest have norm^2=4
    
    # Compute dot products
    dots = all_cands @ all_cands.T
    
    # Conflict: dist^2 < max_norm^2
    # max_norm^2 = 4 if any weight-4 is in the set, else 2
    # This makes it a non-standard problem. Let's handle it by trying R2=4 (conservative)
    # and R2=2 (pure weight-2).
    
    # For R2 = 4: conflict if dist^2 < 4
    # dist^2 = n_i^2 + n_j^2 - 2*dot_ij < 4
    # 2*dot_ij > n_i^2 + n_j^2 - 4
    threshold = (norms_sq[:, None] + norms_sq[None, :] - 4) / 2.0
    conflicts = dots > threshold
    np.fill_diagonal(conflicts, False)
    
    # Now find maximum independent set using greedy + local search
    rng = np.random.default_rng(42)
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N_all)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    def greedy_min_degree():
        in_set = np.zeros(N_all, dtype=bool)
        alive = np.ones(N_all, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 10:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Update degrees
                for r in rem:
                    deg_change = np.sum(conflicts[r, to_remove])
                    degrees[r] -= deg_change
        return in_set
    
    def local_search(in_set_init, max_iters=5000):
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        # Count conflicts for each vertex
        count_conflicts = np.zeros(N_all, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            
            # Gain from adding vertex v: 1 - (number of in_set neighbors of v)
            gains = 1 - count_conflicts[not_in_set]
            max_gain = np.max(gains)
            
            if max_gain > 0:
                best_idx = np.argmax(gains)
                best_v = not_in_set[best_idx]
                
                # Add best_v, remove its in_set neighbors
                to_remove = np.where(conflicts[best_v] & in_set)[0]
                
                # Update count_conflicts for removed vertices
                for u in to_remove:
                    count_conflicts[neighbors[u]] -= 1
                
                in_set[to_remove] = False
                in_set[best_v] = True
                current_size += max_gain
                
                # Update count_conflicts for best_v's neighbors
                count_conflicts[neighbors[best_v]] += 1
                improved = True
        
        return in_set, current_size
    
    # Multiple restarts
    best_size_r2 = 0
    best_in_set_r2 = np.zeros(N_all, dtype=bool)
    
    n_restarts = 0
    max_restarts = 20
    
    while time.time() < deadline - 15 and n_restarts < max_restarts:
        n_restarts += 1
        in_set = greedy_min_degree()
        current_size = int(in_set.sum())
        
        if current_size > best_size_r2:
            best_size_r2 = current_size
            best_in_set_r2 = in_set.copy()
        
        ls_result, ls_size = local_search(in_set)
        if ls_size > best_size_r2:
            best_size_r2 = ls_size
            best_in_set_r2 = ls_result.copy()
        
        # Early termination if we already beat the pure weight-2 solution significantly
        if best_size_r2 > best_size + 50:
            break
    
    # Compare with pure weight-2
    if best_size_r2 > best_size:
        final_indices = np.where(best_in_set_r2)[0]
        best_result = all_cands[final_indices]
        best_size = best_size_r2
    
    return best_result

# EVOLVE-BLOCK-END
