"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import sys
from collections import defaultdict

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy:
    1. Start with all weight-2 vectors (norm^2=2). There are C(11,2)*4 = 220.
       These form a valid kissing configuration since min pairwise distance >= max norm.
       For two weight-2 vectors, distance^2 = 4 - 2*dot. If they share no indices, dot=0, dist^2=4.
       If they share one index, dot=±1, dist^2=2. If they share both (same pair), dot=±2.
       Same pair with same signs: dot=2, dist=0 (not allowed, but we don't include duplicates).
       Same pair, opposite signs: dot=-2, dist^2=8.
       Different pairs sharing one index: dot=±1, dist^2=2 or 4-2*(±1).
       Max norm^2 = 2, so max norm = sqrt(2). Min distance must be >= sqrt(2), i.e., dist^2 >= 2.
       All distinct weight-2 vectors have dist^2 >= 2. So all 220 are valid.
    
    2. Try to add weight-4 vectors (norm^2=4, norm=2). For a weight-4 vector v and weight-2 vector u,
       dist^2 = ||v||^2 + ||u||^2 - 2*v·u = 4 + 2 - 2*v·u = 6 - 2*v·u.
       We need dist^2 >= max_norm^2. If we include weight-4 vectors, max_norm^2 becomes 4.
       So we need dist^2 >= 4 for all pairs.
       For weight-4 v and weight-2 u: 6 - 2*v·u >= 4 => v·u <= 1.
       For two weight-4 vectors v, w: ||v-w||^2 = 4+4-2*v·w = 8-2*v·w >= 4 => v·w <= 2.
       For two weight-2 vectors u, v: dist^2 >= 2 as before, but now we need >= 4? 
       Wait, the constraint is min pairwise distance >= max norm. If max norm^2 = 4, then max norm = 2.
       So min pairwise distance must be >= 2, i.e., dist^2 >= 4.
       
       But weight-2 vectors have dist^2 >= 2, not necessarily >= 4. So if we add any weight-4 vector,
       we must also ensure all weight-2 pairs have dist^2 >= 4, which means v·u <= 0 for all weight-2 pairs.
       That's very restrictive. Let's check: two weight-2 vectors sharing one index have dot = ±1.
       If dot = 1, dist^2 = 4-2 = 2 < 4. So we can't have both if max norm^2 = 4.
       
       Alternative: keep max norm^2 = 2 (only weight-2 vectors) and try to find a larger set?
       No, there are only 220 weight-2 vectors.
       
       Better approach: Use weight-3 vectors? norm^2 = 3.
       Or mix: use vectors of norm^2 = 2 and norm^2 = 4, but then max norm^2 = 4.
       
       Let me reconsider. The seed gives 220. Let's try to exceed it.
       
       Known kissing number in 11D is at least 556 (from known bounds). Let's aim higher.
       
       Strategy: Use a greedy approach with vectors of various weights.
       Actually, let's use the fact that the Leech lattice or similar constructions exist.
       
       Simpler: Start with all weight-2 (220 vectors). Then try to add weight-4 vectors that are
       compatible. But as noted, this requires removing many weight-2 vectors.
       
       Alternative: Use only vectors with norm^2 = 4 (weight 4). 
       Number of weight-4 vectors: C(11,4) * 4^4 = 330 * 256 = 84480.
       Constraint: for any two, v·w <= 2 (since dist^2 = 8-2*v·w >= 4 => v·w <= 2).
       
       Or use norm^2 = 3 (weight 3): C(11,3)*4^3 = 165*64 = 10560.
       Constraint: v·w <= 1 (dist^2 = 6-2*v·w >= 3 => v·w <= 1.5, so v·w <= 1).
       
       Let's try a greedy maximum independent set approach on a graph where edges connect
       vectors that violate the constraint.
       
       Given time constraints, let's do:
       1. Generate all weight-2 vectors (220) - always valid together.
       2. Try to add weight-4 vectors greedily, checking compatibility with existing set.
          When adding weight-4 vectors, max_norm^2 becomes 4, so we need all pairs to have dist^2 >= 4.
          This means we may need to remove some weight-2 vectors.
       
       Actually, let's try a different approach: use vectors of norm^2 = 4 only, and find a large
       set with pairwise dot product <= 2.
    """
    start_time = time.time()
    deadline = 140.0  # seconds, with 10s safety margin
    
    def elapsed():
        return time.time() - start_time
    
    # Generate weight-2 vectors
    weight2_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                weight2_vectors.append(tuple(v))
    
    # All weight-2 vectors together: min dist^2 = 2, max norm^2 = 2. Valid.
    best_set = list(weight2_vectors)
    best_score = len(best_set)
    
    # Try to build a larger set using weight-4 vectors with max_norm^2 = 4
    # Generate weight-4 vectors
    if elapsed() < deadline - 60:
        weight4_vectors = []
        for indices in itertools.combinations(range(D), 4):
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for idx, s in zip(indices, signs):
                    v[idx] = s
                weight4_vectors.append(tuple(v))
        
        # Greedy: start with all weight-2, try adding weight-4 one by one
        # But need to check: if we add any weight-4, max_norm^2 = 4
        # So all pairs must have dist^2 >= 4
        # For weight-2 pair: need dist^2 >= 4, i.e., dot <= 0
        # For weight-2 and weight-4: need 6 - 2*dot >= 4, i.e., dot <= 1
        # For weight-4 pair: need 8 - 2*dot >= 4, i.e., dot <= 2
        
        # First, filter weight-2 vectors to those with pairwise dot <= 0
        # This is a maximum independent set problem on a graph where edges connect weight-2 with dot > 0
        # Let's do greedy: start with all, remove conflicts
        w2_arr = np.array(weight2_vectors, dtype=np.int8)
        # Compute dot products
        dots_w2 = w2_arr @ w2_arr.T
        # Conflict if dot > 0
        conflict_w2 = (dots_w2 > 0) & (np.arange(D) != np.arange(D)[:, None])
        
        # Greedy independent set
        n_w2 = len(weight2_vectors)
        in_set = [True] * n_w2
        # Sort by degree (number of conflicts) ascending, pick low-degree first
        degrees = [conflict_w2[i].sum() for i in range(n_w2)]
        order = sorted(range(n_w2), key=lambda i: degrees[i])
        
        selected_w2 = []
        for i in order:
            if not in_set[i]:
                continue
            # Check if i conflicts with any already selected
            conflict = False
            for j in selected_w2:
                if dots_w2[i, j] > 0:
                    conflict = True
                    break
            if not conflict:
                selected_w2.append(i)
        
        # Now try to add weight-4 vectors
        w4_arr = np.array(weight4_vectors, dtype=np.int8)
        selected_w2_arr = w2_arr[selected_w2]
        
        # Check compatibility of weight-4 with selected weight-2
        # dot_w2_w4[i, j] = selected_w2_arr[i] · w4_arr[j]
        # Need dot <= 1 for all
        
        # This is expensive. Let's be smarter: process weight-4 in batches
        # For each weight-4 vector, check against selected_w2 and against already selected weight-4
        
        selected_w4 = []
        w4_dots_with_w2 = selected_w2_arr @ w4_arr.T  # shape (len(selected_w2), len(weight4_vectors))
        
        # For efficiency, process in chunks
        chunk_size = 10000
        n_w4 = len(weight4_vectors)
        
        # Precompute: for each w4, max dot with selected_w2
        max_dot_w2 = w4_dots_with_w2.max(axis=0) if len(selected_w2) > 0 else np.zeros(n_w4, dtype=np.int8)
        
        # Only consider w4 with max_dot_w2 <= 1
        candidate_w4_indices = np.where(max_dot_w2 <= 1)[0]
        
        # Now among candidates, find large set with pairwise dot <= 2
        # Greedy
        cand_arr = w4_arr[candidate_w4_indices]
        n_cand = len(cand_arr)
        
        if n_cand > 0 and elapsed() < deadline - 30:
            # Compute pairwise dots for candidates - this is O(n_cand^2 * D)
            # n_cand could be large. Let's limit.
            if n_cand > 50000:
                # Take a random subset
                np.random.seed(42)
                idx = np.random.choice(n_cand, 50000, replace=False)
                cand_arr = cand_arr[idx]
                n_cand = len(cand_arr)
            
            dots_cand = cand_arr @ cand_arr.T
            conflict_cand = (dots_cand > 2) & (np.arange(n_cand) != np.arange(n_cand)[:, None])
            
            # Greedy independent set
            degrees_cand = [conflict_cand[i].sum() for i in range(n_cand)]
            order_cand = sorted(range(n_cand), key=lambda i: degrees_cand[i])
            
            selected_cand = []
            for i in order_cand:
                if elapsed() > deadline - 15:
                    break
                conflict = False
                for j in selected_cand:
                    if dots_cand[i, j] > 2:
                        conflict = True
                        break
                if not conflict:
                    selected_cand.append(i)
            
            # Build final set
            final_set = [selected_w2_arr[i] for i in selected_w2]
            for i in selected_cand:
                final_set.append(cand_arr[i])
            
            if len(final_set) > best_score:
                best_set = final_set
                best_score = len(final_set)
    
    # Also try: only weight-2, which is always valid
    if len(weight2_vectors) > best_score:
        best_set = list(weight2_vectors)
        best_score = len(best_set)
    
    # Convert to array
    result = np.array(best_set, dtype=np.int64)
    
    # Verify: min pairwise distance >= max norm
    if len(result) > 0:
        norms_sq = np.sum(result ** 2, axis=1)
        max_norm_sq = norms_sq.max()
        
        # Check pairwise distances (sample if too large)
        n = len(result)
        if n <= 2000:
            dists_sq = np.sum(result[:, None, :] ** 2, axis=2) + np.sum(result[None, :, :] ** 2, axis=2) - 2 * (result @ result.T)
            np.fill_diagonal(dists_sq, 0)
            min_dist_sq = dists_sq.min()
            if min_dist_sq < max_norm_sq - 1e-9:
                # Fallback to weight-2 only
                result = np.array(weight2_vectors, dtype=np.int64)
        else:
            # Trust the construction
            pass
    
    return result

# EVOLVE-BLOCK-END
