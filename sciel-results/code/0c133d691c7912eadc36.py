"""Rewritten program for kissing number in dimension 11 using angular sectoring."""
import numpy as np
import time
import itertools

D = 11
MAX_TIME = 130.0  # Safety margin before 150s kill

def construct():
    """Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm. Returns integer array of shape (k, d)."""
    start_time = time.time()
    
    def check_valid(vectors):
        """Check if all pairwise distances >= max norm."""
        if len(vectors) == 0:
            return True
        norms = np.array([np.sqrt(np.sum(v**2)) for v in vectors])
        max_norm = np.max(norms)
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                d = np.sqrt(np.sum((vectors[i] - vectors[j])**2))
                if d < max_norm - 1e-9:
                    return False
        return True
    
    # Angular sectoring approach:
    # Partition the sphere into conical regions based on sign patterns of first 2 coords.
    # For each sector, select the unique maximal weight-4 vector that fits.
    # This avoids constructing the full conflict graph.
    
    # Strategy: Use a hybrid construction combining:
    # 1. Weight-2 vectors (440 total) as the base
    # 2. Add weight-4 vectors in sectors where they don't conflict
    
    # First, generate all weight-2 vectors
    weight2_vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    weight2_vectors.append(v)
    
    # Generate weight-4 vectors
    weight4_vectors = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(combo, signs):
                v[idx] = s
            weight4_vectors.append(v)
    
    # Normalize all vectors for angular comparison
    def normalize(v):
        n = np.sqrt(np.sum(v**2))
        return v / n if n > 0 else v
    
    # Angular sectoring: use sign pattern of first 2 coordinates as sector key
    # Each sector is defined by (sign(v[0]), sign(v[1])) where sign is -1, 0, or 1
    
    def sector_key(v):
        s0 = np.sign(v[0])
        s1 = np.sign(v[1])
        return (s0, s1)
    
    # For each sector, we want to select vectors that are "maximal" in some sense
    # and don't conflict with each other
    
    # Approach: Use a greedy selection within each sector, prioritizing vectors
    # with higher norms (weight-4 over weight-2) to maximize density
    
    # First, let's try a pure weight-4 construction
    # Two weight-4 vectors v, w with same norm 2:
    # ||v-w|| >= 2 means ||v-w||^2 >= 4
    # ||v||^2 + ||w||^2 - 2*v·w >= 4
    # 4 + 4 - 2*v·w >= 4
    # v·w <= 2
    
    # For weight-4 vectors, v·w is the sum of products of overlapping coordinates
    # If they share k coordinates with aligned signs, contribution is k
    # If they share k coordinates with opposed signs, contribution is -k
    # So v·w can range from -4 to 4
    
    # We need v·w <= 2 for all pairs
    
    # Let's use a greedy approach with angular sectoring
    # Partition by sign pattern of first 4 coordinates to create finer sectors
    
    def fine_sector_key(v):
        """Create a fine sector key based on sign patterns of first 4 coords."""
        signs = []
        for i in range(4):
            signs.append(np.sign(v[i]))
        return tuple(signs)
    
    # Group weight-4 vectors by their fine sector
    sectors = {}
    for v in weight4_vectors:
        key = fine_sector_key(v)
        if key not in sectors:
            sectors[key] = []
        sectors[key].append(v)
    
    # For each sector, greedily select non-conflicting vectors
    selected_w4 = []
    for key, vecs in sectors.items():
        sector_selected = []
        for v in vecs:
            if time.time() - start_time > MAX_TIME - 10:
                break
            # Check if v conflicts with any already selected in this sector
            conflict = False
            for u in sector_selected:
                dot = np.dot(v, u)
                if dot > 2:  # v·w > 2 means ||v-w|| < 2
                    conflict = True
                    break
            if not conflict:
                sector_selected.append(v)
        selected_w4.extend(sector_selected)
        if time.time() - start_time > MAX_TIME - 10:
            break
    
    # Now try to add weight-2 vectors that don't conflict with weight-4 vectors
    # Weight-2 has norm sqrt(2), weight-4 has norm 2
    # For v (w4, norm 2) and w (w2, norm sqrt(2)):
    # ||v-w|| >= max(2, sqrt(2)) = 2
    # ||v-w||^2 >= 4
    # 4 + 2 - 2*v·w >= 4
    # v·w <= 1
    
    # So we need v·w <= 1 for all pairs of weight-4 and weight-2 vectors
    
    selected_all = list(selected_w4)
    
    for w2 in weight2_vectors:
        if time.time() - start_time > MAX_TIME - 5:
            break
        conflict = False
        for v in selected_w4:
            dot = np.dot(v, w2)
            if dot > 1:  # v·w > 1 means conflict
                conflict = True
                break
        if not conflict:
            selected_all.append(w2)
    
    # Also add weight-2 vectors among themselves
    # Two weight-2 vectors v, w with norm sqrt(2):
    # ||v-w|| >= sqrt(2)
    # ||v-w||^2 >= 2
    # 2 + 2 - 2*v·w >= 2
    # v·w <= 1
    
    # But we already checked w2 against w4, now check w2 against w2
    # Actually, we need to check all pairs in selected_all
    
    # Let's do a final validation and remove any conflicts
    final_vectors = []
    for v in selected_all:
        if time.time() - start_time > MAX_TIME - 3:
            break
        conflict = False
        for u in final_vectors:
            dot = np.dot(v, u)
            norm_v = np.sqrt(np.sum(v**2))
            norm_u = np.sqrt(np.sum(u**2))
            max_norm = max(norm_v, norm_u)
            dist_sq = np.sum((v - u)**2)
            if dist_sq < max_norm**2 - 1e-9:
                conflict = True
                break
        if not conflict:
            final_vectors.append(v)
    
    if len(final_vectors) == 0:
        # Fallback to seed
        return np.array(weight2_vectors, dtype=np.int64)
    
    # Final verification
    if not check_valid(final_vectors):
        # Fall back to just weight-2 vectors
        return np.array(weight2_vectors, dtype=np.int64)
    
    return np.array(final_vectors, dtype=np.int64)
