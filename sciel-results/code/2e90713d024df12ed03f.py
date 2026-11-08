"""Direct construction using the D11 lattice with greedy independent set over larger candidate pool."""
import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # safety margin from 150s limit

    def generate_candidates(max_weight=4):
        """Generate all nonzero integer vectors with entries in {-1, 0, 1} up to given weight."""
        candidates = []
        # Weight 2
        for i, j in combinations(range(D), 2):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
        # Weight 3
        if max_weight >= 3:
            for i, j, k in combinations(range(D), 3):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        for sk in [1, -1]:
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
        # Weight 4
        if max_weight >= 4:
            for i, j, k, l in combinations(range(D), 4):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        for sk in [1, -1]:
                            for sl in [1, -1]:
                                v = [0] * D
                                v[i] = si
                                v[j] = sj
                                v[k] = sk
                                v[l] = sl
                                candidates.append(v)
        return np.array(candidates, dtype=np.int64)

    def min_degree_greedy(candidates, threshold):
        """
        Find a large independent set in the conflict graph.
        Two candidates conflict if their squared distance < threshold.
        Uses minimum-degree greedy heuristic.
        """
        n = len(candidates)
        if n == 0:
            return []
        
        # Compute squared distances
        # dist2[i,j] = ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
        norms2 = np.sum(candidates * candidates, axis=1)
        dot = candidates @ candidates.T
        dist2 = norms2[:, None] + norms2[None, :] - 2 * dot
        
        # Conflict: dist2 < threshold
        conflict = (dist2 < threshold).astype(np.int8)
        np.fill_diagonal(conflict, 0)
        
        # Minimum degree greedy
        in_set = set()
        remaining = set(range(n))
        
        while remaining:
            # Find vertex with minimum degree in remaining subgraph
            best = -1
            best_deg = float('inf')
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and conflict[v, u] == 1)
                if deg < best_deg:
                    best_deg = deg
                    best = v
                if time.time() > deadline:
                    break
            if best == -1 or time.time() > deadline:
                break
            
            in_set.add(best)
            remaining.discard(best)
            # Remove all neighbors of best from remaining
            to_remove = set()
            for u in remaining:
                if conflict[best, u] == 1:
                    to_remove.add(u)
            remaining -= to_remove
        
        return list(in_set)

    # Strategy 1: All weight-2 vectors (baseline, 220)
    candidates_w2 = generate_candidates(max_weight=2)
    baseline = np.array(candidates_w2, dtype=np.int64)
    best_result = baseline
    best_k = len(baseline)
    
    if time.time() > deadline - 5:
        return best_result
    
    # Strategy 2: Greedy with weight 2+3, threshold = 3 (all pairwise dist^2 >= 3)
    candidates_w23 = generate_candidates(max_weight=3)
    if len(candidates_w23) > 0:
        selected_idx = min_degree_greedy(candidates_w23, threshold=3)
        if len(selected_idx) > best_k:
            best_result = candidates_w23[selected_idx]
            best_k = len(selected_idx)
    
    if time.time() > deadline - 5:
        return best_result
    
    # Strategy 3: Greedy with weight 2+3+4, threshold = 4 (all pairwise dist^2 >= 4)
    candidates_w234 = generate_candidates(max_weight=4)
    if len(candidates_w234) > 0:
        selected_idx = min_degree_greedy(candidates_w234, threshold=4)
        if len(selected_idx) > best_k:
            best_result = candidates_w234[selected_idx]
            best_k = len(selected_idx)
    
    if time.time() > deadline - 5:
        return best_result
    
    # Strategy 4: Greedy with weight 2+3, threshold = 2 (all pairwise dist^2 >= 2)
    # This allows more vectors but the max norm might be 3, so we need dist^2 >= 3
    # Actually if we include weight-3 vectors, max norm^2 = 3, so threshold must be 3.
    # If only weight-2, threshold = 2, and we get all 220.
    
    # Strategy 5: Try threshold=3 but only weight-2 candidates (should give subset of 220)
    # Not useful, skip.
    
    # Strategy 6: Greedy with weight 2+3+4, threshold = 3
    if time.time() < deadline - 10:
        selected_idx = min_degree_greedy(candidates_w234, threshold=3)
        if len(selected_idx) > best_k:
            best_result = candidates_w234[selected_idx]
            best_k = len(selected_idx)
    
    return best_result
# EVOLVE-BLOCK-END
