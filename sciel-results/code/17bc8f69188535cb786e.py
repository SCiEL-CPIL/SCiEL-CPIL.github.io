import numpy as np
import time
import random
import sys
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 600  # Safety margin for 630s limit

    def generate_all_vectors(max_norm_sq):
        """Generate all integer vectors in Z^11 with squared norm <= max_norm_sq."""
        vectors = set()
        
        # We'll generate vectors with at most 4 non-zero coordinates
        # Norm 1: e_i, -e_i
        for i in range(d):
            v = [0]*d
            v[i] = 1
            vectors.add(tuple(v))
            v[i] = -1
            vectors.add(tuple(v))
        
        # Norm 2: e_i +/- e_j
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = [0]*d
                        v[i] = s1
                        v[j] = s2
                        vectors.add(tuple(v))
        
        # Norm 3: e_i +/- e_j +/- e_k
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    for s1 in [1, -1]:
                        for s2 in [1, -1]:
                            for s3 in [1, -1]:
                                v = [0]*d
                                v[i] = s1
                                v[j] = s2
                                v[k] = s3
                                vectors.add(tuple(v))
        
        # Norm 4: 2e_i, -2e_i, and e_i +/- e_j +/- e_k +/- e_l
        for i in range(d):
            v = [0]*d
            v[i] = 2
            vectors.add(tuple(v))
            v[i] = -2
            vectors.add(tuple(v))
        
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    for l in range(k + 1, d):
                        for s1 in [1, -1]:
                            for s2 in [1, -1]:
                                for s3 in [1, -1]:
                                    for s4 in [1, -1]:
                                        v = [0]*d
                                        v[i] = s1
                                        v[j] = s2
                                        v[k] = s3
                                        v[l] = s4
                                        vectors.add(tuple(v))
        
        # Also include vectors with norm 4 from 2e_i + something? No, 2e_i has norm 4.
        # What about vectors like (2, 1, 0, ...) - norm 5, too large for R=4.
        
        # Filter by max_norm_sq
        filtered = []
        for v in vectors:
            norm_sq = sum(x*x for x in v)
            if norm_sq <= max_norm_sq:
                filtered.append(v)
        
        return np.array(filtered, dtype=np.int64)

    def solve_mis_branch_and_bound(adj_matrix, time_limit=300):
        """
        Maximum Independent Set via branch and bound.
        adj_matrix: boolean adjacency matrix (True = edge/conflict)
        Returns list of indices in the maximum independent set.
        """
        n = adj_matrix.shape[0]
        if n == 0:
            return []
        
        # Convert to list of sets for faster neighbor lookup
        neighbors = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j]:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
        
        # Degree-based ordering
        degrees = [len(neighbors[i]) for i in range(n)]
        order = list(range(n))
        order.sort(key=lambda x: -degrees[x])
        
        best_set = []
        best_size = 0
        
        # Upper bound: simple greedy coloring bound
        def greedy_color_bound(candidates):
            """Return upper bound on MIS size using greedy coloring."""
            if not candidates:
                return 0
            # Simple: use max degree + 1 as a loose bound, or count vertices
            # Better: greedy coloring
            color = {}
            colors_used = 0
            cand_list = list(candidates)
            cand_list.sort(key=lambda x: -degrees[x])
            for v in cand_list:
                if v in color:
                    continue
                # Find a color not used by neighbors
                neighbor_colors = set()
                for u in neighbors[v]:
                    if u in color:
                        neighbor_colors.add(color[u])
                c = 0
                while c in neighbor_colors:
                    c += 1
                color[v] = c
                colors_used = max(colors_used, c + 1)
            return colors_used
        
        # Branch and bound
        sys.setrecursionlimit(10000)
        
        def branch_and_bound(candidates, current_set, depth=0):
            nonlocal best_set, best_size
            
            if time.time() > start_time + time_limit:
                return
            
            if not candidates:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                return
            
            # Upper bound
            ub = len(candidates) + len(current_set)
            if ub <= best_size:
                return
            
            # Try to get a tighter bound
            ub_color = greedy_color_bound(candidates) + len(current_set)
            if ub_color <= best_size:
                return
            
            # Pick a vertex to branch on - choose one with max degree in candidate subgraph
            max_deg = -1
            chosen = None
            for v in candidates:
                deg_in_sub = len(neighbors[v] & set(candidates))
                if deg_in_sub > max_deg:
                    max_deg = deg_in_sub
                    chosen = v
            
            # Branch 1: Include chosen
            new_candidates = [v for v in candidates if v != chosen and v not in neighbors[chosen]]
            branch_and_bound(new_candidates, current_set + [chosen], depth + 1)
            
            # Branch 2: Exclude chosen
            new_candidates = [v for v in candidates if v != chosen]
            branch_and_bound(new_candidates, current_set, depth + 1)
        
        branch_and_bound(list(range(n)), [])
        
        return best_set

    # Main search
    best_result = None
    best_size = 0
    
    # Try R = 4 (norm squared <= 4)
    R = 4
    candidates = generate_all_vectors(R)
    n = len(candidates)
    
    if n < 2:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    # Compute norms
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Compute distance matrix
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Conflict if dist_sq < R
    conflict_mask = dist_sq < R
    np.fill_diagonal(conflict_mask, False)
    
    # Solve MIS
    mis_indices = solve_mis_branch_and_bound(conflict_mask, time_limit=500)
    
    if len(mis_indices) > best_size:
        best_size = len(mis_indices)
        best_result = candidates[mis_indices]
    
    # If branch and bound didn't find anything good, try greedy
    if best_size < 2:
        # Greedy approach
        available = set(range(n))
        selected = []
        while available:
            # Pick vertex with min degree in remaining graph
            min_deg = float('inf')
            pick = None
            for v in available:
                deg = sum(1 for u in available if conflict_mask[v, u])
                if deg < min_deg:
                    min_deg = deg
                    pick = v
            if pick is not None:
                selected.append(pick)
                available.discard(pick)
                # Remove neighbors
                to_remove = [u for u in available if conflict_mask[pick, u]]
                for u in to_remove:
                    available.discard(u)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = candidates[selected]
    
    # Verify result
    if best_result is not None and best_size >= 2:
        result = np.array(best_result, dtype=np.int64)
        max_norm = np.max(np.sum(result**2, axis=1))
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d2 = np.dot(result[i] - result[j], result[i] - result[j])
                if d2 < max_norm:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return result
    
    # Fallback
    return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

if __name__ == "__main__":
    res = kissing_number11()
    print(f"Size: {len(res)}")
