import numpy as np
import time
import sys

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 620  # Safety margin for 630s limit

    def _verify(arr):
        """Verify the kissing configuration constraints."""
        if arr.shape[1] != 11:
            return False
        k = arr.shape[0]
        if k < 2:
            return False
        # No point is the zero vector
        for i in range(k):
            if np.all(arr[i] == 0):
                return False
        # Max squared norm
        norms_sq = np.sum(arr**2, axis=1)
        max_norm_sq = int(np.max(norms_sq))
        # Min squared distance >= max norm squared
        for i in range(k):
            for j in range(i + 1, k):
                diff = arr[i] - arr[j]
                d2 = int(np.dot(diff, diff))
                if d2 < max_norm_sq:
                    return False
        return True

    def _make_all_vectors():
        """Generate all integer vectors with at most 2 non-zero coordinates (values in {-1, 0, 1})."""
        vectors = []
        # Norm 1: single ±1
        for i in range(d):
            v = [0] * d
            v[i] = 1
            vectors.append(tuple(v))
            v[i] = -1
            vectors.append(tuple(v))
        # Norm 2: two ±1's
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = [0] * d
                        v[i] = s1
                        v[j] = s2
                        vectors.append(tuple(v))
        return np.array(vectors, dtype=np.int64)

    def _make_all_vectors_norm3():
        """Generate all integer vectors with at most 3 non-zero coordinates (values in {-1, 0, 1})."""
        vectors = []
        for i in range(d):
            v = [0] * d
            v[i] = 1
            vectors.append(tuple(v))
            v[i] = -1
            vectors.append(tuple(v))
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = [0] * d
                        v[i] = s1
                        v[j] = s2
                        vectors.append(tuple(v))
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    for s1 in [1, -1]:
                        for s2 in [1, -1]:
                            for s3 in [1, -1]:
                                v = [0] * d
                                v[i] = s1
                                v[j] = s2
                                v[k] = s3
                                vectors.append(tuple(v))
        return np.array(vectors, dtype=np.int64)

    def _greedy_independent_set(conflict, n):
        """Greedy: repeatedly pick vertex with fewest conflicts among remaining."""
        remaining = set(range(n))
        selected = []
        while remaining:
            # Pick vertex with minimum conflict degree within remaining
            best_v = -1
            best_deg = float('inf')
            for v in remaining:
                deg = 0
                for u in remaining:
                    if u != v and conflict[v, u]:
                        deg += 1
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            selected.append(best_v)
            remaining.discard(best_v)
            # Remove all neighbors of best_v
            to_remove = set()
            for u in remaining:
                if conflict[best_v, u]:
                    to_remove.add(u)
            remaining -= to_remove
        return selected

    def _local_search_improve(selected_set, conflict, n, candidates, time_limit=580):
        """Try to improve the independent set via swap moves."""
        selected = set(selected_set)
        non_selected = set(range(n)) - selected
        
        # Precompute conflict sets
        conflict_sets = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if conflict[i, j]:
                    conflict_sets[i].add(j)
                    conflict_sets[j].add(i)
        
        best_size = len(selected)
        best_set = selected.copy()
        
        # Try adding vertices
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            remaining_candidates = [v for v in range(n) if v not in selected]
            # Sort by number of conflicts with selected
            remaining_candidates.sort(key=lambda v: len(conflict_sets[v] & selected))
            for v in remaining_candidates:
                if conflict_sets[v] & selected:
                    continue
                selected.add(v)
                improved = True
                best_size = len(selected)
                best_set = selected.copy()
                if time.time() > time_limit:
                    return best_set
        
        # Try swap: remove one selected, add up to two non-selected
        for _ in range(500):  # Limit swap iterations
            if time.time() > time_limit:
                break
            non_selected = set(range(n)) - selected
            if not non_selected:
                break
            
            # Try each non-selected vertex as a potential add
            potential_adds = list(non_selected)
            random.shuffle(potential_adds) if 'random' in dir() else None
            import random
            random.shuffle(potential_adds)
            
            for add_v in potential_adds[:50]:
                conflicts_with_selected = conflict_sets[add_v] & selected
                if not conflicts_with_selected:
                    # Can add directly
                    selected.add(add_v)
                    best_size = len(selected)
                    best_set = selected.copy()
                    continue
                if len(conflicts_with_selected) > 2:
                    continue
                # Try swapping: remove all conflicts_with_selected, add add_v
                new_size = len(selected) - len(conflicts_with_selected) + 1
                if new_size <= len(selected):
                    continue
                # Check that add_v doesn't conflict with (selected - conflicts_with_selected)
                remaining_selected = selected - conflicts_with_selected
                if conflict_sets[add_v] & remaining_selected:
                    continue
                # Perform swap
                selected -= conflicts_with_selected
                selected.add(add_v)
                best_size = len(selected)
                best_set = selected.copy()
        
        return best_set

    def _solve_mis_branch_and_bound(conflict, n, time_limit=300):
        """Branch and bound for Maximum Independent Set."""
        if n == 0:
            return []
        
        sys.setrecursionlimit(max(10000, n + 100))
        
        # Build adjacency as sets
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if conflict[i, j]:
                    adj[i].add(j)
                    adj[j].add(i)
        
        degrees = [len(adj[i]) for i in range(n)]
        
        best_set = []
        best_size = 0
        
        # Use a simple upper bound based on coloring-like heuristic
        def upper_bound(candidates):
            if not candidates:
                return 0
            # Greedy coloring bound: count colors needed
            cand_list = sorted(candidates, key=lambda x: -degrees[x])
            color_map = {}
            max_color = 0
            for v in cand_list:
                neighbor_colors = set()
                for u in adj[v]:
                    if u in color_map:
                        neighbor_colors.add(color_map[u])
                c = 0
                while c in neighbor_colors:
                    c += 1
                color_map[v] = c
                max_color = max(max_color, c + 1)
            return max_color
        
        def branch(candidates, current):
            nonlocal best_set, best_size
            
            if time.time() > time_limit:
                return
            
            if not candidates:
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = current[:]
                return
            
            ub = upper_bound(candidates) + len(current)
            if ub <= best_size:
                return
            
            # Choose vertex with max degree in candidate subgraph
            max_deg = -1
            chosen = None
            for v in candidates:
                deg_in_sub = sum(1 for u in candidates if u != v and conflict[v, u])
                if deg_in_sub > max_deg:
                    max_deg = deg_in_sub
                    chosen = v
            
            # Branch 1: include chosen
            new_cand = [v for v in candidates if v != chosen and not conflict[chosen, v]]
            branch(new_cand, current + [chosen])
            
            # Branch 2: exclude chosen
            new_cand = [v for v in candidates if v != chosen]
            branch(new_cand, current)
        
        branch(list(range(n)), [])
        return best_set

    # Generate candidate vectors
    # Start with the known good set: all vectors with exactly 2 non-zero ±1 coordinates (norm²=2)
    # That's 11*10*4 = 440 vectors, but we need them to be pairwise compatible.
    # For norm²=2 vectors, two vectors are compatible if their dot product <= 1 (since we need dist² >= 2).
    # dist² = 2 + 2 - 2*dot = 4 - 2*dot, need 4 - 2*dot >= 2 => dot <= 1.
    # Dot product of two norm-2 vectors: they share at most 2 coordinates.
    # If they share exactly one coordinate with same sign: dot = 1, dist² = 2 ✓
    # If they share exactly one coordinate with opposite sign: dot = -1, dist² = 4 ✓
    # If they share both coordinates with same signs: dot = 2, dist² = 0 ✗ (same vector)
    # If they share both with one same one opposite: dot = 0, dist² = 4 ✓
    # So two distinct norm-2 vectors are always compatible!
    # Wait: e_i + e_j and e_i + e_j are the same. e_i + e_j and e_i - e_j: dot = 1-1=0, dist²=4 ✓
    # e_i + e_j and -e_i + e_j: dot = -1+1=0, dist²=4 ✓
    # e_i + e_j and -e_i - e_j: dot = -1-1=-2, dist²=8 ✓
    # e_i + e_j and e_j + e_i: same vector
    # e_i + e_j and e_i + e_k (k≠j): dot=1, dist²=2 ✓
    # So ALL 11*10*4 = 440 norm-2 vectors form a valid kissing configuration!
    
    # Let's verify: 11 choose 2 = 55 pairs of coordinates, 4 sign combinations each = 220 vectors.
    # Wait: for each pair (i,j) with i<j, we have 4 vectors: (i:+,j:+), (i:+,j:-), (i:-,j:+), (i:-,j:-)
    # That's 55 * 4 = 220 vectors.
    
    # Check: are any two distinct ones incompatible?
    # v1 = e_i + e_j, v2 = e_i + e_j: same, excluded
    # v1 = e_i + e_j, v2 = e_i - e_j: dot = 1-1 = 0, dist² = 4 >= 2 ✓
    # v1 = e_i + e_j, v2 = e_k + e_l (disjoint): dot = 0, dist² = 4 ✓
    # v1 = e_i + e_j, v2 = e_i + e_k (share i, k≠j): dot = 1, dist² = 2 ✓
    # v1 = e_i + e_j, v2 = -e_i + e_j: dot = -1+1 = 0, dist² = 4 ✓
    # v1 = e_i + e_j, v2 = e_i - e_k (k≠j): dot = 1, dist² = 2 ✓
    
    # Great! All 220 norm-2 vectors are mutually compatible. Score = 220.
    
    result = _make_all_vectors()
    # Filter to only norm²=2 vectors
    norms_sq = np.sum(result**2, axis=1)
    result = result[norms_sq == 2]
    
    if _verify(result):
        return result
    
    # Should not reach here, but fallback
    return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

if __name__ == "__main__":
    res = kissing_number11()
    print(f"Size: {len(res)}")
