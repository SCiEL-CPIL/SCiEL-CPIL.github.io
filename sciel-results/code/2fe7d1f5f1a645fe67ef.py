import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    def generate_candidates(r):
        """Generate all vectors with exactly r non-zero entries, each +/-1."""
        cands = []
        for comb in combinations(range(d), r):
            for mask in range(1 << r):
                v = np.zeros(d, dtype=np.int64)
                for idx, pos in enumerate(comb):
                    v[pos] = 1 if (mask >> idx) & 1 else -1
                cands.append(v)
        return np.array(cands)
    
    def build_conflict_graph(candidates, max_dot):
        """Build adjacency list for conflict graph where conflict means dot > max_dot."""
        n = len(candidates)
        # Compute pairwise dot products
        dots = candidates @ candidates.T
        # Conflict: dot > max_dot (and i != j)
        conflict = (dots > max_dot)
        np.fill_diagonal(conflict, False)
        # Build adjacency list
        adj = [np.where(conflict[i])[0].tolist() for i in range(n)]
        return conflict, adj
    
    def greedy_max_independent_set(n, adj, order=None):
        """Greedy: repeatedly pick vertex with fewest remaining conflicts."""
        if order is None:
            order = list(range(n))
        else:
            order = list(order)
        
        selected = set()
        remaining = set(order)
        
        while remaining:
            # Pick vertex with minimum degree among remaining
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                deg = len([u for u in adj[v] if u in remaining])
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                if deg == 0:
                    break
            if best_v is None:
                break
            selected.add(best_v)
            # Remove best_v and its neighbors from remaining
            remaining.discard(best_v)
            for u in adj[best_v]:
                remaining.discard(u)
        
        return sorted(selected)
    
    def local_search_1swap(selected_idx, candidates, conflict, n, max_iterations=500):
        """Try to improve by swapping one selected vertex with one unselected."""
        selected_set = set(selected_idx)
        unselected = list(set(range(n)) - selected_set)
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > deadline - 30:
                break
            
            for i in selected_idx:
                for j in unselected:
                    if time.time() > deadline - 30:
                        break
                    # Check if swapping i->j is valid
                    # Remove i, add j: j must not conflict with any other selected
                    new_selected = selected_set - {i} | {j}
                    valid = True
                    for k in new_selected:
                        if k == j:
                            continue
                        if conflict[j][k]:
                            valid = False
                            break
                    if valid:
                        # Accept swap
                        selected_set = new_selected
                        selected_idx = list(selected_set)
                        unselected = list(set(range(n)) - selected_set)
                        improved = True
                        break
                if improved or time.time() > deadline - 30:
                    break
            if time.time() > deadline - 30:
                break
        
        return sorted(selected_set)
    
    best_points = None
    best_k = 0
    
    # Try different norms
    for r in [2, 3, 4, 5]:
        if time.time() > deadline - 60:
            break
        
        candidates = generate_candidates(r)
        n = len(candidates)
        
        if n == 0:
            continue
        
        # For norm r, max_dot = r-1 (since we need min dist >= max norm = r)
        # dist^2 = |a|^2 + |b|^2 - 2*a.b = 2r - 2*a.b >= r => a.b <= r/2
        # For integer dots, a.b <= floor(r/2)
        max_dot = r // 2
        
        conflict, adj = build_conflict_graph(candidates, max_dot)
        
        # Greedy initial solution
        selected = greedy_max_independent_set(n, adj)
        k_greedy = len(selected)
        
        if k_greedy > best_k:
            best_k = k_greedy
            best_points = candidates[selected]
        
        # Local search to improve
        if time.time() < deadline - 60:
            improved_selected = local_search_1swap(selected, candidates, conflict, n)
            k_improved = len(improved_selected)
            if k_improved > best_k:
                best_k = k_improved
                best_points = candidates[improved_selected]
    
    # If no points found, return empty (shouldn't happen)
    if best_points is None:
        return np.array([], dtype=np.int64).reshape(0, d)
    
    # Final check and formatting
    points = np.around(best_points).astype(np.int64)
    return points
