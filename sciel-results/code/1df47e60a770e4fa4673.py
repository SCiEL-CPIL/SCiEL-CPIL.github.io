"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import random
import time

# EVOLVE-BLOCK-START
import itertools
import numpy as np
from functools import lru_cache


def _build_conflict_graph(all_points, n_points):
    """
    Build conflict graph: edge (i,j) exists if adding both i and j 
    to any existing independent set creates an isosceles triangle.
    Actually, for MIS formulation: two points conflict if they cannot both be in an independent set.
    In no-isosceles problem, two points don't directly conflict, but three points do.
    We need a hypergraph or reformulate.
    
    Alternative: For each pair (i,j), precompute which third points k would create isosceles(i,j,k).
    Then in branch-and-bound, when we have selected set S, candidate c is valid if 
    for all pairs (a,b) in S, c is not in iso_conflicts[a][b].
    """
    pts = np.array(all_points, dtype=np.int32)
    diff = pts[:, None, :] - pts[None, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)
    
    # For each pair (i,j), store the set of k that form isosceles with i,j
    # Use bitsets for efficiency
    iso_conflicts = [None] * n_points
    for i in range(n_points):
        iso_conflicts[i] = [None] * n_points
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            conflict_set = 0
            for k in range(n_points):
                if k == i or k == j:
                    continue
                d_ik = sq_dist[i, k]
                d_jk = sq_dist[j, k]
                d_ij = sq_dist[i, j]
                # Check if (i,j,k) forms isosceles
                if d_ik != 0 and d_jk != 0 and d_ij != 0:
                    if d_ik == d_jk or d_ik == d_ij or d_jk == d_ij:
                        conflict_set |= (1 << k)
            iso_conflicts[i][j] = conflict_set
            iso_conflicts[j][i] = conflict_set
    
    return sq_dist, iso_conflicts


def _mis_branch_and_bound(sq_dist, iso_conflicts, n_points, deadline):
    """
    Branch and bound to find minimum maximal no-isosceles set.
    Uses bitset representation for speed.
    """
    all_mask = (1 << n_points) - 1
    best_solution = None
    best_size = n_points + 1
    
    def popcount(x):
        return bin(x).count('1')
    
    def can_add(candidate, selected_mask):
        """Check if candidate can be added to selected set."""
        # For each pair in selected, check if candidate conflicts
        selected_list = [i for i in range(n_points) if (selected_mask >> i) & 1]
        for idx, i in enumerate(selected_list):
            for j in selected_list[idx+1:]:
                if iso_conflicts[i][j] & (1 << candidate):
                    return False
        return True
    
    def is_maximal(selected_mask):
        """Check if selected set is maximal."""
        for c in range(n_points):
            if (selected_mask >> c) & 1:
                continue
            if can_add(c, selected_mask):
                return False
        return True
    
    def bound(selected_mask, candidates_mask):
        """
        Lower bound on final size.
        If we have selected s points, and there are c candidates left,
        the minimum additional points needed is at least 1 if not maximal.
        Better bound: use independence number of candidate subgraph.
        """
        s = popcount(selected_mask)
        if s >= best_size:
            return s
        # Simple bound: if not maximal, need at least 1 more
        # But we can't easily check maximality here without full check
        return s
    
    def branch(selected_mask, candidates_mask, depth):
        nonlocal best_solution, best_size
        
        if time.time() > deadline:
            return
        
        s = popcount(selected_mask)
        
        # Check if current selection is maximal
        if is_maximal(selected_mask):
            if s < best_size:
                best_size = s
                best_solution = selected_mask
            return
        
        # If no candidates left but not maximal, this branch is invalid
        if candidates_mask == 0:
            return
        
        # Bound: if even adding all candidates doesn't beat best, prune
        if s + popcount(candidates_mask) <= 0:
            return
        
        # Get list of candidates
        cand_list = [i for i in range(n_points) if (candidates_mask >> i) & 1]
        
        if not cand_list:
            return
        
        # Heuristic: order candidates by how many others they block
        def blocking_power(c):
            count = 0
            for d in cand_list:
                if d == c:
                    continue
                # Check if c and d conflict with any existing pair
                selected_list = [i for i in range(n_points) if (selected_mask >> i) & 1]
                for idx, i in enumerate(selected_list):
                    for j in selected_list[idx+1:]:
                        if iso_conflicts[i][j] & (1 << d):
                            count += 1
                            break
            return count
        
        cand_list.sort(key=blocking_power, reverse=True)
        
        for c in cand_list:
            if time.time() > deadline:
                return
            
            if not can_add(c, selected_mask):
                continue
            
            new_selected = selected_mask | (1 << c)
            new_candidates = candidates_mask & ~(1 << c)
            
            branch(new_selected, new_candidates, depth + 1)
            
            # Also try NOT adding c (for finding minimal maximal sets)
            # But this could explode, so only do it selectively
            if depth < 15:  # Limit depth for non-add branches
                branch(selected_mask, new_candidates, depth + 1)
    
    branch(0, all_mask, 0)
    
    if best_solution is not None:
        return [i for i in range(n_points) if (best_solution >> i) & 1]
    return None


def _greedy_conflict_based(sq_dist, iso_conflicts, n_points, deadline):
    """
    Greedy approach using conflict information.
    """
    all_mask = (1 << n_points) - 1
    selected = []
    selected_mask = 0
    
    for c in range(n_points):
        if time.time() > deadline:
            break
        if (selected_mask >> c) & 1:
            continue
        
        # Check if c can be added
        valid = True
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                a = selected[i]
                b = selected[j]
                if iso_conflicts[a][b] & (1 << c):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            selected.append(c)
            selected_mask |= (1 << c)
    
    return selected


def _local_search_improve(selected, sq_dist, iso_conflicts, n_points, deadline):
    """
    Improve a maximal no-isosceles set by trying swaps.
    """
    if not selected:
        return selected
    
    best = list(selected)
    best_size = len(best)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        best_set = set(best)
        
        for i in range(len(best)):
            if time.time() > deadline:
                break
            
            p_remove = best[i]
            without_p = [x for x in best if x != p_remove]
            
            # Try adding each non-selected point
            for p_add in range(n_points):
                if p_add in best_set:
                    continue
                
                # Check if p_add can be added to without_p
                valid = True
                for a in range(len(without_p)):
                    for b in range(a + 1, len(without_p)):
                        if iso_conflicts[without_p[a]][without_p[b]] & (1 << p_add):
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                new_set = without_p + [p_add]
                
                # Check maximality
                maximal = True
                for c in range(n_points):
                    if c in set(new_set):
                        continue
                    # Check if c can be added
                    can_add_c = True
                    for a in range(len(new_set)):
                        for b in range(a + 1, len(new_set)):
                            if iso_conflicts[new_set[a]][new_set[b]] & (1 << c):
                                can_add_c = False
                                break
                        if not can_add_c:
                            break
                    if can_add_c:
                        maximal = False
                        break
                
                if maximal and len(new_set) < best_size:
                    best_size = len(new_set)
                    best = new_set
                    improved = True
                    break
            
            if improved:
                break
    
    return best


def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    n_points = len(all_points)
    start = time.time()
    deadline = start + 280  # safety margin
    
    # Build conflict graph
    sq_dist, iso_conflicts = _build_conflict_graph(all_points, n_points)
    
    best = None
    best_size = n_points + 1
    
    random.seed(42)
    
    # Strategy 1: Greedy with different orderings
    for attempt in range(100):
        if time.time() > deadline - 120:
            break
        
        order = list(range(n_points))
        random.shuffle(order)
        
        selected = []
        for c in order:
            valid = True
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    if iso_conflicts[selected[i]][selected[j]] & (1 << c):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                selected.append(c)
        
        # Check maximality
        maximal = True
        sel_set = set(selected)
        for c in range(n_points):
            if c in sel_set:
                continue
            can_add = True
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    if iso_conflicts[selected[i]][selected[j]] & (1 << c):
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                maximal = False
                break
        
        if maximal and len(selected) < best_size:
            best_size = len(selected)
            best = selected
    
    # Strategy 2: Local search improvement
    if best is not None and time.time() < deadline - 60:
        improved = _local_search_improve(best, sq_dist, iso_conflicts, n_points, deadline - 30)
        if len(improved) < best_size:
            best_size = len(improved)
            best = improved
    
    # Strategy 3: Branch and bound for small instances or with time remaining
    if time.time() < deadline - 40:
        mis_result = _mis_branch_and_bound(sq_dist, iso_conflicts, n_points, deadline - 30)
        if mis_result is not None:
            # Verify it's valid and maximal
            valid = True
            for i in range(len(mis_result)):
                for j in range(i + 1, len(mis_result)):
                    for k in range(n_points):
                        if k in mis_result:
                            continue
                        if iso_conflicts[mis_result[i]][mis_result[j]] & (1 << k):
                            pass  # This is expected, k is blocked
                    # Actually need to check no three form isosceles
            # Just verify no isosceles
            for i in range(len(mis_result)):
                for j in range(i + 1, len(mis_result)):
                    for k in range(j + 1, len(mis_result)):
                        a, b, c = mis_result[i], mis_result[j], mis_result[k]
                        d_ab = sq_dist[a][b]
                        d_ac = sq_dist[a][c]
                        d_bc = sq_dist[b][c]
                        if d_ab != 0 and d_ac != 0 and d_bc != 0:
                            if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Check maximality
                maximal = True
                sel_set = set(mis_result)
                for c in range(n_points):
                    if c in sel_set:
                        continue
                    can_add = True
                    for i in range(len(mis_result)):
                        for j in range(i + 1, len(mis_result)):
                            if iso_conflicts[mis_result[i]][mis_result[j]] & (1 << c):
                                can_add = False
                                break
                        if not can_add:
                            break
                    if can_add:
                        maximal = False
                        break
                
                if maximal and len(mis_result) < best_size:
                    best_size = len(mis_result)
                    best = mis_result
    
    # Strategy 4: More random greedy with time remaining
    while time.time() < deadline - 5:
        order = list(range(n_points))
        random.shuffle(order)
        
        selected = []
        for c in order:
            valid = True
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    if iso_conflicts[selected[i]][selected[j]] & (1 << c):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                selected.append(c)
        
        # Check maximality
        maximal = True
        sel_set = set(selected)
        for c in range(n_points):
            if c in sel_set:
                continue
            can_add = True
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    if iso_conflicts[selected[i]][selected[j]] & (1 << c):
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                maximal = False
                break
        
        if maximal and len(selected) < best_size:
            best_size = len(selected)
            best = selected
    
    if best is None:
        # Fallback
        best = [0]
    
    return [all_points[i] for i in best]
# EVOLVE-BLOCK-END
