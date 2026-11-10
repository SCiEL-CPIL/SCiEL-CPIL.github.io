"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from itertools import combinations
import time

_DEADLINE = 120.0  # wall-clock seconds allowed by evaluator
_START_TIME = time.time()

def _dist2(a1, b1, a2, b2):
    """Squared Euclidean distance between two lattice points (a,b) and (c,d).
    Distance^2 = (a-c)^2 + (a-c)(b-d) + (b-d)^2
    """
    da = a1 - a2
    db = b1 - b2
    return da*da + da*db + db*db

def _count_edges(points):
    """Count number of pairs at distance exactly 1 (dist2 == 1)."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if _dist2(points[i][0], points[i][1], points[j][0], points[j][1]) == 1:
                count += 1
    return count

def _hex_dist(a, b):
    """Hex distance (graph distance on triangular lattice) from origin.
    For basis vectors at 60 degrees, hex_dist = max(|a|, |b|, |a+b|).
    """
    return max(abs(a), abs(b), abs(a+b))

def _generate_hex_region(radius):
    """Generate all lattice points with hex_dist <= radius."""
    points = []
    for a in range(-radius, radius+1):
        for b in range(-radius, radius+1):
            if _hex_dist(a, b) <= radius:
                points.append((a, b))
    return points

def _find_optimal_32():
    """
    Find the optimal 32-point subset of the 37-point hexagonal region (radius 3)
    that maximizes unit-distance edges using branch-and-bound.
    
    The 37-point region has 78 edges total. We need to remove 5 points.
    For a removed set R, edges_remaining = total_edges - sum(degrees[r] for r in R) + edges_within_R.
    We maximize this, equivalently minimize sum(degrees[r]) - edges_within_R.
    
    Branch-and-bound: build the removed set one vertex at a time. At each step,
    compute an upper bound on the best possible completion. Prune if the bound
    cannot beat the current best.
    """
    all_points = _generate_hex_region(3)
    n = len(all_points)
    assert n == 37, f"Expected 37 points, got {n}"
    
    # Build adjacency as bitmasks for fast edge checks
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if _dist2(all_points[i][0], all_points[i][1], all_points[j][0], all_points[j][1]) == 1:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    total_edges = sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
    degrees = [bin(adj_mask[i]).count('1') for i in range(n)]
    
    # Sort vertices by degree descending for better pruning
    order = sorted(range(n), key=lambda v: -degrees[v])
    
    best_edges = -1
    best_removed = None
    deadline = _START_TIME + _DEADLINE - 5.0  # 5s safety margin
    
    # Branch and bound: select 5 vertices to remove
    # State: (depth, current_removed_list, current_deg_sum, current_edges_within)
    # Upper bound at depth d: we've picked d vertices, need to pick (5-d) more.
    # Best case: the remaining picks contribute minimum deg_sum and maximum edges_within.
    # A simple upper bound: current_edges_remaining + max possible additional edges.
    # edges_remaining = total_edges - deg_sum + edges_within
    # To maximize, we want to minimize deg_sum and maximize edges_within for the remaining picks.
    # Upper bound on edges_remaining: total_edges - (current_deg_sum + min_possible_additional_deg_sum) + (current_edges_within + max_possible_additional_edges_within)
    # min_possible_additional_deg_sum: sum of the (5-d) smallest degrees among remaining vertices
    # max_possible_additional_edges_within: at most C(5-d, 2) + edges between already picked and new picks
    
    def _upper_bound(depth, removed_set, deg_sum, edges_within, remaining_candidates):
        """Compute an upper bound on edges_remaining for any completion of this partial solution."""
        if depth == 5:
            return total_edges - deg_sum + edges_within
        
        num_to_pick = 5 - depth
        # Lower bound on additional degree sum: pick the num_to_pick vertices with smallest degrees
        sorted_remaining = sorted(remaining_candidates, key=lambda v: degrees[v])
        min_add_deg = sum(degrees[v] for v in sorted_remaining[:num_to_pick])
        
        # Upper bound on additional edges within: 
        # Edges among the new picks: at most C(num_to_pick, 2)
        # Edges between already-picked and new picks: for each already-picked vertex, at most num_to_pick edges
        max_add_edges_within = (num_to_pick * (num_to_pick - 1)) // 2 + len(removed_set) * num_to_pick
        
        ub = total_edges - (deg_sum + min_add_deg) + (edges_within + max_add_edges_within)
        return ub
    
    def _branch_and_bound(depth, removed_list, removed_set, deg_sum, edges_within, candidates):
        nonlocal best_edges, best_removed
        
        if time.time() > deadline:
            return
        
        if depth == 5:
            edges_remaining = total_edges - deg_sum + edges_within
            if edges_remaining > best_edges:
                best_edges = edges_remaining
                best_removed = list(removed_list)
            return
        
        # Upper bound check
        ub = _upper_bound(depth, removed_set, deg_sum, edges_within, candidates)
        if ub <= best_edges:
            return
        
        # Try each candidate
        for idx, v in enumerate(candidates):
            new_deg_sum = deg_sum + degrees[v]
            
            # Count edges from v to already-removed vertices
            edges_to_removed = 0
            for r in removed_list:
                if adj_mask[v] & (1 << r):
                    edges_to_removed += 1
            
            new_edges_within = edges_within + edges_to_removed
            
            # Quick prune: even with best case, can we beat current best?
            # Optimistic: remaining picks add 0 to deg_sum and max to edges_within
            if depth + 1 < 5:
                num_remaining = 5 - (depth + 1)
                # Very optimistic bound
                opt_ub = total_edges - new_deg_sum + new_edges_within + num_remaining * 6  # 6 is max degree
                if opt_ub <= best_edges:
                    continue
            
            new_candidates = candidates[idx+1:]
            _branch_and_bound(depth + 1, removed_list + [v], removed_set | {v}, 
                             new_deg_sum, new_edges_within, new_candidates)
        
    # Initial candidates: all 37 vertices
    _branch_and_bound(0, [], set(), 0, 0, order)
    
    # Fallback: if B&B didn't find a solution (shouldn't happen), use brute force
    if best_removed is None:
        for removed in combinations(range(n), 5):
            deg_sum = sum(degrees[r] for r in removed)
            edges_within = 0
            for i in range(5):
                for j in range(i+1, 5):
                    if adj_mask[removed[i]] & (1 << removed[j]):
                        edges_within += 1
            edges_remaining = total_edges - deg_sum + edges_within
            if edges_remaining > best_edges:
                best_edges = edges_remaining
                best_removed = list(removed)
    
    # Construct the final set of 32 points
    removed_set = set(best_removed)
    result = [list(all_points[i]) for i in range(n) if i not in removed_set]
    
    assert len(result) == 32
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64
    
    return result, best_edges

# Pre-compute the optimal solution at module load time
_OPTIMAL_POINTS, _OPTIMAL_EDGES = _find_optimal_32()

def construct():
    """
    Construct 32 points on the triangular lattice to maximize unit-distance edges.
    Uses branch-and-bound over the 37-point hexagonal region to find the optimal 32-point subset.
    """
    return _OPTIMAL_POINTS

# EVOLVE-BLOCK-END
