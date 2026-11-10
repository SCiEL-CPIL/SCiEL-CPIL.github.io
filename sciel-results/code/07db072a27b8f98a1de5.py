"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from itertools import combinations
import time

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
    Wait, let me verify. The neighbors of (0,0) are (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1).
    hex_dist(1,0) = max(1,0,1) = 1. OK.
    hex_dist(0,1) = max(0,1,1) = 1. OK.
    hex_dist(1,-1) = max(1,1,0) = 1. OK.
    hex_dist(1,1) = max(1,1,2) = 2. OK (it's 2 steps away).
    So this is correct.
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
    that maximizes unit-distance edges.
    
    The 37-point region has 78 edges total (for a full hex of radius 3).
    We need to remove 5 points. Removing a point removes its edges to remaining points.
    We want to minimize the number of edges removed.
    
    Actually, let's think differently. The full hex of radius 3 has 37 points and 78 edges.
    We want to pick 32 points (remove 5) to maximize edges in the induced subgraph.
    Equivalently, minimize edges "lost" = edges incident to removed points that connect to kept points.
    
    Total edges in full graph: 78.
    Edges in induced subgraph = 78 - (edges incident to removed vertices that go to kept vertices).
    
    To maximize induced edges, we want to minimize cross edges between removed and kept sets.
    This is equivalent to: for the 5 removed vertices R, minimize |{(u,v) in E : u in R, v in V\\R}|.
    
    Since the graph is small (37 vertices), we can enumerate all C(37,5) = 435897 ways to remove 5 vertices.
    For each, compute the number of edges in the remaining 32-vertex induced subgraph.
    """
    # Generate the 37-point hex region
    all_points = _generate_hex_region(3)
    n = len(all_points)
    assert n == 37, f"Expected 37 points, got {n}"
    
    # Build adjacency: for each point, list of indices of neighbors at distance 1
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if _dist2(all_points[i][0], all_points[i][1], all_points[j][0], all_points[j][1]) == 1:
                adj[i].append(j)
                adj[j].append(i)
    
    total_edges = sum(len(adj[i]) for i in range(n)) // 2
    # For hex of radius 3, total edges should be 78
    
    # For each point, precompute its degree
    degrees = [len(adj[i]) for i in range(n)]
    
    # Enumerate all ways to remove 5 points
    # For a given set R of removed points, the number of edges in the induced subgraph on V\R is:
    # total_edges - (sum of degrees of points in R) + (edges within R)
    # Because: edges removed = edges from R to (V\R) + edges within R
    # edges from R to (V\R) = sum(degrees[r] for r in R) - 2*(edges within R)
    # Wait: sum of degrees of R counts each edge within R twice, and each edge from R to V\R once.
    # So edges from R to V\R = sum(degrees[r]) - 2*edges_within_R
    # Edges removed from total = edges_from_R_to_VR + edges_within_R = sum(degrees[r]) - edges_within_R
    # Edges remaining = total_edges - (sum(degrees[r]) - edges_within_R) = total_edges - sum(degrees[r]) + edges_within_R
    
    best_edges = -1
    best_removed = None
    
    # To speed up, we can be smart. But C(37,5) = 435897 is small enough for brute force.
    for removed in combinations(range(n), 5):
        # Compute sum of degrees of removed points
        deg_sum = sum(degrees[r] for r in removed)
        # Compute edges within removed set
        edges_within = 0
        for i in range(5):
            for j in range(i+1, 5):
                if removed[j] in adj[removed[i]]:
                    edges_within += 1
        edges_remaining = total_edges - deg_sum + edges_within
        if edges_remaining > best_edges:
            best_edges = edges_remaining
            best_removed = removed
    
    # Construct the final set of 32 points
    removed_set = set(best_removed)
    result = [list(all_points[i]) for i in range(n) if i not in removed_set]
    
    assert len(result) == 32
    # Verify all coordinates are within bounds
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64
    
    return result, best_edges

# Pre-compute the optimal solution at module load time
_OPTIMAL_POINTS, _OPTIMAL_EDGES = _find_optimal_32()

def construct():
    """
    Construct 32 points on the triangular lattice to maximize unit-distance edges.
    Uses exact enumeration over the 37-point hexagonal region to find the optimal 32-point subset.
    """
    return _OPTIMAL_POINTS

# EVOLVE-BLOCK-END
