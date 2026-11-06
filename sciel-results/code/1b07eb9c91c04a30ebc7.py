import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    def hex_dist(a, b):
        return max(abs(a), abs(b), abs(a + b))

    def point_dist_sq(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        da = a1 - a2
        db = b1 - b2
        return da*da + da*db + db*db

    # Generate all points with hex_dist <= 3 (37 points total)
    all_pts = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist(a, b) <= 3:
                all_pts.append((a, b))

    outer_pts = [p for p in all_pts if hex_dist(p[0], p[1]) == 3]
    inner_pts = [p for p in all_pts if hex_dist(p[0], p[1]) < 3]

    # Precompute adjacency: for each pair, whether they're at distance 1
    # Map point -> index
    pt_to_idx = {p: i for i, p in enumerate(all_pts)}
    n_all = len(all_pts)
    adj = [[False] * n_all for _ in range(n_all)]
    for i in range(n_all):
        for j in range(i + 1, n_all):
            if point_dist_sq(all_pts[i], all_pts[j]) == 1:
                adj[i][j] = True
                adj[j][i] = True

    # Precompute for each outer point, how many neighbors it has in inner_pts
    # and among other outer points
    inner_set = set(inner_pts)
    outer_set = set(outer_pts)

    best_score = 0
    best_remove = None

    # For each combination of 5 outer points to remove, count edges in remaining 32
    # Edge count = edges within inner + edges between inner and kept outer + edges within kept outer
    # Precompute inner-inner edges
    inner_inner_edges = 0
    for i in range(len(inner_pts)):
        for j in range(i + 1, len(inner_pts)):
            if point_dist_sq(inner_pts[i], inner_pts[j]) == 1:
                inner_inner_edges += 1

    # For each outer point, count its edges to inner points and to other outer points
    outer_to_inner = {}
    outer_to_outer = {}
    for p in outer_pts:
        cnt_inner = 0
        cnt_outer = 0
        for q in inner_pts:
            if point_dist_sq(p, q) == 1:
                cnt_inner += 1
        for q in outer_pts:
            if q != p and point_dist_sq(p, q) == 1:
                cnt_outer += 1
        outer_to_inner[p] = cnt_inner
        outer_to_outer[p] = cnt_outer

    # Total edges for a set of kept outer points K:
    # inner_inner_edges + sum(p in K) outer_to_inner[p] + edges among K
    # edges among K = sum(p in K) outer_to_outer[p] / 2  ... wait, that double counts
    # Actually: sum over p in K of (number of neighbors of p in K) / 2

    # Precompute edge existence between outer points for faster lookup
    outer_adj = {}
    for i, p1 in enumerate(outer_pts):
        for j, p2 in enumerate(outer_pts):
            if i < j and point_dist_sq(p1, p2) == 1:
                outer_adj[(p1, p2)] = True
                outer_adj[(p2, p1)] = True

    # Optimization: Precompute the base score (all outer kept) and then subtract contributions of removed points?
    # Total edges = E(inner) + E(outer) + E(inner, outer)
    # If we remove set R from outer:
    # New E = E(inner) + E(outer\R) + E(inner, outer\R)
    # E(outer\R) = E(outer) - E(R) - E(R, outer\R)
    # E(inner, outer\R) = E(inner, outer) - E(inner, R)
    # So Total(R) = Base - [E(R) + E(R, outer\R) + E(inner, R)]
    # Where Base = E(inner) + E(outer) + E(inner, outer)
    # Let's calculate Base first.
    
    total_outer_edges = sum(outer_to_outer[p] for p in outer_pts) // 2
    total_outer_inner_edges = sum(outer_to_inner[p] for p in outer_pts)
    base_score = inner_inner_edges + total_outer_edges + total_outer_inner_edges
    
    # For a removal set R:
    # Loss = E(R) + E(R, kept_outer) + E(inner, R)
    # Note: E(R) is edges among removed.
    # E(R, kept_outer) is edges between removed and kept.
    # E(inner, R) is edges between inner and removed.
    # Total Loss = E(R) + E(R, kept_outer) + E(inner, R)
    # Also note that for any p in R, its total degree to all other outer + inner is fixed.
    # deg(p) = outer_to_outer[p] + outer_to_inner[p]
    # Sum_{p in R} deg(p) = 2*E(R) + E(R, kept_outer) + E(inner, R)
    # So E(R) + E(R, kept_outer) + E(inner, R) = Sum_{p in R} deg(p) - E(R)
    
    # So Total(R) = Base - (Sum_{p in R} deg(p) - E(R))
    # We want to maximize Total(R), which is minimizing (Sum_{p in R} deg(p) - E(R))
    
    best_loss = float('inf')
    best_remove = None
    
    # Precompute degrees
    deg = {p: outer_to_outer[p] + outer_to_inner[p] for p in outer_pts}
    
    for remove_combo in combinations(outer_pts, 5):
        if time.time() > deadline:
            break
        sum_deg = sum(deg[p] for p in remove_combo)
        
        # Calculate E(R) - edges among removed points
        e_r = 0
        for i in range(len(remove_combo)):
            for j in range(i + 1, len(remove_combo)):
                if (remove_combo[i], remove_combo[j]) in outer_adj:
                    e_r += 1
                    
        loss = sum_deg - e_r
        
        if loss < best_loss:
            best_loss = loss
            best_remove = remove_combo
            # Max possible score is 76. Base is likely around 80-90.
            # If Base - loss >= 76, we can break.
            if base_score - loss >= 76:
                break

    if best_remove is None:
        # Fallback: just use first 32 points
        best_result = all_pts[:32]
    else:
        remove_set = set(best_remove)
        best_result = [p for p in all_pts if p not in remove_set]

    return [[a, b] for a, b in best_result]
