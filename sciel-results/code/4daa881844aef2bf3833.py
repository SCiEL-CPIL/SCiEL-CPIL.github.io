"""TSP next-node selection with multi-stage heuristic: hull peeling + look-ahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 142.0  # safety margin before 150s hard kill


def _compute_hull_scores(dm, cands, c, dest):
    """
    Compute convex-hull-peeling scores for candidates.
    
    A point on the convex hull has high average distance to other unvisited points.
    We combine:
      - distance from current node (prefer close)
      - hull-ness (prefer peripheral)
      - angle alignment toward destination (prefer aligned)
    
    Returns normalized score array where lower = better.
    """
    n = len(cands)
    d_to_c = dm[c, cands]
    
    # Hull-ness: average distance to other unvisited nodes
    sub_dm = dm[np.ix_(cands, cands)]
    d_jk = np.where(np.eye(n, dtype=bool), 0.0, sub_dm)
    avg_dist = d_jk.sum(axis=1) / max(n - 1, 1)
    
    # Angle component: how well does the direction from c to cand align with c to dest?
    # Use dot product of unit vectors
    d_to_dest = dm[c, dest]
    if d_to_dest > 1e-12 and n > 1:
        # Approximate angle via law of cosines: cos(angle) = (d(c,cand)^2 + d(c,dest)^2 - d(cand,dest)^2) / (2*d(c,cand)*d(c,dest))
        d_cand_dest = dm[cands, dest]
        d_cc = d_to_c
        cos_angles = (d_cc**2 + d_to_dest**2 - d_cand_dest**2) / (2.0 * d_cc * d_to_dest + 1e-12)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
    else:
        cos_angles = np.ones(n)
    
    # Normalize each component to [0, 1]
    def _normalize(x):
        mn, mx = x.min(), x.max()
        rng = mx - mn
        if rng > 1e-12:
            return (x - mn) / rng
        return np.zeros_like(x)
    
    norm_d_c = _normalize(d_to_c)
    norm_hull = _normalize(avg_dist)
    # cos_angles: higher = more aligned. We want to prefer high cos, so invert for "lower is better"
    norm_angle = _normalize(-cos_angles)  # lower score = more aligned
    
    # Weighted combination
    w_dist = 0.45
    w_hull = 0.35
    w_angle = 0.20
    
    score = w_dist * norm_d_c + w_hull * (1.0 - norm_hull) + w_angle * norm_angle
    return score


def _lookahead_cost(dm, cands, c, dest, k=3):
    """
    For each candidate, estimate the cost of visiting it next and then
    greedily visiting the next k-1 closest unvisited nodes, then going to dest.
    
    This gives a short-horizon look-ahead that captures local structure.
    """
    n = len(cands)
    if n <= k:
        # Just compute full greedy cost from each candidate
        costs = np.zeros(n)
        for i in range(n):
            cur = c
            rem = [cands[j] for j in range(n) if j != i]
            costs[i] = dm[c, cands[i]]
            cur = cands[i]
            while rem:
                d_to = dm[cur, rem]
                idx = int(np.argmin(d_to))
                costs[i] += d_to[idx]
                cur = rem[idx]
                rem = rem[:idx] + rem[idx+1:]
            costs[i] += dm[cur, dest]
        return costs
    
    costs = np.zeros(n)
    for i in range(n):
        cur = c
        total = dm[c, cands[i]]
        cur = cands[i]
        rem = [cands[j] for j in range(n) if j != i]
        for _ in range(min(k - 1, len(rem))):
            d_to = dm[cur, rem]
            idx = int(np.argmin(d_to))
            total += d_to[idx]
            cur = rem[idx]
            rem = rem[:idx] + rem[idx+1:]
        # Estimate remaining cost: sum of nearest-neighbor distances for remaining + distance to dest
        if rem:
            sub = dm[np.ix_(rem, rem)]
            nn_dists = np.min(sub, axis=1)
            total += np.sum(nn_dists) + dm[cur, dest]
        else:
            total += dm[cur, dest]
        costs[i] = total
    return costs


def _exact_small(dm, cands, c, dest):
    """Exact solution for small n using brute-force permutations with pruning."""
    n = len(cands)
    if n == 1:
        return int(cands[0])
    if n == 2:
        u, v = cands
        cost_u = dm[c, u] + dm[u, v] + dm[v, dest]
        cost_v = dm[c, v] + dm[v, u] + dm[u, dest]
        return int(cands[0] if cost_u <= cost_v else cands[1])
    if n == 3:
        best_cost = np.inf
        best_node = cands[0]
        for i in range(3):
            for j in range(3):
                if j == i:
                    continue
                k = 3 - i - j
                cost = dm[c, cands[i]] + dm[cands[i], cands[j]] + dm[cands[j], cands[k]] + dm[cands[k], dest]
                if cost < best_cost:
                    best_cost = cost
                    best_node = cands[i]
        return int(best_node)
    if n == 4:
        best_cost = np.inf
        best_node = cands[0]
        for i in range(4):
            for j in range(4):
                if j == i: continue
                for k in range(4):
                    if k == i or k == j: continue
                    l = 6 - i - j - k
                    cost = dm[c, cands[i]] + dm[cands[i], cands[j]] + dm[cands[j], cands[k]] + dm[cands[k], cands[l]] + dm[cands[l], dest]
                    if cost < best_cost:
                        best_cost = cost
                        best_node = cands[i]
        return int(best_node)
    if n <= 8:
        # Use nearest-neighbor + 2-opt for each starting node
        best_first = None
        best_total = np.inf
        
        for first_idx in range(n):
            first_node = cands[first_idx]
            rest = [cands[i] for i in range(n) if i != first_idx]
            
            # NN tour
            sub_tour = []
            cur2 = first_node
            rem2 = list(rest)
            while rem2:
                d_to = dm[cur2, rem2]
                idx2 = int(np.argmin(d_to))
                nxt2 = rem2[idx2]
                sub_tour.append(nxt2)
                rem2 = rem2[:idx2] + rem2[idx2+1:]
                cur2 = nxt2
            
            # 2-opt
            for _ in range(10):
                improved = False
                for i in range(len(sub_tour) - 1):
                    for j in range(i + 2, len(sub_tour)):
                        prev_node = first_node if i == 0 else sub_tour[i-1]
                        next_node = dest if j == len(sub_tour)-1 else sub_tour[j+1]
                        old_cost = dm[prev_node, sub_tour[i]] + dm[sub_tour[j], next_node]
                        new_cost = dm[prev_node, sub_tour[j]] + dm[sub_tour[i], next_node]
                        if new_cost < old_cost - 1e-9:
                            sub_tour = sub_tour[:i] + sub_tour[i:j+1][::-1] + sub_tour[j+1:]
                            improved = True
                if not improved:
                    break
            
            total = (dm[c, first_node] +
                     dm[first_node, sub_tour[0]] +
                     sum(dm[sub_tour[k], sub_tour[k+1]] for k in range(len(sub_tour)-1)) +
                     dm[sub_tour[-1], dest])
            
            if total < best_total:
                best_total = total
                best_first = first_node
        
        return int(best_first)
    return None


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction.
    
    Strategy:
    - For very small n (<=8): exact or near-exact search
    - For larger n: combine hull-peeling score with short look-ahead cost
    """
    # Time guard
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    dm = distance_matrix
    c = current_node
    dest = destination_node
    cands = unvisited_nodes
    
    # Exact for small n
    if n_unvisited <= 8:
        result = _exact_small(dm, cands, c, dest)
        if result is not None:
            return result
    
    # Time check before expensive computation
    if time.time() - _START_TIME > _DEADLINE - 2:
        # Fallback: nearest neighbor
        return int(cands[np.argmin(dm[c, cands])])
    
    # Compute hull score
    hull_scores = _compute_hull_scores(dm, cands, c, dest)
    
    # Compute look-ahead cost
    # Use k=3 for speed
    la_costs = _lookahead_cost(dm, cands, c, dest, k=3)
    
    # Normalize
    def _normalize(x):
        mn, mx = x.min(), x.max()
        rng = mx - mn
        if rng > 1e-12:
            return (x - mn) / rng
        return np.zeros_like(x)
    
    norm_hull = _normalize(hull_scores)
    norm_la = _normalize(la_costs)
    
    # Blend: 55% hull, 45% look-ahead
    blended = 0.55 * norm_hull + 0.45 * norm_la
    
    best_idx = int(np.argmin(blended))
    return int(cands[best_idx])
# EVOLVE-BLOCK-END
