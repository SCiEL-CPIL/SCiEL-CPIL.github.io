"""Guided local search penalty for TSP: 2-opt move regret model."""
import numpy as np
import time

_DEADLINE = time.perf_counter() + 310.0  # safety margin before the 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = edge_distance.shape[0]
    if n <= 2:
        return edge_distance.copy()

    # True distances (always input is the true matrix)
    d = np.asarray(edge_distance, dtype=float)

    # Build the edge list of the current tour (closed cycle)
    tour = np.asarray(local_opt_tour, dtype=int)
    # Ensure length == n and is a permutation; fallback to identity if malformed
    if tour.shape[0] != n:
        tour = np.arange(n)

    # Edge endpoints for the closed tour
    i = tour[:-1]
    j = tour[1:]
    # closing edge
    i = np.append(i, tour[0])
    j = np.append(j, tour[-1])

    # Lengths of each tour edge (true)
    tour_len = d[i, j]

    # Compute 2-opt move regret for each edge in the tour
    # Regret of edge (a,b) = minimum length increase to remove it via a single 2-opt swap
    # A 2-opt swap removes edges (a,b) and (c,d) and adds (a,c) and (b,d) or (a,d) and (b,c)
    # To remove edge (a,b), we need to find the best (c,d) such that
    # d[a,c] + d[b,d] < d[a,b] + d[c,d]  OR  d[a,d] + d[b,c] < d[a,b] + d[c,d]
    # Regret = d[a,b] + d[c,d] - min(d[a,c]+d[b,d], d[a,d]+d[b,c])
    # If regret <= 0, the edge can be removed with improvement (negative regret)
    # If regret > 0, removing it requires paying that increase

    # For efficiency, we compute regret for each tour edge
    # This is O(n^3) naive, but for n=100 it's manageable
    # We can vectorize partially

    # For each edge index e with endpoints (i[e], j[e]):
    #   We need min over all other edges (c,d) in the tour of:
    #     tour_len[e] + tour_len[f] - max(d[i[e],c]+d[j[e],d], d[i[e],d]+d[j[e],c])
    #   where (c,d) are endpoints of edge f, and we consider both orientations
    
    # Actually, let's think more carefully.
    # A 2-opt move removes two edges (a,b) and (c,d) from the tour.
    # The new edges are (a,c) and (b,d) [if we reverse the segment between b and c]
    # or (a,d) and (b,c) [if we reverse the segment between c and b]
    
    # To remove edge (a,b), we pair it with some other edge (c,d).
    # The cost change is: -d[a,b] - d[c,d] + min(d[a,c]+d[b,d], d[a,d]+d[b,c])
    # Regret of removing (a,b) alone = d[a,b] - max over compatible (c,d) of [d[c,d] - min(d[a,c]+d[b,d], d[a,d]+d[b,c])]
    
    # Simpler: for each edge e, compute the best 2-opt move that includes edge e.
    # If no such move improves, regret = 0 (edge is "locked" in local optimum)
    # If best move has cost delta_e, regret = max(0, -delta_e) where delta_e is the improvement
    
    # Let's compute for each edge e, the minimum delta over all 2-opt moves involving e
    # delta(e, f) = -d[i_e,j_e] - d[i_f,j_f] + min(d[i_e,i_f]+d[j_e,j_f], d[i_e,j_f]+d[j_e,i_f])
    # Wait, need to be careful with orientation.
    
    # Let me use a cleaner formulation.
    # Tour edges: (v_0, v_1), (v_1, v_2), ..., (v_{n-1}, v_0)
    # Edge e connects v_e to v_{e+1 mod n}
    
    # For a 2-opt move removing edges e and f:
    # The segment between v_{e+1} and v_f gets reversed.
    # New edges: (v_e, v_f) and (v_{e+1}, v_{f+1 mod n})
    # OR the other orientation depending on which way we cut.
    
    # Actually, standard 2-opt: remove (a,b) and (c,d), add (a,c) and (b,d) where
    # a,b,c,d are in tour order.
    
    # Let's just compute regret numerically for each edge.
    # For edge e with endpoints a=i[e], b=j[e]:
    #   For each other edge f with endpoints c=i[f], d=j[f]:
    #     Consider both ways to reconnect:
    #       option1: new edges (a,c) and (b,d), removed edges (a,b) and (c,d)
    #       option2: new edges (a,d) and (b,c), removed edges (a,b) and (c,d)
    #     delta1 = d[a,c] + d[b,d] - d[a,b] - d[c,d]
    #     delta2 = d[a,d] + d[b,c] - d[a,b] - d[c,d]
    #     best_delta = min(delta1, delta2)
    #   regret[e] = max(0, -min_f best_delta(e,f))
    
    # This is O(n^2) per edge, so O(n^3) total. For n=100, that's 1e6 operations, fine.
    
    # But we can vectorize! For each edge e, compute best_delta over all f simultaneously.
    
    # Let's build matrices.
    # E[i, j] = 1 if (i,j) is a tour edge, else 0
    # Actually, let's just do it directly.
    
    # For each pair of edges (e, f), compute the two possible deltas.
    # We'll create arrays for all edge pairs.
    
    # Edge endpoints: i[0..n-1], j[0..n-1]
    # For edge e and edge f:
    #   a = i[e], b = j[e], c = i[f], d = j[f]
    #   delta1 = d[a,c] + d[b,d] - d[a,b] - d[c,d]
    #   delta2 = d[a,d] + d[b,c] - d[a,b] - d[c,d]
    
    # We want for each e: min_f min(delta1, delta2)
    
    # Vectorized: create n x n matrix where entry (e,f) = min(delta1(e,f), delta2(e,f))
    # Then regret[e] = max(0, -min_f M[e,f])
    
    # Build index matrices
    # I_e = i[e] for all e, J_e = j[e] for all e
    # For matrix M, we need I_e for rows, I_f for cols, etc.
    
    I_e = i[:, None]  # n x 1
    J_e = j[:, None]  # n x 1
    I_f = i[None, :]  # 1 x n
    J_f = j[None, :]  # 1 x n
    
    # d[a,c] where a=I_e, c=I_f: shape n x n
    # We need d[I_e, I_f] but I_e is n x 1 and I_f is 1 x n, so broadcasting gives n x n
    d_ac = d[I_e, I_f]  # d[i[e], i[f]]
    d_bd = d[J_e, J_f]  # d[j[e], j[f]]
    d_ad = d[I_e, J_f]  # d[i[e], j[f]]
    d_bc = d[J_e, I_f]  # d[j[e], i[f]]
    
    d_ab = tour_len[:, None]  # d[i[e], j[e]] for each e, broadcast to n x n
    d_cd = tour_len[None, :]  # d[i[f], j[f]] for each f, broadcast to n x n
    
    delta1 = d_ac + d_bd - d_ab - d_cd
    delta2 = d_ad + d_bc - d_ab - d_cd
    
    M = np.minimum(delta1, delta2)
    
    # For each edge e, find the best (most negative) delta over all f != e
    # Set diagonal to +inf to exclude self-pairing
    np.fill_diagonal(M, np.inf)
    
    best_delta = np.min(M, axis=1)  # shape (n,)
    
    # Regret: if best_delta < 0, edge can be removed with improvement, regret = 0
    # If best_delta >= 0, regret = best_delta (cost to remove)
    regret = np.maximum(0, best_delta)
    
    # Now, the penalty for edge e should be proportional to (max_regret - regret[e])
    # This means edges with LOW regret (easy to remove) get HIGH penalty, forcing 2-opt to break them
    # Wait, that seems backwards. Let me re-read the directive.
    
    # "set the perturbed distance of that edge to true_dist + penalty_factor * (max_regret - edge_regret)"
    # So edges with low regret (easy to remove) get high penalty. This forces 2-opt to avoid them?
    # No wait - if we increase their distance, 2-opt is less likely to KEEP them, so it will remove them.
    # Yes! Edges that are "cheap" to remove (low regret) are the ones 2-opt would naturally remove,
    # but by penalizing them MORE, we force 2-opt to remove them even more aggressively.
    # Edges with HIGH regret (hard to remove) get less penalty, so 2-opt might keep them.
    # This seems counterintuitive. Let me think again.
    
    # Actually, I think the idea is:
    # - Edges with low regret are "fragile" - 2-opt can easily remove them
    # - By penalizing them heavily, we make 2-opt definitely remove them
    # - This breaks the current tour structure
    # - Edges with high regret are "robust" - 2-opt can't easily remove them
    # - They get less penalty, so they might survive
    
    # This makes sense for disrupting the tour.
    
    max_regret = np.max(regret)
    if max_regret > 1e-12:
        regret_norm = regret / max_regret  # in [0, 1]
    else:
        regret_norm = np.zeros_like(regret)
    
    # Penalty for each tour edge: penalty_factor * (1 - regret_norm)
    # So low regret -> high penalty, high regret -> low penalty
    
    penalty_factor = 0.5 * np.mean(tour_len)  # scale with average edge length
    
    edge_penalty = penalty_factor * (1.0 - regret_norm)
    
    # Build penalty matrix
    penalty = np.zeros((n, n), dtype=float)
    
    # Add edge penalties
    for e in range(n):
        a, b = i[e], j[e]
        penalty[a, b] += edge_penalty[e]
        penalty[b, a] += edge_penalty[e]
    
    # Add mild usage-based penalty
    usage = np.asarray(edge_n_used, dtype=float)
    usage = np.maximum(usage, 0.0)
    u_max = usage.max()
    if u_max > 0:
        usage_scaled = np.log1p(usage) / np.log1p(u_max)
    else:
        usage_scaled = np.zeros_like(usage)
    penalty += 0.02 * usage_scaled
    
    # Symmetrize
    penalty = 0.5 * (penalty + penalty.T)
    
    # Result
    result = d + penalty
    
    # Ensure finiteness
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Deadline guard
    if time.perf_counter() > _DEADLINE:
        # Return something safe
        return edge_distance.copy()
    
    return result
