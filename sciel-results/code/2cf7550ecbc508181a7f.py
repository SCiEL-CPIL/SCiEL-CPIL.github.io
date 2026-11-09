"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d) using Pareto-frontier hypervolume search."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite acquisition scores (higher = evaluate next).
    
    Uses a Pareto-frontier approach over (cost, -expected_improvement) to select
    candidates that maximize hypervolume contribution relative to the current
    non-dominated set, explicitly modeling the multi-objective trade-off.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    n = len(mu)
    
    # Compute expected improvement (EI) for each candidate
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # We want to maximize EI and minimize cost
    # So the objectives are: maximize ei, minimize cost
    # Equivalently: minimize (-ei), minimize cost
    
    # Objective vectors: (cost, -ei) -- we want to minimize both
    obj_cost = cost.copy()
    obj_neg_ei = -ei.copy()
    
    # Find the non-dominated set in the (cost, -ei) space
    # A point i dominates point j if cost[i] <= cost[j] AND -ei[i] <= -ei[j] AND at least one is strict
    # i.e., cost[i] <= cost[j] AND ei[i] >= ei[j] AND (cost[i] < cost[j] OR ei[i] > ei[j])
    
    # Efficient non-dominated set finding
    # Sort by cost, then scan
    indices = np.argsort(obj_cost)
    sorted_cost = obj_cost[indices]
    sorted_neg_ei = obj_neg_ei[indices]
    sorted_ei = ei[indices]
    
    # A point is non-dominated if no other point has both cost <= and ei >=
    # After sorting by cost, a point is dominated if there exists a previous point with ei >= current ei
    is_nondominated = np.ones(n, dtype=bool)
    max_ei_so_far = -np.inf
    for i in range(n):
        idx = indices[i]
        if sorted_ei[i] <= max_ei_so_far:
            is_nondominated[idx] = False
        else:
            max_ei_so_far = sorted_ei[i]
    
    # Get the non-dominated points
    nd_indices = np.where(is_nondominated)[0]
    nd_cost = obj_cost[nd_indices]
    nd_ei = ei[nd_indices]
    
    # Compute hypervolume contribution for each candidate
    # For a point (c, e) where c is cost and e is ei, the hypervolume contribution
    # relative to the non-dominated set is the area of the rectangle in the 
    # (cost, -ei) space that this point would add if inserted.
    
    # Actually, let's think about this differently. The hypervolume of a set of points
    # in 2D (minimizing both objectives) with reference point (ref_cost, ref_neg_ei) is:
    # HV = sum over non-dominated points of the area of the rectangle they contribute.
    
    # For our purposes, we want to score each candidate by how much it would improve
    # the hypervolume if added to the current non-dominated set.
    
    # Reference point: use the worst cost and worst (most negative) ei among all points
    # Or better, use a reasonable reference. Let's use max cost and min ei (which is 0).
    ref_cost = np.max(obj_cost) * 1.1
    ref_neg_ei = np.min(obj_neg_ei) - 0.1  # ref for -ei, so lower is worse
    
    # For a single point (c, ne) where ne = -ei, the hypervolume contribution relative to
    # the current non-dominated front is:
    # It's the area of the region dominated by (c, ne) that is not already dominated by the front.
    
    # In 2D, for the non-dominated front sorted by cost, the hypervolume can be computed as:
    # For each consecutive pair of non-dominated points, the contribution is a rectangle.
    
    # Let's compute the hypervolume of the current non-dominated set first.
    # Sort non-dominated points by cost (ascending)
    nd_order = np.argsort(nd_cost)
    nd_cost_sorted = nd_cost[nd_order]
    nd_neg_ei_sorted = obj_neg_ei[nd_indices[nd_order]]
    
    # Compute HV of current front
    def compute_hv(costs, neg_eis, ref_c, ref_ne):
        """Compute hypervolume of a set of points minimizing (cost, neg_ei)."""
        if len(costs) == 0:
            return 0.0
        # Sort by cost
        order = np.argsort(costs)
        c_sorted = costs[order]
        ne_sorted = neg_eis[order]
        
        hv = 0.0
        # The last point extends to the reference
        for i in range(len(c_sorted)):
            c_next = ref_c if i == len(c_sorted) - 1 else c_sorted[i + 1]
            # The rectangle for point i has width (c_next - c_sorted[i]) and height (ref_ne - ne_sorted[i])
            # But we need to be careful: the height is the extent in the neg_ei direction
            width = c_next - c_sorted[i]
            height = ref_ne - ne_sorted[i]
            if width > 0 and height > 0:
                hv += width * height
        return hv
    
    current_hv = compute_hv(nd_cost, obj_neg_ei[nd_indices], ref_cost, ref_neg_ei)
    
    # For each candidate, compute the new HV if added, then the delta
    # To be efficient, we can compute the marginal contribution analytically.
    
    # For a point (c, ne), its marginal hypervolume contribution when added to the front is:
    # The area of the rectangle from (c, ne) to (ref_c, ref_ne) that is NOT already covered by the front.
    
    # In 2D, this can be computed by finding where the point would be inserted in the sorted front
    # and computing the difference.
    
    # Let's compute scores for all candidates efficiently.
    scores = np.zeros(n)
    
    # For each candidate i with (c_i, ne_i):
    # The marginal HV contribution is the area of the region dominated by (c_i, ne_i) 
    # that is not dominated by any point in the current non-dominated set.
    
    # A point (c_j, ne_j) dominates (c_i, ne_i) if c_j <= c_i and ne_j <= ne_i.
    # The region dominated by (c_i, ne_i) is [c_i, ref_c] x [ne_i, ref_ne].
    # The part of this region already dominated by the front is the union of rectangles from front points.
    
    # For efficiency, let's precompute the front structure.
    # Sort the non-dominated front by cost.
    if len(nd_cost) > 0:
        nd_order = np.argsort(nd_cost)
        front_c = nd_cost[nd_order]
        front_ne = obj_neg_ei[nd_indices[nd_order]]
        front_ei = ei[nd_indices[nd_order]]
        
        # For each candidate, find the marginal contribution
        # The marginal HV = area under the curve of the front from c_i to ref_c, 
        # minus the rectangle from (c_i, ne_i) to (ref_c, ref_ne), but only the part where front is above ne_i
        
        # Actually, let's think more carefully.
        # The hypervolume of a set S is the measure of the region dominated by at least one point in S.
        # When we add point p, the new HV = old HV + marginal(p).
        # marginal(p) = measure of region dominated by p but not by any point in S.
        
        # Region dominated by p: [c_p, ref_c] x [ne_p, ref_ne]
        # Region dominated by S: union of [c_j, ref_c] x [ne_j, ref_ne] for j in S
        
        # The region dominated by p but not by S is the part of [c_p, ref_c] x [ne_p, ref_ne] 
        # that is not in any [c_j, ref_c] x [ne_j, ref_ne].
        
        # This is equivalent to: for x in [c_p, ref_c], the y-range [ne_p, ref_ne] minus 
        # the union of [ne_j, ref_ne] for j with c_j <= x.
        
        # For a given x, the points in S with c_j <= x dominate y in [min_{j: c_j<=x} ne_j, ref_ne].
        # So the undominated y-range at x is [ne_p, min(ne_p, min_{j: c_j<=x} ne_j)] if ne_p < min_{j: c_j<=x} ne_j, else empty.
        
        # Let m(x) = min_{j in S, c_j <= x} ne_j. Then the undominated height at x is max(0, min(ne_p, m(x)) - ne_p) 
        # wait, that's not right.
        
        # The region dominated by p at x is [ne_p, ref_ne].
        # The region dominated by S at x is [m(x), ref_ne] where m(x) = min_{j: c_j <= x} ne_j.
        # The part of [ne_p, ref_ne] not in [m(x), ref_ne] is [ne_p, min(ne_p, m(x))] if ne_p < m(x), else empty.
        # So the undominated height at x is max(0, m(x) - ne_p) if m(x) > ne_p, else 0.
        
        # Wait, if m(x) < ne_p, then [m(x), ref_ne] covers [ne_p, ref_ne] entirely, so undominated height is 0.
        # If m(x) > ne_p, then [m(x), ref_ne] doesn't cover [ne_p, m(x)], so undominated height is m(x) - ne_p.
        
        # So marginal(p) = integral from c_p to ref_c of max(0, m(x) - ne_p) dx
        
        # m(x) is a step function that changes at the cost values of the front points.
        
        # Let's precompute the front points sorted by cost.
        # front_c[k], front_ne[k] for k = 0..K-1
        
        # For x in [front_c[k], front_c[k+1]), m(x) = min(front_ne[0], ..., front_ne[k]) = front_ne[k] 
        # (since the front is non-dominated, front_ne is decreasing as front_c increases)
        
        # Actually, for a non-dominated front sorted by cost ascending, the ne values should be descending 
        # (since if cost increases, ne must also increase for non-domination... wait no).
        
        # Non-dominated: point i dominates point j if c_i <= c_j and ne_i <= ne_j.
        # So if c_i < c_j, we must have ne_i > ne_j for both to be non-dominated.
        # So as cost increases, ne decreases.
        
        # So front_c is ascending, front_ne is descending.
        
        # m(x) for x in [front_c[k], front_c[k+1]) is min(front_ne[0], ..., front_ne[k]) = front_ne[k] 
        # since front_ne is descending, the min of the first k+1 is front_ne[k].
        
        # Wait, that's the min, but we want the minimum ne among points with c_j <= x.
        # Since front_ne is descending, front_ne[0] > front_ne[1] > ... > front_ne[K-1].
        # For x in [front_c[k], front_c[k+1]), the points with c_j <= x are indices 0..k.
        # The min ne among these is front_ne[k] (the smallest ne, which is the last one).
        
        # So m(x) = front_ne[k] for x in [front_c[k], front_c[k+1]).
        
        # And for x < front_c[0], m(x) = +inf (no points dominate in cost), so undominated height is 0 
        # (since m(x) - ne_p = inf, but we cap at... actually if no points have c_j <= x, then S doesn't 
        # dominate anything at x, so the entire [ne_p, ref_ne] is undominated. So undominated height = ref_ne - ne_p.
        
        # Hmm, let me reconsider. If no point in S has c_j <= x, then S doesn't dominate any y at x.
        # So the undominated height is ref_ne - ne_p (the full range).
        
        # Let's define m(x) = min_{j: c_j <= x} ne_j, with m(x) = ref_ne if no such j (i.e., S doesn't cover x).
        # Then undominated height at x = max(0, min(m(x), ref_ne) - ne_p) if m(x) > ne_p, else 0.
        # Wait, if m(x) = ref_ne (no coverage), then undominated height = ref_ne - ne_p.
        # If m(x) < ne_p, undominated height = 0.
        # If ne_p < m(x) < ref_ne, undominated height = m(x) - ne_p.
        
        # So undominated height = clip(m(x) - ne_p, 0, ref_ne - ne_p)
        
        # Let's compute this numerically for each candidate.
        
        K = len(front_c)
        
        # For each candidate, we need to integrate max(0, m(x) - ne_i) from c_i to ref_c.
        
        # This is a bit expensive to do for all 2000 candidates with a loop, but let's try.
        
        # Actually, we can vectorize this.
        
        # For each candidate i, the integral is:
        # Sum over intervals [max(c_i, front_c[k]), min(ref_c, front_c[k+1])] of (front_ne[k] - ne_i) * interval_length
        # where front_ne[k] > ne_i, plus the part before front_c[0] if c_i < front_c[0].
        
        # Let's just do a loop over candidates since n=2000 and K is likely small.
        
        for i in range(n):
            c_i = obj_cost[i]
            ne_i = obj_neg_ei[i]
            
            if c_i >= ref_cost:
                scores[i] = 0.0
                continue
            
            # Compute the integral
            integral = 0.0
            
            # Part before the first front point
            if c_i < front_c[0]:
                # From c_i to front_c[0], m(x) = ref_ne (no coverage)
                length = front_c[0] - c_i
                height = ref_neg_ei - ne_i
                if height > 0:
                    integral += length * height
            
            # Part within the front intervals
            for k in range(K):
                interval_start = max(c_i, front_c[k])
                interval_end = min(ref_cost, front_c[k + 1] if k + 1 < K else ref_cost)
                
                if interval_start >= interval_end:
                    continue
                
                # m(x) = front_ne[k] in this interval
                height = front_ne[k] - ne_i
                if height > 0:
                    integral += (interval_end - interval_start) * height
            
            scores[i] = integral
    else:
        # No non-dominated points (shouldn't happen), use simple EI/cost
        scores = ei / np.maximum(cost, 1e-9)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
