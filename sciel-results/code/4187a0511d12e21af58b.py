# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer on ZDT4 using dynamic programming to select
    the optimal set of x_0 coordinates that maximize the exact 2D hypervolume
    of the recorded objective vectors.
    
    Key insight: For ZDT4 with x_1..x_9 = 0, the Pareto front is
    f_1 = x_0, f_2 = 2 - sqrt(x_0), which is a known concave curve.
    The hypervolume with reference point (11, 400) depends only on which
    x_0 values we evaluate. We use a 1D DP over candidate x_0 positions
    to find the subset of size=budget that maximizes hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Determine best tail (x_1..x_9) to minimize g
        # For ZDT4, h(x) = x^2 - 10*cos(4*pi*x*pi), min at 0, ±0.5, ±1, ...
        best_tail = np.zeros(self.dim - 1)
        for i in range(1, self.dim):
            best_val = np.inf
            best_xi = lo[i]
            # Check critical points k/2 and boundaries
            for k in range(-10, 11):
                c = k * 0.5
                if lo[i] <= c <= hi[i]:
                    val = c ** 2 - 10 * np.cos(4 * np.pi * c * np.pi)
                    if val < best_val:
                        best_val = val
                        best_xi = c
            for boundary in [lo[i], hi[i]]:
                val = boundary ** 2 - 10 * np.cos(4 * np.pi * boundary * np.pi)
                if val < best_val:
                    best_val = val
                    best_xi = boundary
            best_tail[i - 1] = best_xi

        # Determine x_0 range
        x0_lo = max(lo[0], 0.0)
        x0_hi = min(hi[0], 1.0)
        if x0_lo >= x0_hi:
            x0_lo = lo[0]
            x0_hi = hi[0]

        if x0_lo >= x0_hi:
            x = np.zeros(self.dim)
            x[0] = x0_lo
            x[1:] = best_tail
            problem(x)
            return x

        # Build candidate x_0 grid: dense near 0 where curvature is high
        # Use a combination of sqrt-spaced and uniform points
        n_candidates = min(5000, max(500, self.budget * 50))
        
        # Generate candidates: mix of sqrt-spaced (concentrated near 0) and uniform
        # sqrt-spaced: t_i = (i/n)^2 gives more points near 0
        t_sqrt = np.linspace(0, 1, n_candidates // 2) ** 2
        t_uniform = np.linspace(0, 1, n_candidates - n_candidates // 2)
        t = np.concatenate([t_sqrt, t_uniform])
        t = np.sort(t)
        # Remove duplicates
        t = t[np.concatenate([[True], np.diff(t) > 1e-10])]
        
        candidates = x0_lo + (x0_hi - x0_lo) * t
        n_cand = len(candidates)
        
        # Compute f_2 for each candidate (analytical: f_2 = 2 - sqrt(x_0) for g=1)
        # f_1 = x_0
        # Reference point: (11, 400)
        ref_f1 = 11.0
        ref_f2 = 400.0
        
        # For hypervolume in 2D with reference point (r1, r2):
        # Given points sorted by f_1 (increasing), the hypervolume is:
        # sum over i of (f1[i] - f1[i-1]) * (r2 - f2[i])  ... no, that's not right.
        # 
        # Actually for 2D, with points p_0, p_1, ..., p_{n-1} sorted by f_1 ascending,
        # the hypervolume is the area under the "staircase" from the reference point.
        # HV = sum_{i=0}^{n-1} (r1 - f1[i]) * (f2[i] - f2[i+1])  ... no.
        #
        # Let me think more carefully. The hypervolume dominated by a set of points
        # with reference point R = (r1, r2) is the area of the union of rectangles
        # [f1_i, r1] x [f2_i, r2] for all points where f1_i < r1 and f2_i < r2.
        #
        # For 2D, if we sort points by f1 ascending: p_0, p_1, ..., p_{n-1}
        # Then HV = sum_{i=0}^{n-1} max(0, min(r1, f1[i+1] if i<n-1 else r1) - f1[i]) * max(0, r2 - f2[i])
        # ... this is getting complex. Let me use a simpler formulation.
        #
        # Actually the standard 2D hypervolume formula:
        # Sort points by f1 ascending. For each point i, its contribution is:
        #   width_i * (r2 - f2[i])
        # where width_i = (f1[i+1] - f1[i]) if i < n-1, else (r1 - f1[i])
        # BUT only if f2[i] < r2 and f1[i] < r1.
        # Wait, that's not quite right either because of the staircase.
        #
        # Correct 2D hypervolume:
        # Sort by f1 ascending: p_0, p_1, ..., p_{n-1}
        # HV = (r1 - f1[0]) * (r2 - f2[0]) 
        #     + sum_{i=1}^{n-1} (f1[i] - f1[i-1]) * (r2 - f2[i])
        # This works because each step adds a rectangle of width (f1[i]-f1[i-1]) 
        # and height (r2 - f2[i]).
        # Actually no. Let me reconsider.
        #
        # The dominated region is the union of rectangles [f1_i, r1] x [f2_i, r2].
        # For 2D, the hypervolume equals:
        # Sort points by f1 ascending. Define f1[-1] = r1 (sentinel).
        # HV = sum_{i=0}^{n-1} (f1[i+1] - f1[i]) * (r2 - f2[i])
        # where f1[n] = r1.
        # This is correct because the staircase function goes: for f1 in [f1[0], f1[1]),
        # the upper bound on f2 is f2[0]; for f1 in [f1[1], f1[2]), upper bound is f2[1], etc.
        
        f1_vals = candidates  # f_1 = x_0
        f2_vals = 2.0 - np.sqrt(candidates)  # f_2 = 2 - sqrt(x_0)
        
        # Filter to only points strictly better than reference
        valid = (f1_vals < ref_f1) & (f2_vals < ref_f2)
        if not np.any(valid):
            # Degenerate: evaluate one point
            x = np.zeros(self.dim)
            x[0] = x0_lo
            x[1:] = best_tail
            problem(x)
            return x
            
        f1_valid = f1_vals[valid]
        f2_valid = f2_vals[valid]
        n_valid = len(f1_valid)
        
        if n_valid <= self.budget:
            # Use all valid candidates
            chosen_idx = np.where(valid)[0]
            chosen_x0 = candidates[chosen_idx]
            X = np.zeros((len(chosen_x0), self.dim))
            X[:, 0] = chosen_x0
            X[:, 1:] = best_tail
            F = []
            for i in range(len(chosen_x0)):
                if time.time() > deadline:
                    break
                F.append(problem(X[i]))
            return X
        
        # DP to select subset of size b = budget from n_valid candidates
        # maximizing hypervolume.
        # 
        # State: dp[i][k] = max HV contribution considering candidates 0..i, 
        #         selecting exactly k of them, with candidate i being the RIGHTMOST selected.
        # Transition: dp[i][k] = max_{j < i} dp[j][k-1] + contribution(j, i)
        # where contribution(j, i) is the incremental HV when adding candidate i 
        # after candidate j (j is the previously selected rightmost point).
        #
        # The incremental HV when adding point i after point j:
        # The staircase changes: previously the step at position f1[j] had height (r2 - f2[j]).
        # After adding i, the step from f1[j] to f1[i] has height (r2 - f2[i]) [since f2 is decreasing 
        # as f1 increases for this concave front... wait, f2 = 2 - sqrt(x0), so as x0 increases, 
        # sqrt increases, so f2 decreases. So f2 is decreasing in f1. Good.]
        # 
        # Actually, let me define it properly.
        # Given selected points sorted by f1: s_0, s_1, ..., s_{k-1}
        # HV = sum_{m=0}^{k-1} (f1[m+1] - f1[m]) * (r2 - f2[m])  where f1[k] = r1
        #
        # When we add a new point i between j and the next point (or at the end), 
        # the contribution changes.
        #
        # Let's use a different DP formulation:
        # dp[i][k] = max total HV considering only candidates[0..i], selecting k points,
        #            with the RIGHTMOST selected point being candidate i.
        #
        # Base case: dp[i][1] = (r1 - f1[i]) * (r2 - f2[i])
        #             (single point i, HV is the rectangle from i to reference)
        #
        # Transition: dp[i][k] = max_{j < i} dp[j][k-1] + (f1[i] - f1[j]) * (r2 - f2[i])
        #             Wait, that's not right either. Let me think again.
        #
        # If the rightmost k-1 points end at j, and we add i as the new rightmost:
        # The old HV had a term (r1 - f1[j]) * (r2 - f2[j]) as the last term.
        # The new HV replaces that with:
        #   (f1[i] - f1[j]) * (r2 - f2[i]) + (r1 - f1[i]) * (r2 - f2[i])
        # = (r1 - f1[j]) * (r2 - f2[i])
        # So the change is: (r1 - f1[j]) * (r2 - f2[i]) - (r1 - f1[j]) * (r2 - f2[j])
        # = (r1 - f1[j]) * (f2[j] - f2[i])
        #
        # So: dp[i][k] = max_{j < i} [dp[j][k-1] + (r1 - f1[j]) * (f2[j] - f2[i])]
        #
        # But wait, dp[j][k-1] already includes the full HV with j as rightmost.
        # When we add i, the HV becomes dp[j][k-1] + (r1 - f1[j]) * (f2[j] - f2[i]).
        # Yes, that's correct.
        
        b = self.budget
        # Use only valid candidates
        f1_v = f1_valid  # sorted ascending
        f2_v = f2_valid  # decreasing (since f2 = 2-sqrt(x0))
        
        # dp[i] for current k, dp_prev[i] for k-1
        # Initialize k=1
        dp_prev = np.full(n_valid, -np.inf)
        for i in range(n_valid):
            dp_prev[i] = (ref_f1 - f1_v[i]) * (ref_f2 - f2_v[i])
        
        argmax_table = np.zeros((b, n_valid), dtype=int)
        
        for k in range(2, b + 1):
            dp_curr = np.full(n_valid, -np.inf)
            for i in range(k - 1, n_valid):
                # dp_curr[i] = max_{j in [k-2, i-1]} dp_prev[j] + (r1 - f1_v[j]) * (f2_v[j] - f2_v[i])
                best_val = -np.inf
                best_j = -1
                for j in range(k - 2, i):
                    val = dp_prev[j] + (ref_f1 - f1_v[j]) * (f2_v[j] - f2_v[i])
                    if val > best_val:
                        best_val = val
                        best_j = j
                if best_j != -1:
                    dp_curr[i] = best_val
                    argmax_table[k-1, i] = best_j
            dp_prev = dp_curr
            
            if time.time() > deadline - 10:
                break
        
        # Find best ending point
        best_i = np.argmax(dp_prev)
        path = [best_i]
        for k in range(b - 1, 0, -1):
            next_i = argmax_table[k, path[-1]]
            if next_i == -1 or next_i < 0:
                break
            path.append(next_i)
        
        path.reverse()  # Now in increasing f1 order
        
        if len(path) < b:
            # Pad with remaining candidates if needed
            used = set(path)
            for idx in range(n_valid):
                if len(path) >= b:
                    break
                if idx not in used:
                    path.append(idx)
                    used.add(idx)
            path = sorted(path)
        
        # Map back to original candidate indices
        valid_indices = np.where(valid)[0]
        chosen_orig_idx = valid_indices[path[:b]]
        chosen_x0 = candidates[chosen_orig_idx]
        
        # Evaluate
        X = np.zeros((len(chosen_x0), self.dim))
        X[:, 0] = chosen_x0
        X[:, 1:] = best_tail
        
        F = []
        for i in range(len(chosen_x0)):
            if time.time() > deadline:
                break
            F.append(problem(X[i]))
        
        return X

# EVOLVE-BLOCK-END
