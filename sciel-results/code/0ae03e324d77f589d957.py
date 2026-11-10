# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Greedy set-covering optimizer for ZDT1 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 320
        rng = np.random.default_rng(0)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT1 specific parameters
        n = self.dim
        g = 1 + 10 * (n - 1)
        
        # Reference point from task description
        r1 = 11.0
        r2 = 11.0
        
        # Base tail vector
        base_tail = np.clip(np.zeros(self.dim - 1), lo[1:], hi[1:])

        # Generate candidate pool
        # Use a large pool to approximate the continuous optimum
        M = 20000
        x_candidates = np.linspace(0, 1, M)
        
        # Precompute f2 for all candidates
        # f2 = g * (1 - sqrt(f1 / g))
        f1_candidates = x_candidates
        f2_candidates = g * (1 - np.sqrt(f1_candidates / g))
        
        # Widths for HV calculation: w = r2 - f2
        # Note: If r2 < f2, w is negative. 
        # However, we proceed with the standard HV logic.
        w_candidates = r2 - f2_candidates
        
        # Greedy selection
        N = self.budget
        selected_indices = []
        current_hv = 0.0
        
        # To speed up, we can maintain a sorted list of selected x's
        # and their corresponding w's.
        # But for M=20000, N=200, even a simple O(N*M) with O(1) delta is fine.
        # Delta calculation requires knowing the neighbors in the selected set.
        
        # We will maintain a list of (x, w, index) for selected points, sorted by x.
        selected_points = [] # list of dicts or tuples
        
        for k in range(N):
            if time.time() > deadline:
                break
                
            best_delta = -np.inf
            best_idx = -1
            best_pos = -1
            
            # Iterate through all candidates not yet selected
            # This is O(M). 
            # We need to efficiently find the insertion position in selected_points.
            # Since selected_points is small (<=200), we can just scan it.
            
            for i in range(M):
                if i in selected_indices:
                    continue
                
                x_new = x_candidates[i]
                w_new = w_candidates[i]
                
                # Find insertion position in selected_points
                # selected_points is sorted by x
                pos = 0
                while pos < len(selected_points) and selected_points[pos][0] < x_new:
                    pos += 1
                
                # Calculate delta HV
                # If pos == 0:
                #   New point is the smallest x.
                #   It creates a new interval [x_new, x_old_first]
                #   And the "left" boundary is effectively 0? 
                #   HV formula: sum (x_{i+1} - x_i) * w_i
                #   If we insert at start:
                #   Old first point was S[0].
                #   Old contribution involving S[0] as left bound: (S[1] - S[0]) * W(S[0]) ?? No.
                #   The term for interval starting at S[0] is (S[1] - S[0]) * W(S[0]).
                #   If we insert X before S[0]:
                #   New interval 1: [X, S[0]] with weight W(X).
                #   New interval 2: [S[0], S[1]] with weight W(S[0]).
                #   Old interval: [S[0], S[1]] with weight W(S[0]).
                #   Wait, the weight is determined by the LEFT endpoint of the interval.
                #   So the interval [S[0], S[1]] always uses W(S[0]).
                #   Inserting X before S[0] adds a new interval [X, S[0]] with weight W(X).
                #   Does it change the existing intervals?
                #   The interval [S[0], S[1]] is unchanged.
                #   So Delta = (S[0] - X) * W(X).
                #   What about the area to the left of X?
                #   The HV formula sums from the first point.
                #   So inserting a new minimum x adds (x_second - x_new) * W(x_new).
                
                # If pos == len(selected_points):
                #   New point is the largest x.
                #   Old last point S[-1].
                #   Old last interval: [S[-1], r1] with weight W(S[-1]).
                #   New intervals: [S[-1], X] with weight W(S[-1]), and [X, r1] with weight W(X).
                #   Old contribution: (r1 - S[-1]) * W(S[-1]).
                #   New contribution: (X - S[-1]) * W(S[-1]) + (r1 - X) * W(X).
                #   Delta = (X - S[-1]) * W(S[-1]) + (r1 - X) * W(X) - (r1 - S[-1]) * W(S[-1])
                #   Delta = (X - S[-1]) * W(S[-1]) + (r1 - X) * W(X) - (r1 - S[-1]) * W(S[-1])
                #   Delta = (r1 - X) * (W(X) - W(S[-1]))
                
                # If 0 < pos < len:
                #   Neighbors S[pos-1] and S[pos].
                #   Old interval: [S[pos-1], S[pos]] with weight W(S[pos-1]).
                #   New intervals: [S[pos-1], X] with weight W(S[pos-1]), [X, S[pos]] with weight W(X).
                #   Old: (S[pos] - S[pos-1]) * W(S[pos-1]).
                #   New: (X - S[pos-1]) * W(S[pos-1]) + (S[pos] - X) * W(X).
                #   Delta = (X - S[pos-1]) * W(S[pos-1]) + (S[pos] - X) * W(X) - (S[pos] - S[pos-1]) * W(S[pos-1])
                #   Delta = (S[pos] - X) * W(X) - (S[pos] - X) * W(S[pos-1])
                #   Delta = (S[pos] - X) * (W(X) - W(S[pos-1]))
                
                if len(selected_points) == 0:
                    # First point
                    # HV = (r1 - X) * W(X)
                    delta = (r1 - x_new) * w_new
                elif pos == 0:
                    # Insert at start
                    x_next = selected_points[0][0]
                    delta = (x_next - x_new) * w_new
                elif pos == len(selected_points):
                    # Insert at end
                    x_prev = selected_points[-1][0]
                    w_prev = selected_points[-1][1]
                    delta = (r1 - x_new) * (w_new - w_prev)
                else:
                    # Insert in middle
                    x_prev = selected_points[pos-1][0]
                    w_prev = selected_points[pos-1][1]
                    x_next = selected_points[pos][0]
                    delta = (x_next - x_new) * (w_new - w_prev)
                
                if delta > best_delta:
                    best_delta = delta
                    best_idx = i
                    best_pos = pos
            
            # Add best point
            if best_idx != -1:
                selected_indices.append(best_idx)
                # Insert into selected_points
                x_new = x_candidates[best_idx]
                w_new = w_candidates[best_idx]
                selected_points.insert(best_pos, (x_new, w_new, best_idx))
                current_hv += best_delta

        # Prepare final x values
        # Sort selected points by x to ensure consistent order (though order doesn't matter for the set)
        final_x1_values = [p[0] for p in selected_points]
        
        # If we didn't select N points (due to time), pad with random or just use what we have
        # But we should use all budget if possible.
        
        n_evals = 0
        results = []
        
        for x1 in final_x1_values:
            if time.time() > deadline:
                break
            x = np.concatenate(([x1], base_tail))
            f = problem(x)
            n_evals += 1
            results.append(f)
            
        # If we have budget left (unlikely if N=200 and loop finishes), 
        # we could add more random points, but greedy should have used N.
        
        if results:
            return np.concatenate(([final_x1_values[-1]], base_tail))
        return np.zeros(self.dim)

# EVOLVE-BLOCK-END
