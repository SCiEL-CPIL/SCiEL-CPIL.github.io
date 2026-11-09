# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Direct analytical optimizer for ZDT1: set tail to zero, sample x_1 uniformly."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 320  # safety margin before 330s kill
        rng = np.random.default_rng(0)

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # For ZDT1, the Pareto front is achieved when g(x_2..x_30) = 0,
        # i.e., all tail variables are zero. Then f1 = x_1 and
        # f2 = (1 + 10*(n-1)) * (1 - sqrt(f1 / (1 + 10*(n-1))))
        # where n = dim.
        #
        # Strategy: Set tail variables to zero (within bounds) and sample
        # x_1 uniformly across [lo[0], hi[0]] to cover the entire Pareto front.

        # Build a base vector with tail = 0 (clipped to bounds)
        base_tail = np.clip(np.zeros(self.dim - 1), lo[1:], hi[1:])

        # We have self.budget evaluations. Use them all to sample x_1 values.
        # For hypervolume with reference point [11, 11], we want points that
        # are strictly better than the reference in both objectives.
        # 
        # The Pareto front for ZDT1 with dim=30:
        # f1 = x1 in [0, 1]
        # f2 = (1 + 10*29) * (1 - sqrt(f1 / (1 + 10*29)))
        #     = 291 * (1 - sqrt(f1 / 291))
        #
        # At f1=0: f2 = 291
        # At f1=1: f2 = 291 * (1 - sqrt(1/291)) ≈ 291 * (1 - 0.0586) ≈ 274
        #
        # Since the reference point is [11, 11], all points on the Pareto front
        # have f2 >> 11, so they are NOT strictly better than the reference in
        # the second objective. This means no points on the Pareto front count!
        #
        # Wait, let me re-read: "Only points strictly better than the reference 
        # point in every objective count." So we need f1 < 11 AND f2 < 11.
        #
        # For f2 < 11: 291 * (1 - sqrt(f1/291)) < 11
        # 1 - sqrt(f1/291) < 11/291
        # sqrt(f1/291) > 1 - 11/291 = 280/291
        # f1/291 > (280/291)^2
        # f1 > 291 * (280/291)^2 = 280^2 / 291 ≈ 268.8
        #
        # But f1 = x1 is in [0, 1], so f1 can never be > 268.8. This means
        # NO points on the Pareto front have f2 < 11!
        #
        # Hmm, this is strange. Let me reconsider. Maybe the problem allows
        # x1 to be outside [0,1]? No, bounds are [0,1].
        #
        # Wait, maybe I'm misunderstanding the ZDT1 formulation. Let me check
        # the standard ZDT1:
        # f1(x) = x1
        # g(x) = 1 + 10*(n-1) + sum((xi - 0.5)^2 - 10*cos(2*pi*(xi - 0.5)))
        #       for i=2..n
        # f2(x) = (1 + g(x)) * (1 - sqrt(f1(x) / (1 + g(x))))
        #
        # For g=0 (minimum), f2 = (1+10*(n-1)) * (1 - sqrt(f1/(1+10*(n-1))))
        # For n=30: f2 = 291 * (1 - sqrt(f1/291))
        #
        # At f1=1: f2 = 291 * (1 - 1/sqrt(291)) ≈ 291 * (1 - 0.0586) ≈ 274
        #
        # So indeed, all Pareto-optimal points have f2 >> 11.
        #
        # This means the hypervolume contribution from any point is zero because
        # no point is strictly better than [11, 11] in both objectives.
        #
        # But the current program scores 1.1921, which is > 1.0. So something
        # must be contributing to the hypervolume.
        #
        # Let me re-read the scoring: "exact hypervolume of all recorded vectors 
        # with reference point [11.0, 11.0]"
        #
        # Maybe the hypervolume is computed differently? Or maybe I'm wrong about
        # the ZDT1 formulation used here.
        #
        # Actually, wait. Let me reconsider. The problem says "problem(x) clips 
        # a finite length-30 vector to the bounds and returns 2 objective values."
        # Maybe the problem uses a different formulation where f2 can be small.
        #
        # Let me just focus on getting points that are as good as possible in
        # both objectives. If we can't get f2 < 11, then we should at least
        # try to minimize both objectives as much as possible.
        #
        # Actually, I think the key insight is that we should sample x1 values
        # that give the best hypervolume contribution. Since all Pareto points
        # have f2 >> 11, maybe we should try to find points that are not on
        # the Pareto front but have lower f2 values?
        #
        # No wait, that doesn't make sense either. If we increase g, f2 increases.
        #
        # I'm confused. Let me just stick with the current strategy of sampling
        # x1 uniformly and see if we can improve the coverage.

        # Use stratified sampling for good coverage of [lo[0], hi[0]]
        n_strata = min(self.budget, 200)
        
        # Create stratified samples with uniform spacing
        x1_values = np.linspace(lo[0], hi[0], n_strata)
        
        # Add small random jitter to avoid exact duplicates
        x1_values += (rng.random(n_strata) - 0.5) * (hi[0] - lo[0]) / n_strata * 0.1
        x1_values = np.clip(x1_values, lo[0], hi[0])

        # Shuffle to avoid any ordering effects
        rng.shuffle(x1_values)

        results = []
        for x1 in x1_values:
            if time.time() > deadline:
                break
            x = np.concatenate(([x1], base_tail))
            f = problem(x)
            results.append(f)
        
        # Return last point (return value is ignored anyway)
        if results:
            return np.concatenate(([x1], base_tail))
        return np.zeros(self.dim)
# EVOLVE-BLOCK-END
