import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []
        self.eval_count = 0

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = self.start_time + 280  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT6 Analysis:
        # f0(x) = 1 - exp(-4 * x0 * sum_{i=1}^{m-1} |2xi sin(10 pi xi x0)|)
        # f1(x) = (1 + 9 * sum(x[1:]) / (m-1))^0.5 * (1 - sqrt(f0(x)))
        #
        # Pareto Set: x1 = ... = x9 = 0.
        # On the Pareto Set:
        # f0(x0) = 1 - exp(-40 * pi * x0 * |sin(10 * pi * x0)|)
        # f1(x0) = 1 - sqrt(f0(x0))
        #
        # We want to maximize hypervolume. The reference point is [11, 11].
        # The front lies in [0, 1] x [0, 1].
        # Hypervolume is dominated by the area under the curve f1(f0).
        # Since f0 is not monotonic over [0,1] due to |sin|, we should sample
        # points that cover the range of f0 efficiently.
        #
        # Actually, f0(x0) oscillates. 
        # Let's analyze f0(x0) = 1 - exp(-40*pi*x0*|sin(10*pi*x0)|).
        # Let g(x0) = 40*pi*x0*|sin(10*pi*x0)|.
        # As x0 goes from 0 to 1, sin(10*pi*x0) oscillates 10 times.
        # The term x0*|sin(10*pi*x0)| increases roughly linearly with x0 on average,
        # but has dips at sin=0.
        # At x0=0, g=0, f0=0.
        # At x0=1, g=40*pi*1*0 = 0? No, sin(10*pi)=0. So f0(1)=0.
        # Wait, ZDT6 definition:
        # Usually ZDT6 has x0 in [0,1].
        # f0 = 1 - exp(-4 x0 sum |2 xi sin(10 pi xi x0)|)
        # If x[1:]=0, sum is 0. So f0 = 1 - exp(0) = 0?
        # Let's check standard ZDT6 definition.
        # ZDT6:
        # m=10, n=10.
        # f1(x) = 1 - exp(-4 x1 sum_{i=2}^n |2 xi sin(10 pi xi x1)|)
        # f2(x) = (1 + 9 (sum_{i=2}^n xi)/(n-1))^0.5 (1 - sqrt(f1(x)))
        #
        # If x[1:] = 0 (indices 1 to 9 are 0), then the sum is 0.
        # Then f1(x) = 1 - exp(0) = 0.
        # Then f2(x) = 1 * (1 - 0) = 1.
        # So the only point on the "Pareto set" defined by x[1:]=0 is (0,1).
        # This is incorrect. The Pareto set of ZDT6 is NOT x[1:]=0.
        #
        # Correction:
        # For ZDT6, the Pareto set is defined by x_i = 0 for i=2..n?
        # Let's check the literature.
        # ZDT6 is a problem with a non-uniform distribution of points along the Pareto front.
        # The Pareto set is x_2 = ... = x_n = 0.
        # If x_2=...=x_n=0, then sum_{i=2}^n |2 x_i sin(10 pi x_i x_1)| = 0.
        # Then f1 = 1 - exp(0) = 0.
        # This implies f1 is constant 0? That can't be right for a multi-objective problem.
        #
        # Let's re-read the ZDT6 definition carefully.
        # f1(x) = 1 - exp(-4 x_1 \sum_{i=2}^{n} |2 x_i \sin(10 \pi x_i x_1)|)
        # If x_i = 0 for i>1, the sum is 0.
        # So f1 = 0.
        # f2 = (1 + 0)^0.5 (1 - sqrt(0)) = 1.
        # So (0,1) is a point.
        #
        # Is there any other x that gives f1 > 0?
        # Yes, if any x_i > 0, the sum is > 0, so exp(-...) < 1, so f1 > 0.
        # But if x_i > 0, f2 increases because of the (1 + 9 * sum/(n-1))^0.5 term.
        # So increasing x_i increases both f1 and f2?
        # If both increase, then (0,1) dominates all other points?
        # (0,1) has f1=0, f2=1.
        # A point with f1>0 and f2>1 is dominated by (0,1)? No, (0,1) has lower f1 but lower f2.
        # Wait, minimization.
        # (0,1) vs (f1'>0, f2'>1).
        # (0,1) is better in f1. (f1', f2') is worse in f1.
        # (0,1) is better in f2? No, 1 < f2'. So (0,1) is better in f2.
        # So (0,1) dominates everything?
        #
        # Let's check the function again.
        # f2(x) = (1 + 9 \frac{\sum_{i=2}^n x_i}{n-1})^{0.5} (1 - \sqrt{f_1(x)})
        # If x_i=0, sum=0, factor=1. f1=0, sqrt=0, f2=1.
        # If we increase x_i, sum increases, factor increases.
        # f1 increases (exp becomes smaller, 1-exp becomes larger).
        # sqrt(f1) increases. (1 - sqrt(f1)) decreases.
        # So f2 is product of increasing factor and decreasing term.
        #
        # Let's test a point. x1=0.5, x2=0.1, rest 0.
        # Sum = |2*0.1*sin(10*pi*0.1*0.5)| = 0.2*|sin(0.5*pi)| = 0.2*1 = 0.2.
        # f1 = 1 - exp(-4*0.5*0.2) = 1 - exp(-0.4) = 1 - 0.67 = 0.33.
        # f2 = (1 + 9*0.1/9)^0.5 * (1 - sqrt(0.33)) = (1.1)^0.5 * (1 - 0.57) = 1.05 * 0.43 = 0.45.
        # Point: (0.33, 0.45).
        # Compare with (0,1).
        # 0.33 > 0 (worse f1).
        # 0.45 < 1 (better f2).
        # So neither dominates.
        #
        # So the Pareto front is NOT just x[1:]=0.
        # The Pareto set for ZDT6 is actually x_2=...=x_n=0 is WRONG.
        # The Pareto set is defined by x_i = 0 for i=2..n?
        # Many sources say the Pareto set of ZDT6 is x_2=...=x_n=0.
        # But my analysis shows that if x_2..x_n=0, f1=0.
        # Let's check the ZDT6 paper or standard definition.
        # De Jong et al. 2002.
        # ZDT6:
        # f1(x) = 1 - exp(-4 x1 sum_{i=2}^n |2 xi sin(10 pi xi x1)|)
        # f2(x) = (1 + 9 (sum_{i=2}^n xi)/(n-1))^0.5 (1 - sqrt(f1(x)))
        #
        # If x2..xn=0, sum=0, f1=0, f2=1.
        # This suggests (0,1) is the only point on the "boundary" defined by zeros.
        # However, if we vary x1, and keep x2..xn=0, f1 stays 0.
        # So the line segment x1 in [0,1], x2..xn=0 maps to the single point (0,1).
        #
        # This implies the Pareto front is formed by varying x1 AND x2..xn.
        # Actually, the Pareto set of ZDT6 is often described as having x_i=0 for i>1.
        # Why?
        # Maybe I am misinterpreting the sum.
        # If the sum is 0, f1=0.
        # To get f1>0, we need sum>0.
        # To minimize f2, we want small sum (small factor) and large f1 (small 1-sqrt(f1)).
        # There is a trade-off.
        #
        # However, it is a known property that for ZDT6, the Pareto set is x_2=...=x_n=0.
        # Let's re-evaluate f1 when x2..xn=0.
        # Sum is 0. f1=0.
        # This is a constant.
        #
        # Wait, is it possible that the problem definition in the task is slightly different?
        # "minimizing all 2 objectives of ZDT6"
        #
        # Let's look at the inspiration.
        # Inspiration 2: "pre-compute the exact analytic Pareto front of ZDT6 offline"
        # Inspiration 3: "compute the analytic Pareto front directly"
        #
        # If the Pareto set is x2..xn=0, then f1=0 for all points.
        # Then the front is a single point (0,1).
        # This would make the problem trivial and hypervolume 0 (or just that point).
        # But the score is 1.46, implying a non-trivial front.
        #
        # Let's check another definition of ZDT6.
        # Some definitions use:
        # f1(x) = 1 - exp(-4 x1 \sum_{i=2}^n |2 x_i \sin(10 \pi x_i x_1)|)
        # This is the standard one.
        #
        # Is it possible that the Pareto set is NOT x2..xn=0?
        # Yes. The Pareto set of ZDT6 is complex.
        # However, it is often approximated or the "ideal" front is generated by varying x1 and setting other variables to specific values?
        #
        # Actually, let's look at the behavior.
        # If we set x2..xn=0, we get (0,1).
        # If we set x1=1, x2=1, rest 0.
        # Sum = |2*1*sin(10*pi*1*1)| = 0. f1=0. f2=1.
        #
        # If we set x1=0.5, x2=0.5.
        # Sum = |2*0.5*sin(10*pi*0.5*0.5)| = 1*|sin(2.5*pi)| = 1*1 = 1.
        # f1 = 1 - exp(-4*0.5*1) = 1 - exp(-2) = 0.86.
        # f2 = (1 + 9*0.5/9)^0.5 * (1 - sqrt(0.86)) = (1.5)^0.5 * (1 - 0.93) = 1.22 * 0.07 = 0.085.
        # Point: (0.86, 0.085).
        # This is much better in f2 than (0,1).
        #
        # So the Pareto front extends from (0,1) to somewhere near (1,0).
        # The points are generated by choosing x1 and x2..xn.
        #
        # Strategy:
        # We cannot easily invert the full multi-variable function.
        # However, we can observe that for a fixed x1, the term sum |2 xi sin(10 pi xi x1)| is minimized when xi=0?
        # If xi=0, term is 0.
        # But we saw that xi=0 leads to f1=0.
        # We want to maximize f1 (to minimize 1-sqrt(f1)) while minimizing the sum (to minimize the factor).
        #
        # Let's try a different approach.
        # The known Pareto front of ZDT6 is often sampled by varying x1 from 0 to 1 and setting x2..xn=0?
        # NO, that gives (0,1).
        #
        # Let's check the "Pareto set" description in literature again.
        # "The Pareto set of ZDT6 is x2=...=xn=0."
        # This statement appears in many papers.
        # If this is true, then f1 must be non-constant when x2..xn=0.
        # How?
        # f1 = 1 - exp(-4 x1 * 0) = 0.
        #
        # There is a contradiction between my calculation and the literature statement.
        # Let's check the sum index.
        # Sum_{i=2}^n.
        # If x2..xn=0, sum is 0.
        #
        # Maybe the definition is:
        # f1(x) = 1 - exp(-4 x1 \sum_{i=2}^n |2 x_i \sin(10 \pi x_i)|) ?
        # No, standard is x_i x_1.
        #
        # Let's assume the literature is right and I am missing something.
        # Or maybe the literature is wrong?
        # Or maybe the "Pareto set" refers to the set of x that maps to the front, and for ZDT6, it is indeed x2..xn=0, BUT the function is different?
        #
        # Let's try to find a point with x2..xn=0 that is not (0,1).
        # Impossible with the given formula.
        #
        # Hypothesis: The "Pareto set" description in literature might be for ZDT1-5, or I am misremembering.
        # For ZDT6, the front is non-uniform.
        #
        # Let's use a heuristic search that is smarter than random.
        # We know the front goes from (0,1) to approx (1,0).
        # We can parameterize the front.
        #
        # Alternative Strategy:
        # Use the budget to perform a grid search or refined sampling in the space of x1 and a representative x2.
        # Since n=10, and the function is symmetric in x2..xn (mostly), we can set x2=...=xk = v and xk+1..xn=0?
        #
        # Let's try a specific structure.
        # Set x2 = x3 = ... = x10 = v.
        # Then sum = 9 * |2 v sin(10 pi v x1)|.
        # f1 = 1 - exp(-36 pi v x1 |sin(10 pi v x1)|).
        # f2 = (1 + 9 * 9v / 9)^0.5 * (1 - sqrt(f1)) = (1 + 9v)^0.5 * (1 - sqrt(f1)).
        #
        # Now we have a 2D problem: x1 in [0,1], v in [0,1].
        # We can sample this 2D space to find the Pareto front.
        #
        # With 200 evaluations, we can do a 14x14 grid (196 points).
        # This should cover the front well.
        
        # Grid search on x1 and v
        n_grid = 14
        x1_vals = np.linspace(0, 1, n_grid)
        v_vals = np.linspace(0, 1, n_grid)
        
        points = []
        for x1 in x1_vals:
            for v in v_vals:
                x = np.zeros(self.dim)
                x[0] = x1
                x[1:] = v
                points.append(x)
        
        # Evaluate all
        for x in points:
            if self.eval_count >= self.budget:
                break
            if time.time() > deadline:
                break
            f = self._evaluate(problem, x)
            self._update_archive(x, f)
            
        return self.archive

    def _evaluate(self, problem, x):
        self.eval_count += 1
        return np.array(problem(x))

    def _update_archive(self, x, f):
        # Check if x is dominated by any in archive
        for ax, af in self.archive:
            if np.all(f >= af) and np.any(f > af):
                return # Dominated, skip
        
        # Remove any in archive dominated by x
        self.archive = [(ax, af) for ax, af in self.archive if not (np.all(f <= af) and np.any(f < af))]
        self.archive.append((x, f))
