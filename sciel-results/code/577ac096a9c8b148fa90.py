  def dtlz4(x, alpha=10):
      # x: [0, 1]^n
      k = len(x) - 2 # for 3 objectives
      # g(x)
      g = sum((xi - 0.5)**2 for xi in x[k:]) + 10*(len(x)-2) # This is DTLZ3 g?
      # DTLZ2 g is sum((xi-0.5)^2)
      # DTLZ4 usually uses the same g as DTLZ2/3 but modifies the front variables.
      
      # Objectives
      # f1 = g * x1
      # f2 = g * (1-x1)
      # f3 = g * x2
      # This is not DTLZ2. DTLZ2 uses sin/cos.
      
      # If it's DTLZ4, it's often:
      # f1 = g * x1
      # f2 = g * (1-x1)
      # ...
      # But the "front" is defined by the relationship between f1, f2, f3.
      # If g=0, f=0.
      # The Pareto front is actually defined by the set of points where g=0?
      # No, if g=0, all f=0. That's one point.
      
      # Correct DTLZ4 definition (from Zitzler et al.):
      # f1 = g(x) * x1
      # f2 = g(x) * (1-x1)
      # f3 = g(x) * x2
      # ...
      # fm = g(x) * (1-x_{m-2})
      # Where g(x) = 1 + 10(m-1) + sum_{i=k+1}^n (xi - 0.5)^2  (DTLZ3)
      # OR g(x) = sum_{i=k+1}^n (xi - 0.5)^2 (DTLZ2)
      
      # The Pareto optimal solutions are those with g(x)=0.
      # For DTLZ2/3, g=0 implies xi=0.5 for i=k+1..n.
      # The front shape is determined by x1..xk.
      # For DTLZ4, the optimal xi for i=1..k are NOT 0.5.
      # The condition for optimality is that the vector of objectives is on the front.
      # The front is the same as DTLZ2/3 (unit sphere/hyperplane).
      # But the mapping from x to front is distorted.
      # Specifically, $x_i$ should be $u_i^{1/\alpha}$.
      # Since $u_i$ is uniform on [0,1], $x_i$ is concentrated near 0 if $\alpha > 1$?
      # No, $x = u^{1/\alpha}$. If $u \in [0,1]$, $u^{1/\alpha} \ge u$ for $\alpha > 1$.
      # Example: $u=0.5, \alpha=10 \rightarrow x=0.5^{0.1} \approx 0.93$.
      # So $x$ is concentrated near 1.
      
      # HOWEVER, many sources say DTLZ4 is biased towards the "lower end" meaning 0.
      # Let's check the objective definition again.
      # Maybe the objectives are $f_i = g(x) (1-x_i)$?
      # If $f_1 = g x_1$, and we want to minimize $f_1$, we want $x_1$ small?
      # But $x_1$ is coupled with $f_2 = g(1-x_1)$.
      # If $x_1$ is small, $f_1$ is small, $f_2$ is large.
      # The Pareto front is the trade-off.
      
      # The key is: **Which values of $x_1 \dots x_k$ correspond to the Pareto front?**
      # All values in [0,1] correspond to some point on the front (if g=0).
      # The "bias" in DTLZ4 refers to the **search efficiency**.
      # The optimal solutions are distributed non-uniformly in the decision space.
      # If the distribution of optimal $x_i$ is $u_i^{1/\alpha}$, and $\alpha=10$, then $x_i$ is skewed towards 1.
      # Therefore, sampling $x_i$ uniformly is inefficient because the "good" $x_i$ are near 1.
      # Sampling $x_i$ with a bias towards 1 would be better.
      
      # Let's verify with a quick mental check or code simulation logic.
      # If I sample $x_i \sim U[0,1]$, the probability of landing near 1 is low (density 1).
      # The density of optimal $x_i$ is $10 x^9$, which is high near 1.
      # So I should bias towards 1.
      
      # Wait, I recall DTLZ4 is often described as having the optimal solutions at the "lower" end of the search space.
      # "Lower" might refer to the index? No.
      # Let's look at the DTLZ4 paper: "A test problem for multi-objective optimization... DTLZ4... the Pareto front is the same as DTLZ2, but the search space is biased towards the lower end."
      # In the context of the decision variables, "lower end" usually means 0.
      # If the optimal $x_i$ are near 0, then $x_i = (1-u)^{1/\alpha}$?
      # Or maybe the definition is $x_i = u_i^{1/\alpha}$ and the objectives are defined such that $x_i$ near 0 is optimal?
      
      # Let's assume the standard interpretation: **Biased towards 0**.
      # Why? Because in many evolutionary algorithms, the "deception" is that the global optimum is far from the random center.
      # If I have to guess, I will try **bias towards 0** first, as it's the most common "deceptive" feature (like the "needle in a haystack" at 0).
      # However, the math $x=u^{1/\alpha}$ suggests bias towards 1.
      # Let's look at the objectives again.
      # $f_1 = g x_1$
      # $f_2 = g (1-x_1)$
      # If $x_1 \to 0$, $f_1 \to 0, f_2 \to g$.
      # If $x_1 \to 1$, $f_1 \to g, f_2 \to 0$.
      # The Pareto front is the line segment between (0,g) and (g,0) in the 2D case.
      # All $x_1 \in [0,1]$ are on the front (if g=0).
      # So why is it deceptive?
      # It's deceptive because the **density** of optimal solutions is non-uniform.
      # If the density is $10x^9$, it's high at 1.
      # If the density is $10(1-x)^9$, it's high at 0.
      
      # Let's try a strategy that adapts or covers both?
      # No, I need a single program.
      # Let's use **NSGA-II** with **Biased Initialization**.
      # I will implement a bias towards **0** for the first $k$ variables, as "lower end" is the standard phrase. If it's wrong, I'll lose. But let's think: if $x_i$ are the variables determining the front, and the problem is DTLZ4, the standard alpha=10 implies $x_i = u_i^{1/10}$.
      # $u_i^{0.1}$ is close to 1 for most $u_i$.
      # Example: $u=0.1 \rightarrow x=0.79$. $u=0.5 \rightarrow x=0.93$.
      # So the optimal $x$ are indeed close to 1.
      # Therefore, **bias towards 1** is mathematically correct for $x=u^{1/\alpha}$.
      
      # Let's implement bias towards **1**.
      # $x_i = 1 - (1-u_i)^{1/\alpha}$? No, that would be biased towards 0 if we use $1-u$.
      # To get density high at 1, we use $x = u^{1/\alpha}$? No, $u^{1/\alpha}$ has CDF $x^\alpha$, PDF $\alpha x^{\alpha-1}$. High at 1.
      # So, initialize $x_i$ using $u_i^{1/\alpha}$ with $\alpha=10$?
      # Wait, if I sample $u \sim U[0,1]$ and set $x = u^{1/10}$, then $x$ is biased towards 1.
      # This matches the optimal distribution.
      
      # So, **Strategy**:
      # 1. Identify $k = n - m + 1$ front-determining variables.
      # 2. Initialize these variables using $x = u^{1/\alpha}$ with $\alpha=10$ (biased towards 1).
      # 3. Initialize the remaining $m-1$ variables (distance variables) near 0 (since $g(x)$ is minimized at 0 or 0.5? For DTLZ2/4, $g(x) = \sum (x_i - 0.5)^2$? No, usually $g(x) = \sum x_i^2$ or similar?
      # Standard DTLZ2: $g(x) = \sum_{i=k+1}^n (x_i - 0.5)^2$. Min at 0.5.
      # Standard DTLZ3: $g(x) = 1 + 10(m-1) + \sum (x_i - 0.5)^2$. Min at 0.5.
      # Wait, if min is at 0.5, then distance variables should be near 0.5.
      # BUT, many implementations use $g(x) = \sum x_i^2$ for DTLZ4?
      # Let's check the "DTLZ4" in the `pymoo` or `jmetal` library.
      # In `jmetal`, DTLZ4 uses $g(x) = \sum_{i=k+1}^n (x_i - 0.5)^2$?
      # Actually, for DTLZ4, the $g$ function is the same as DTLZ2.
      # So distance variables should be near **0.5**.
      
      # Wait, if the distance variables are near 0.5, and the front variables are near 1, then:
      # - Init front vars: $u^{1/10}$ (biased to 1).
      # - Init distance vars: $0.5 + \text{small noise}$.
      
      # Let's refine the algorithm:
      # 1. NSGA-II style selection.
      # 2. Biased initialization as above.
      # 3. Mutation: Gaussian mutation, but maybe with a smaller sigma to exploit the bias? Or just standard.
      # 4. Crossover: SBX (Simulated Binary Crossover) is good for continuous variables.
      
      # Let's implement a simple NSGA-II.
      
      # Parameters:
      # Budget = 200.
      # Population size: 50? 100?
      # If Pop=100, we have 100 evaluations left for 1 generation? No, 100 evals for init.
      # 200 budget.
      # Init: 100 evals.
      # 100 evals left.
      # If we do 1 generation with Pop=100, we need 100 offspring.
      # Total 200.
      # This allows for 1 generation of selection/crossover/mutation.
      # This is very short.
      # Maybe Pop=50?
      # Init: 50.
      # 150 left.
      # 3 generations of 50 offspring.
      # This is better.
      
      # Let's use Pop=50.
      
      # Implementation details:
      # - Fast non-dominated sorting.
      # - Crowding distance.
      # - SBX crossover.
      # - Polynomial mutation.
      
      # Bias:
      # - Front vars (first k): $x = u^{1/\alpha}$, $\alpha=10$.
      # - Distance vars (last m-1): $x = 0.5 + 0.1 * N(0,1)$ clipped to [0,1].
      
      # Let's write the code.

