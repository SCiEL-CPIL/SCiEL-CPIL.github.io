import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 135.0  # Safety margin below 150s kill
_START = time.time()

def remaining_time():
    return DEADLINE - (time.time() - _START)

def compute_value(h):
    """Compute the raw value (lower is better)."""
    if not np.all(np.isfinite(h)):
        return float('inf')
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def construct():
    """
    Convex optimization approach.
    
    The objective is to minimize t such that:
    1. max_k (h * (1-h))[k] <= t * N / 2
    2. sum(h) = N / 2
    3. 0 <= h_i <= 1
    
    The correlation c = correlate(h, 1-h, 'full') is a quadratic form in h.
    Specifically, c[k] = sum_i h[i] * (1-h)[i+k] for valid indices.
    Since (1-h) = 1 - h, c[k] = sum h[i] - sum h[i]h[i+k].
    This is not convex in h directly because of the -h*h term.
    
    Wait, the task is to minimize the MAX of the correlation.
    Let f_k(h) = correlate(h, 1-h, mode='full')[k].
    We want to minimize t subject to f_k(h) <= t for all k.
    f_k(h) is a quadratic function. If the quadratic form is indefinite, the set {h | f_k(h) <= t} is not convex.
    
    Let's check the structure of f_k(h).
    f_k(h) = sum_i h_i (1 - h_{i+k}) = sum_i h_i - sum_i h_i h_{i+k}.
    This is a quadratic form with a negative definite part (the -h_i h_j terms) and linear terms.
    Minimizing the maximum of such functions is generally non-convex.
    
    However, the directive suggests using convex optimization. Is there a convex relaxation or reformulation?
    Or perhaps the "convex optimization" hint implies that for the specific structure of "overlap" or "correlation" in this context, a different variable change makes it convex?
    
    Actually, let's look at the inspirations.
    Inspiration 1 mentions "cyc" (cyclic?).
    Inspiration 3 mentions "cosine basis".
    
    Let's re-read the directive: "Formulate the problem as a convex optimization by introducing an auxiliary variable t to bound the maximum correlation, then solve the resulting convex quadratic program using scipy.optimize.linprog or a specialized QP solver to find the exact global minimum of the epigraph."
    
    This implies the problem *can* be formulated as a convex QP.
    How?
    Maybe the variable is not h, but something else?
    Or maybe the "correlation" here allows a convex formulation?
    
    Let's look at the term: $c_k = \sum h_i (1 - h_{i+k})$.
    This is not convex.
    
    However, if we fix the multiset of values, the problem is discrete.
    But the directive says "convex optimization".
    
    Let's consider the possibility that the "convex" part refers to a relaxation that provides a strong lower bound, or perhaps the problem structure allows a specific convex representation.
    
    Alternative interpretation:
    Maybe the "correlation" is linear in some transformed variables?
    
    Let's try a different angle. The current program uses discrete search.
    A "materially different" strategy is continuous optimization.
    Even if it's non-convex, using a gradient-based continuous optimizer (like SLSQP or BFGS with a smooth approximation of the max) is a different family than discrete combinatorial search.
    
    The directive specifically asks for "convex optimization... solve the resulting convex quadratic program".
    If the problem isn't convex, maybe I should use a solver that handles non-convex quadratics? Or maybe the prompt implies that the *bound* on the max correlation can be optimized convexly?
    
    Actually, let's look at the math again.
    $h \in [0,1]^N$. $\sum h = N/2$.
    $c_k = \sum h_i - \sum h_i h_{i+k}$.
    We want to minimize $\max_k c_k$.
    
    If we relax the integrality/discrete constraint and allow continuous $h$, is the feasible set convex? Yes, box constraints and linear equality.
    Is the objective convex? No, it's the max of non-convex functions.
    
    However, `scipy.optimize.minimize` with `method='SLSQP'` can handle non-convex problems. It's not a "convex QP solver" in the strict sense, but it is a continuous optimization method.
    
    Perhaps the "convex optimization" hint is a red herring or refers to a specific relaxation (like SDP) which is too complex for 120s.
    
    Let's stick to the "Structural Change" directive: "a different design or modelling family, a different optimization or search method, or a different representation of the problem."
    
    The current program is **Discrete Combinatorial Search**.
    The new program will be **Continuous Gradient-Based Optimization**.
    
    I will use `scipy.optimize.minimize` with `SLSQP` to minimize the smooth approximation of the max correlation:
    $L_\mu(h) = \frac{1}{\mu} \log(\sum_k \exp(\mu c_k(h)))$
    This is the log-sum-exp trick, which is a convex function of $c_k$, but since $c_k$ is non-convex, the whole thing is non-convex. But it is smooth and differentiable, allowing gradient descent.
    
    To ensure the sum constraint $\sum h = N/2$ and bounds $0 \le h \le 1$, I will use SLSQP with these constraints.
    
    Initial guess: uniform 0.5.
    
    This is a fundamentally different approach from the discrete swaps.
    
    Implementation details:
    1. Define $c(h)$ vector.
    2. Define objective $J(h) = \text{logsumexp}(\mu c(h)) / \mu$.
    3. Define gradient of $J(h)$.
       $c_k(h) = \sum_i h_i (1 - h_{i+k})$.
       $\frac{\partial c_k}{\partial h_j} = (1 - h_{j+k}) - h_j \delta_{j+k, i} \dots$ wait.
       $c_k = \sum_i h_i - \sum_i h_i h_{i+k}$.
       $\frac{\partial c_k}{\partial h_j} = 1 - \sum_i \frac{\partial (h_i h_{i+k})}{\partial h_j}$.
       Term $h_i h_{i+k}$ depends on $h_j$ if $j=i$ or $j=i+k$.
       So $\frac{\partial c_k}{\partial h_j} = 1 - (h_{j+k} + h_{j-k})$ (with appropriate boundary handling for the correlation definition).
       
       Actually, `np.correlate(h, 1-h, 'full')` computes:
       $c_k = \sum_{i} h[i] (1-h)[i+k]$ where the index shifts.
       Let's just use numerical gradient if analytical is tricky, or derive it carefully.
       For N=100, numerical gradient is slow but SLSQP might be okay? No, SLSQP needs many evaluations.
       
       Let's use `jac` provided by SLSQP? No, SLSQP uses finite differences if not provided.
       Providing the gradient is crucial for speed.
       
       Let's derive the gradient of $c_k$ w.r.t $h$.
       $c_k = \sum_{i} h_i (1 - h_{i+k})$.
       Let $v = 1-h$.
       $c_k = \sum h_i v_{i+k}$.
       $\frac{\partial c_k}{\partial h_j} = v_{j+k} + h_j \frac{\partial v_{j+k}}{\partial h_j} + \dots$
       Actually, it's simpler:
       $c_k = \sum_i h_i - \sum_i h_i h_{i+k}$.
       $\frac{\partial c_k}{\partial h_j} = 1 - \frac{\partial}{\partial h_j} (\sum_i h_i h_{i+k})$.
       The term $h_i h_{i+k}$ involves $h_j$ if $j=i$ (then term is $h_j h_{j+k}$) or $j=i+k$ (then term is $h_{j-k} h_j$).
       So $\frac{\partial c_k}{\partial h_j} = 1 - (h_{j+k} + h_{j-k})$ where out-of-bounds indices are treated as 0 contribution (since the sum is over valid $i$).
       Note: `np.correlate` 'full' mode length is $2N-1$.
       Index $k$ in `correlate` output corresponds to lag $k - (N-1)$.
       Let's just compute the gradient numerically for the log-sum-exp objective?
       For N=100, a vectorized gradient calculation is possible.
       
       Let's try a simpler continuous approach:
       Minimize $\max_k c_k$ using SLSQP with the objective defined as the max (non-smooth) or log-sum-exp (smooth).
       SLSQP can handle non-smooth max, but log-sum-exp is better.
       
       I will implement the gradient of the log-sum-exp objective.
       
       $J(h) = \frac{1}{\mu} \ln \sum_k e^{\mu c_k}$.
       $\nabla J = \frac{1}{\mu} \frac{\sum_k e^{\mu c_k} \nabla c_k}{\sum_k e^{\mu c_k}} = \sum_k p_k \nabla c_k$, where $p_k = \frac{e^{\mu c_k}}{\sum e^{\mu c_k}}$.
       
       $\nabla c_k$ is a vector of length N.
       $\frac{\partial c_k}{\partial h_j} = 1 - (h_{j+k'} + h_{j-k'})$ where $k'$ is the lag corresponding to index $k$ in the full correlation.
       Lag $l = k - (N-1)$.
       $c_l = \sum_i h_i (1-h_{i+l})$.
       $\frac{\partial c_l}{\partial h_j} = 1 - (h_{j+l} + h_{j-l})$ with $h_{idx}=0$ if $idx < 0$ or $idx \ge N$.
       
       This can be vectorized.
       
    Strategy:
    1. Start with $h = 0.5$.
    2. Use SLSQP to minimize log-sum-exp of correlations.
    3. Constraints: sum(h)=50, bounds [0,1].
    4. Return best h.
    
    This is a continuous, gradient-based method, structurally different from discrete combinatorial search.
    """
    
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)
    
    # Objective function using log-sum-exp for smoothness
    mu = 10.0 # Sharpness of max approximation
    
    def obj_and_grad(h):
        h = np.clip(h, 0.0, 1.0)
        c = np.correlate(h, 1.0 - h, mode="full")
        
        # Log-sum-exp for numerical stability
        max_c = np.max(c)
        if max_c > 100: # Avoid overflow
            max_c = 100
        exp_c = np.exp(mu * (c - max_c))
        sum_exp = np.sum(exp_c)
        val = (max_c + np.log(sum_exp) / mu)
        
        # Gradient
        # p_k = exp(mu * c_k) / sum(exp(mu * c_k))
        p = exp_c / sum_exp
        
        # grad_c_k is a matrix of shape (2N-1, N)
        # grad_c_k[j] = 1 - (h[j+l] + h[j-l])
        # l = k - (N-1)
        
        # We can compute the gradient of the objective as sum_k p_k * grad_c_k
        # grad_obj[j] = sum_k p_k * (1 - h[j+l_k] - h[j-l_k])
        
        # Note: sum_k p_k * 1 = 1.
        # So grad_obj[j] = 1 - sum_k p_k * (h[j+l_k] + h[j-l_k])
        
        # Let's compute the weighted sum of h shifts.
        # l ranges from -(N-1) to N-1.
        # For a fixed j, we sum over k (or l) such that the term is valid.
        # This is a convolution-like operation.
        
        # p is length 2N-1.
        # We need S[j] = sum_l p[l + N - 1] * (h[j+l] + h[j-l])
        # where h[idx] = 0 if idx out of bounds.
        
        # This is circular convolution if we pad, but it's linear convolution here.
        # Let's use np.convolve or similar?
        # Actually, for N=100, we can just loop or use vectorized shifts.
        
        # Vectorized approach:
        # For each lag l, weight is p[l+N-1].
        # Contribution to grad from lag l: -p[l+N-1] * (shift(h, l) + shift(h, -l))
        
        # shift(h, l) is h shifted by l.
        
        grad_obj = np.ones(N) # From the sum p_k * 1
        
        for l in range(-(N-1), N):
            w = p[l + N - 1]
            if w < 1e-10:
                continue
            
            # shift h by l
            # h[j+l]
            if l >= 0:
                term1 = np.zeros(N)
                term1[:N-l] = h[l:]
                # h[j-l]
                term2 = np.zeros(N)
                term2[l:] = h[:N-l]
            else:
                # l is negative, let m = -l
                m = -l
                term1 = np.zeros(N)
                term1[m:] = h[:N-m]
                term2 = np.zeros(N)
                term2[:N-m] = h[m:]
            
            grad_obj -= w * (term1 + term2)
        
        return val, grad_obj
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2},
    ]
    bounds = [(0, 1) for _ in range(N)]
    
    # Initial guess
    h0 = np.full(N, 0.5)
    
    # Run optimization
    try:
        res = minimize(
            obj_and_grad, 
            h0, 
            method='SLSQP', 
            jac=True, 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9}
        )
        
        if res.success and res.fun < best_val:
            h_opt = np.clip(res.x, 0.0, 1.0)
            # Project to satisfy sum constraint exactly if SLSQP drifted
            diff = N/2 - np.sum(h_opt)
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0.0, 1.0)
            
            val_opt = compute_value(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt
    except Exception as e:
        pass # Keep initial best
    
    return np.array(best_h, dtype=float)

# EVOLVE-BLOCK-START
# (construct is already defined above)
# EVOLVE-BLOCK-END
