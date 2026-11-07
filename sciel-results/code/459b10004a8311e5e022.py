"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Objective: minimize max_k sum_{i=0}^{N-1-k} h[i]*(1-h[i+k])
    # Using epigraph formulation: min t s.t. t >= each correlation value
    # For smoothness, we can use a soft-max or just optimize the max directly with penalty
    
    # Let's use a different approach: parameterize h[i] = 0.5 + 0.5 * sin(theta_i)
    # But sum constraint: sum(h) = 50 => sum(sin(theta_i)) = 0
    # This is complex. Let's stick with direct optimization with constraints.
    
    # Variables: h (length 100) and t (scalar)
    # Minimize t
    # Subject to:
    #   sum(h) = 50
    #   0 <= h[i] <= 1
    #   For all k in 0..99: sum_{i=0}^{N-1-k} h[i]*(1-h[i+k]) <= t * N / 2
    
    # Note: The task value is max_k correlate_k / N * 2
    # So we want to minimize max_k correlate_k.
    # Let C_k = sum_{i=0}^{N-1-k} h[i]*(1-h[i+k])
    # We want min max_k C_k
    # Equivalently, min t s.t. C_k <= t for all k
    
    # Initial guess: alternating 0 and 1?
    # h = [1, 0, 1, 0, ...]
    # Sum = 50. Good.
    # Correlations:
    # k=0: sum h[i](1-h[i]) = 0
    # k=1: sum h[i](1-h[i+1]) = sum 1*1 + 0*0 ... wait.
    # h[i]=1, h[i+1]=0 => 1*(1-0)=1. There are 50 such pairs.
    # h[i]=0, h[i+1]=1 => 0*(1-1)=0.
    # So C_1 = 50.
    # k=2: h[i]=1, h[i+2]=1 => 1*(1-1)=0.
    # So max C_k = 50.
    # Value = 50 / 100 * 2 = 1.0.
    # Seed is 0.5 everywhere.
    # C_0 = sum 0.5*0.5 = 25.
    # C_1 = sum 0.5*0.5 = 25.
    # ...
    # C_k = (100-k)*0.25.
    # Max is at k=0: 25.
    # Value = 25/100*2 = 0.5.
    # Wait, the task says "lower is better".
    # Seed value is 0.5.
    # My alternating example gives 1.0. That's worse.
    # I want to MINIMIZE the max correlation.
    
    # Let's try to make h smooth?
    # If h is constant 0.5, C_k = (100-k)*0.25. Max is 25.
    # If we can make C_k smaller for all k, that's better.
    # Note C_k = sum h[i] - sum h[i]h[i+k].
    # Sum h[i] is fixed at 50 for k=0? No, sum h[i] is 50.
    # C_0 = sum h[i](1-h[i]) = 50 - sum h[i]^2.
    # To minimize C_0, we maximize sum h[i]^2.
    # Max sum of squares with sum=50, 0<=h<=1 is when as many 1s and 0s as possible.
    # 50 ones, 50 zeros. Sum squares = 50.
    # C_0 = 50 - 50 = 0.
    # But then C_1 depends on arrangement.
    # If we group 1s and 0s: 111...1000...0
    # C_1: boundary is 1*0 = 0. Internal 1*1=0, 0*0=0.
    # So C_1 = 0?
    # Let's check.
    # h = [1,1,...,1, 0,0,...,0] (50 ones, 50 zeros)
    # C_0 = 0.
    # C_1: pairs (1,1)->0, (1,0)->0, (0,0)->0. All 0.
    # C_k for k>=1: All pairs are either (1,1), (0,0) or boundary (1,0).
    # (1,1): 1*(1-1)=0.
    # (0,0): 0*(1-0)=0.
    # (1,0): 1*(1-0)=1.
    # There is only ONE boundary (1,0) if we have one block of 1s and one block of 0s.
    # So C_k = 1 for k in [1, 49]?
    # Let's verify.
    # Indices 0..49 are 1. Indices 50..99 are 0.
    # k=1:
    # i=0..48: h[i]=1, h[i+1]=1 -> 0.
    # i=49: h[49]=1, h[50]=0 -> 1.
    # i=50..98: h[i]=0, h[i+1]=0 -> 0.
    # Sum = 1.
    # k=2:
    # i=0..47: (1,1)->0.
    # i=48: (1,1)->0 (h[48]=1, h[50]=0? No, h[50]=0. h[48]=1, h[50]=0. Wait.
    # h[48]=1, h[50]=0. Pair is (1,0). Value 1.
    # i=49: h[49]=1, h[51]=0. Pair (1,0). Value 1.
    # i=50..97: (0,0)->0.
    # Sum = 2.
    # Generally, for k, the number of (1,0) pairs is min(k, 50, 50)?
    # The transition zone has width k.
    # For k <= 50, there are k pairs (1,0).
    # So C_k = k for k=1..50.
    # Max C_k = 50 (at k=50).
    # Value = 50/100*2 = 1.0.
    
    # So blocky is bad.
    # Alternating was bad (1.0).
    # Constant was 0.5.
    
    # We want to minimize max_k C_k.
    # C_k = sum_{i=0}^{N-1-k} h[i](1-h[i+k]).
    # This is like the "overlap" of the 1s in h and the 0s in h shifted.
    
    # Let's try a sinusoidal h[i] = 0.5 + 0.5 * sin(2*pi*i/M + phase).
    # Sum is 0 if M divides N?
    # N=100.
    # If M=4, sin(2*pi*i/4) = [0, 1, 0, -1].
    # h = [0.5, 1.0, 0.5, 0.0, ...]
    # Sum = 100 * 0.5 = 50.
    # Let's calculate C_k for this.
    # This is a specific instance.
    
    # Let's use optimization to find the best h.
    # We will use SLSQP or trust-constr.
    # Variables: h (100).
    # Objective: max_k C_k.
    # This is non-smooth.
    # We can use a smooth approximation: softmax of C_k / tau.
    # Or just use the fact that we can add a penalty for exceeding a target t.
    
    # Let's try to minimize the sum of squares of (C_k - t) + penalty for C_k > t.
    # Or simply minimize max(C_k) using a penalty method.
    
    # Let's try a simple gradient descent on the smooth surrogate:
    # f(h) = log(sum(exp(C_k / tau))) * tau
    # This approximates max(C_k).
    
    # Constraints:
    # sum(h) = 50
    # 0 <= h <= 1
    
    # We will use scipy.optimize.minimize with 'L-BFGS-B' or 'SLSQP'.
    # L-BFGS-B handles bounds well but equality constraints are tricky (needs penalty).
    # SLSQP handles both.
    
    # Let's define the objective function.
    # For speed, we can vectorize the correlation calculation.
    # C_k can be computed via FFT or just a loop since N=100 is small.
    # N=100, so 100 correlations, each sum of up to 100 terms. 10k ops. Very fast.
    
    def objective(h):
        # Compute C_k for all k
        # C_k = sum_{i=0}^{N-1-k} h[i] * (1 - h[i+k])
        Cs = np.zeros(N)
        for k in range(N):
            # h[0:N-k] and h[k:N]
            prod = h[:N-k] * (1 - h[k:N])
            Cs[k] = np.sum(prod)
        
        # Smooth max
        # Use log-sum-exp
        # max(Cs) approx log(sum(exp(Cs/tau))) * tau
        # To avoid overflow, subtract max
        max_c = np.max(Cs)
        # If tau is small, this is sharp.
        tau = 1.0
        # exp((Cs - max_c)/tau)
        exp_cs = np.exp((Cs - max_c) / tau)
        log_sum = np.log(np.sum(exp_cs))
        return max_c + tau * log_sum
    
    def constraint_sum(h):
        return np.sum(h) - 50.0
    
    # Initial guess: 0.5
    h0 = np.full(N, 0.5)
    
    # Try a few different initial guesses
    guesses = [
        h0,
        np.tile([1.0, 0.0], N//2),
        np.tile([0.0, 1.0], N//2),
        0.5 + 0.5 * np.sin(2 * np.pi * np.arange(N) / 10),
        0.5 + 0.5 * np.sin(2 * np.pi * np.arange(N) / 20),
        np.random.rand(N) * 0.0 + 0.5 # just in case
    ]
    
    # Normalize guesses to sum 50?
    # The optimizer will handle it, but starting close is better.
    for i in range(len(guesses)):
        guesses[i] = guesses[i].copy()
        # Project onto sum=50?
        # Just let SLSQP do it.
        
    best_h = h0
    best_val = np.inf
    
    for h_init in guesses:
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(
                objective,
                h_init,
                method='SLSQP',
                bounds=[(0, 1)] * N,
                constraints=[{'type': 'eq', 'fun': constraint_sum}],
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            if time.time() > deadline - 5:
                break
                
            h_opt = res.x
            # Clip to [0,1]
            h_opt = np.clip(h_opt, 0, 1)
            
            # Check sum constraint
            s = np.sum(h_opt)
            if abs(s - 50) > 1e-6:
                # Scale? No, scaling breaks bounds.
                # Adjust?
                # If it's slightly off, we might fail.
                # Let's just evaluate the value.
                pass
            
            # Calculate actual max C_k
            Cs = np.zeros(N)
            for k in range(N):
                prod = h_opt[:N-k] * (1 - h_opt[k:N])
                Cs[k] = np.sum(prod)
            max_c = np.max(Cs)
            
            if max_c < best_val:
                best_val = max_c
                best_h = h_opt.copy()
                
        except Exception as e:
            pass
    
    # Final polish: ensure sum is exactly 50 and bounds are respected.
    # If sum is not 50, we need to fix it.
    # Simple fix: distribute the difference evenly?
    # Or just return the best found if it's close enough.
    # The task requires sum within 1e-9 * 100 = 1e-7.
    
    s = np.sum(best_h)
    diff = 50.0 - s
    if abs(diff) > 1e-7:
        # Adjust
        # Add diff/N to each element?
        best_h += diff / N
        # Clip again
        best_h = np.clip(best_h, 0, 1)
        # Re-check sum
        s2 = np.sum(best_h)
        # If still bad, try a different fix
        # This is a heuristic.
        
    return best_h

# EVOLVE-BLOCK-END
