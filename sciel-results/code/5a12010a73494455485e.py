"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# State tracking for dynamic adaptation
# We infer the number of evaluations by tracking how many times acquisition has been called.
# The loop structure: 5 initial points, then 2000 candidates -> 1 eval.
# Total evals = 30. So acquisition is called 25 times (steps 6-30).
# However, the GP is updated *before* the acquisition call in the standard loop?
# The prompt says: "draws 2000 fresh uniform candidates, computes the posterior there, calls acquisition and evaluates the candidate with the highest score"
# This implies the GP is built from *previous* evaluations.
# Initial state: 5 points. 
# Step 1 (eval 6): GP has 5 points.
# Step 2 (eval 7): GP has 6 points.
# ...
# Step 25 (eval 30): GP has 29 points.

_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    
    # Increment call count
    _call_count += 1
    
    # Estimate current number of evaluations
    # If this is the k-th acquisition call, the GP is built from (5 + (k-1)) points.
    # So n_points = 5 + _call_count - 1 = 4 + _call_count
    n_points = 4 + _call_count
    
    # Total budget is 30 evaluations.
    # Remaining evaluations = 30 - n_points
    # Progress = n_points / 30
    
    # Dynamic Beta for EI
    # Early on (low n_points), we want high exploration.
    # Later (high n_points), we want high exploitation.
    # Standard EI uses f_best. 
    # Improvement = f_best - mu
    # EI = E[max(0, Improvement)]
    # For minimization:
    # If mu < f_best, potential improvement is f_best - mu.
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # To make it adaptive, we can add a small exploration term or adjust how we handle sigma.
    # However, pure EI is often too greedy in early stages if f_best is not yet very good.
    # The inspiration 1 suggests "pure exploitation (maximizing probability of improvement) in the early phase"?? 
    # Wait, inspiration 1 says: "uses pure exploitation ... in the early phase to quickly f | Initial program..." 
    # Actually, usually you explore early. But maybe for Ackley, finding *any* low value quickly is key?
    # Let's stick to the proven winner's logic: "Dynamic adaptation ... aggressive exploitation early to establish a good baseline, followed by strong exploration later"
    # This is counter-intuitive. Usually you explore to find the basin, then exploit.
    # But if the initial 5 points are random, f_best might be high.
    # If we exploit early, we refine around the best of 5 random points.
    # If we explore early, we might jump to a different basin.
    # Ackley is multimodal.
    
    # Let's implement a hybrid: 
    # Score = EI + alpha * sigma
    # Where alpha decays or grows?
    # If inspiration 1 says "exploitation early", then alpha should be LOW early and HIGH later?
    # Or maybe it means the *acquisition function itself* is EI (which is exploitation-focused) early, and LCB (exploration-focused) later?
    
    # Let's try a dynamic EI where the "improvement" target is adjusted or we blend EI and LCB.
    
    # Strategy: Blend EI and LCB based on progress.
    # Progress p = n_points / 30
    # Weight for EI: w_ei
    # Weight for LCB: w_lcb
    
    # If "exploitation early", then w_ei should be high early.
    # If "exploration later", then w_lcb should be high later.
    
    p = n_points / 30.0
    
    # Early: p ~ 0.16 (5/30) to 0.23 (7/30)
    # Late: p ~ 0.93 (28/30)
    
    # Let's say w_ei = 1 - p, w_lcb = p?
    # Or maybe a sharper transition.
    
    # Calculate EI
    sigma_safe = np.maximum(sigma, 1e-9)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate LCB score (higher is better)
    # LCB minimization: min(mu - beta_lcb * sigma)
    # Score for LCB: -(mu - beta_lcb * sigma) = -mu + beta_lcb * sigma
    # We need a beta for LCB. Let's use a fixed reasonable beta for the LCB component, e.g., 2.0
    beta_lcb = 2.0
    lcb_score = -mu + beta_lcb * sigma_safe
    
    # Normalize scores to comparable scales?
    # EI is in objective units. LCB is in objective units (roughly).
    # But their magnitudes might differ.
    
    # Alternative: Use a single dynamic beta in an "Optimistic Improvement" style?
    # Let's stick to the blend.
    
    # Weights:
    # Start with high EI weight, end with high LCB weight.
    # w_ei = 1.0 - p
    # w_lcb = p
    
    # But at p=0, w_ei=1. At p=1, w_lcb=1.
    # This matches "exploitation early, exploration later".
    
    w_ei = 1.0 - p
    w_lcb = p
    
    scores = w_ei * ei + w_lcb * lcb_score
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores

# Reset state if module is reloaded or for safety
# In a standard evaluator loop, the function is called sequentially.
# However, if the evaluator imports the module fresh for each seed, state is reset.
# If it runs all seeds in one process, state persists. 
# The prompt says "For each seed 0, 1 and 2 the evaluator starts from 5 uniform random points".
# It does not explicitly say if the Python process is restarted.
# "Keep the function names and signatures of the seed program."
# Usually, these benchmarks run in a single process or isolated. 
# If isolated, _call_count resets. If not, it might be off.
# To be safe, we can infer _call_count from the data if possible? 
# No, we only have mu, sigma, f_best. 
# We can't reliably infer n_points from mu/sigma without the X data.
# However, the length of mu is always 2000.
# Is there a way to detect reset? 
# If f_best drops significantly, maybe? No, f_best is non-increasing.
# Let's assume the evaluator resets the module state or runs seeds in separate processes. 
# If it doesn't, the _call_count will go 1..25 for seed 0, then 26..50 for seed 1.
# This would make seed 1 and 2 start in "late" exploration mode, which might be bad if they start fresh.
# 
# RISK: State persistence.
# Mitigation: Can we make it stateless?
# If we can't infer step, we can't do dynamic adaptation based on step.
# 
# Alternative Stateless Strategy:
# Use f_best magnitude to determine phase?
# Ackley min is 0. Max is ~10-20.
# If f_best is high (> 5), we are likely in early phase (bad baseline).
# If f_best is low (< 1), we are in late phase (good baseline).
# 
# This is a "value-based" adaptation, not "time-based".
# This is safer and stateless.
# 
# Let's switch to value-based adaptation.
# 
# Revised Strategy:
# Phase determined by f_best.
# High f_best -> Explore (High LCB weight)
# Low f_best -> Exploit (High EI weight)
# 
# Thresholds:
# f_best > 5.0: Early/Exploration
# f_best < 1.0: Late/Exploitation
# In between: Blend.

def acquisition(mu, sigma, f_best):
    # Stateless adaptation based on f_best
    # Normalize f_best to a 0-1 scale where 0 is best (0) and 1 is worst (~10)
    # Ackley 2D range is roughly [0, 10].
    f_scale = 10.0
    f_norm = min(f_best / f_scale, 1.0)
    
    # If f_best is high (f_norm close to 1), we want to explore.
    # If f_best is low (f_norm close to 0), we want to exploit.
    
    # Weight for Exploration (LCB)
    w_lcb = f_norm
    # Weight for Exploitation (EI)
    w_ei = 1.0 - f_norm
    
    # Calculate EI
    sigma_safe = np.maximum(sigma, 1e-9)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate LCB score
    beta_lcb = 2.5
    lcb_score = -mu + beta_lcb * sigma_safe
    
    scores = w_ei * ei + w_lcb * lcb_score
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
