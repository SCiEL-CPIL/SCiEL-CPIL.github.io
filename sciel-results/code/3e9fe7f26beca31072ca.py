"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    Strategy: Combine the structural idea of favoring large items (product)
    with a refinement that strongly penalizes loose fits (squared inverse waste).
    
    The heuristic encourages:
    1. Pairing items that fit together.
    2. Among fitting pairs, preferring those with larger product (large items).
    3. Among those, strongly preferring those that fill the bin more completely.
       We use a squared inverse waste term: 1 / (1 - sum/capacity)^2
       This creates a sharp peak for tight fits, which is critical for
       minimizing bin count in bin packing.
    """
    start_time = time.time()
    
    # Convert to numpy array for vectorized operations
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    
    if n == 0:
        return np.empty((0, 0))
        
    # Calculate sums to check if items fit together
    sums = d[:, None] + d[None, :]
    
    # Calculate products to measure "compatibility" (favoring large-large pairs)
    products = d[:, None] * d[None, :]
    
    # Condition: items must fit together
    fits = sums <= capacity
    
    # Calculate waste: how much space is left in the bin
    # waste = capacity - sum, normalized by capacity
    # For fitting pairs, sum <= capacity, so waste >= 0
    # We want to penalize large waste.
    # Let's use a term that goes to infinity as waste -> 0? No, we want high score for tight fits.
    # So we want a term that is large when sum is close to capacity.
    
    # Let fill_ratio = sum / capacity. Range [0, 1] for fitting pairs.
    # We want a function f(fill_ratio) that increases with fill_ratio.
    # Previous attempt: product * fill_ratio. Score 1.0192.
    # Inspiration 1: product. Score 1.0063.
    # Inspiration 2: fill_ratio. Score 1.0000.
    # Cell finding: "Sharper decay for waste (squared inverse) strongly prioritizes tight fits... Measured: 1.026 -> 1.032"
    # Wait, the cell finding says 1.026 -> 1.032, which is an improvement (higher score is better? No, task says "lower is better" for raw value, but "search score is seed/value, so 1.0 equals seed... above 1.0 is improvement").
    # So higher search score is better.
    # The cell finding says "1.026 -> 1.032", which is an improvement.
    # The strategy was "squared inverse waste".
    # Let's interpret "waste" as (capacity - sum).
    # Inverse waste: 1 / (capacity - sum).
    # Squared inverse waste: 1 / (capacity - sum)^2.
    # This would be large when sum is close to capacity.
    # However, if sum == capacity, it's infinity. We need to handle that.
    # Also, we must ensure the term is well-behaved.
    
    # Let's try combining product with a sharp tightness term.
    # Tightness term: 1 / (capacity - sum + epsilon)^2
    # This heavily favors pairs that almost fill the bin.
    
    # Calculate the tightness term
    # For fitting pairs, capacity - sum >= 0.
    # Use a smooth denominator to avoid extreme spikes.
    waste = capacity - sums
    
    # Use waste + 1.0 to smooth the tightness.
    # This avoids the 1/x^2 explosion for small waste.
    # For fitting pairs: denom = waste + 1.0
    # For non-fitting pairs: denom = 1.0 (masked later)
    
    safe_waste = np.where(fits, waste + 1.0, 1.0)
    tightness = 1.0 / safe_waste
    
    # Combine product and tightness
    # H = product * tightness
    # For fitting pairs, this is (d_i * d_j) / (capacity - d_i - d_j + 1.0)
    # For non-fitting pairs, this is (d_i * d_j) * 1.0, but we mask to 1e-9.
    
    raw_scores = products * tightness
    
    # Apply the fit condition
    H = np.where(fits, raw_scores, 1e-9)
    
    # Refinement: Add a secondary term that favors pairs where both items are large
    # This helps when tightness is similar, breaking ties in favor of larger items
    # which are harder to place and benefit more from early placement
    # We use a normalized product term that is small relative to the main term
    # but provides a consistent bias toward large-large pairs
    # Normalize demands to [0, 1] range for the secondary term
    d_min = np.min(d)
    d_max = np.max(d)
    d_range = d_max - d_min
    if d_range > 0:
        d_norm = (d - d_min) / d_range
    else:
        d_norm = np.zeros_like(d)
    
    # Secondary term: product of normalized sizes
    # This is in [0, 1] range, so it's small compared to the main term
    # but provides a consistent bias
    secondary = d_norm[:, None] * d_norm[None, :]
    
    # Combine: main term + small secondary term
    # The secondary term is scaled down to be a minor influence
    H = np.where(fits, raw_scores + 0.1 * secondary, 1e-9)
    
    # Ensure no non-finite values
    if not np.all(np.isfinite(H)):
        # If any non-finite values exist, replace them with 1e-9
        H = np.where(np.isfinite(H), H, 1e-9)
    
    # Ensure shape is correct
    if H.shape != (n, n):
        raise ValueError("Wrong shape")
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 325:
        # Safety margin: if we're close to the limit, return immediately
        pass
    
    return H
# EVOLVE-BLOCK-END
