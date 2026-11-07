# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Generates an edge-attractiveness matrix using a Soft Density-Aware Potential strategy.
    
    Unlike the Delaunay approach which creates a hard sparse graph, this approach
    uses a Gaussian kernel to create a smooth, fully-connected heuristic.
    Edges within a 'local density' radius are boosted, while very long edges are
    softly penalized. This avoids the risk of the ACO getting stuck in disconnected
    components or forced long detours caused by hard planar constraints, while
    still encouraging local clustering which is key to efficient CVRP routes.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin for 300s limit
    
    # Convert inputs to numpy arrays for safety
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    # Check time budget immediately
    if time.time() > deadline:
        # Fallback to simple inverse distance
        dist_safe = distance_matrix.copy()
        np.fill_diagonal(dist_safe, 1e6)
        fallback = 1.0 / dist_safe
        fallback = np.where(fallback > 0, fallback, 1e-9)
        return fallback

    # 1. Base Heuristic: Inverse Distance
    # This provides the fundamental "closer is better" signal.
    dist_safe = distance_matrix.copy()
    np.fill_diagonal(dist_safe, 1e6) # Avoid division by zero on diagonal
    base_heuristic = 1.0 / dist_safe
    
    # 2. Density-Aware Repulsive Potential
    # We want to boost edges that are "locally relevant" without hard-cutting distant ones.
    # The idea: If two points are very close, they are likely in the same route cluster.
    # If they are far, they are likely in different clusters or the depot.
    
    # Determine a characteristic length scale.
    # We can use the median of the non-zero, non-diagonal distances.
    # This makes the heuristic adaptive to the specific instance's spread.
    valid_dists = distance_matrix[~np.eye(n, dtype=bool)]
    valid_dists = valid_dists[valid_dists > 0]
    
    if len(valid_dists) == 0:
        # Degenerate case, shouldn't happen with 50 customers
        return base_heuristic
    
    # Use a fraction of the median distance as the scale for the Gaussian kernel.
    # A smaller sigma means a sharper peak around close neighbors.
    # A larger sigma means a broader influence.
    # Let's try sigma = 0.25 * median_dist.
    median_dist = np.median(valid_dists)
    sigma = 0.25 * median_dist
    
    # Calculate the Gaussian potential: exp(-d^2 / (2 * sigma^2))
    # This value is 1.0 at d=0 and decays as distance increases.
    # We use this to *boost* the base heuristic for close neighbors.
    
    # To avoid numerical underflow/overflow issues, we compute in a stable way.
    # d^2 / (2 * sigma^2)
    exponent = - (distance_matrix ** 2) / (2.0 * (sigma ** 2))
    
    # Clip exponent to prevent underflow to 0 for very large distances, 
    # though for n=50 in unit square, max dist is ~1.4, sigma ~0.1, 
    # so d=1.4 -> exp(-1.96/0.02) = exp(-98) ~ 1e-43, which is fine (effectively 0).
    # However, we want to ensure the diagonal (self-loops) doesn't blow up or mess things up.
    # The base_heuristic already handles the diagonal with 1e-6 dist -> 1e6 value.
    # The Gaussian for diagonal is exp(0) = 1.0.
    # So diagonal gets base_heuristic * 1.0.
    
    # We construct the final heuristic as:
    # H[i,j] = base_heuristic[i,j] * (1 + alpha * gaussian[i,j])
    # Where alpha controls the strength of the density boost.
    # If alpha is large, close neighbors are much more attractive.
    
    alpha = 10.0
    
    # Apply the boost
    # We must be careful with the diagonal. 
    # For i==j, base_heuristic is 1e6 (approx). Gaussian is 1.
    # H[i,i] = 1e6 * (1 + 10) = 11e6.
    # This is fine as long as the ACO ignores self-loops or we mask them.
    # The task says "picks an unvisited customer", so self-loops are naturally ignored 
    # if the ACO implementation checks visited status. 
    # However, the heuristic matrix is returned and used in probability calc.
    # Usually, ACO implementations set visited nodes to 0 or -inf in the *effective* heuristic/pheromone 
    # during construction. The static heuristic matrix itself can have any values on the diagonal 
    # as long as they are finite.
    
    # Let's refine:
    # We want to strongly encourage moving to nearby unvisited nodes.
    # The term (1 + alpha * exp(...)) ensures that even if base_heuristic is small (long dist),
    # if the Gaussian is high (short dist), the product is boosted.
    # Wait, base_heuristic IS 1/d. So for short d, base_heuristic is already high.
    # Multiplying by (1 + alpha * exp) makes it even higher.
    # For long d, base_heuristic is low, and exp is ~0, so H ~ base_heuristic.
    # This effectively preserves the 1/d behavior for long distances while amplifying 
    # the preference for short distances beyond simple inverse proportionality.
    
    heuristic_matrix = base_heuristic * (1.0 + alpha * np.exp(exponent))
    
    # Ensure the diagonal is handled safely for the return value.
    # Although ACO should ignore it, let's keep it finite.
    # The current logic yields ~11e6 for diagonal. This is finite.
    
    # Validation:
    # 1. Shape check
    if heuristic_matrix.shape != distance_matrix.shape:
        dist_safe = distance_matrix.copy()
        np.fill_diagonal(dist_safe, 1e6)
        heuristic_matrix = 1.0 / dist_safe
        heuristic_matrix = np.where(heuristic_matrix > 0, heuristic_matrix, 1e-9)
        return heuristic_matrix

    # 2. Finiteness check
    if not np.all(np.isfinite(heuristic_matrix)):
        # Replace non-finite values with a large number or fallback
        # If exp overflowed (unlikely with negative exponent), or if distance was 0 and base was inf?
        # base_heuristic used 1e6 for diagonal, so 1/1e6 = 1e-6. 
        # Wait, dist_safe diagonal was set to 1e6. So base_heuristic diagonal is 1e-6.
        # My previous mental math "1e6" was wrong. 1/1e6 is 1e-6.
        # So diagonal H is 1e-6 * (1+10) = 1.1e-5.
        # This is small. This is good. Self-loops are unattractive.
        
        # If any other non-finite values exist:
        heuristic_matrix = np.where(np.isfinite(heuristic_matrix), heuristic_matrix, 1e-9)
        
    # 3. Positivity check
    # "Heuristic entries that are not positive become 1e-9"
    heuristic_matrix = np.where(heuristic_matrix > 0, heuristic_matrix, 1e-9)
    
    return heuristic_matrix
# EVOLVE-BLOCK-END
