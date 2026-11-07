import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic matrix for ACO-based CVRP.
    Uses an inverse-square distance base modified by a Clarke-Wright savings bonus
    for edges between customers to encourage route merging.
    """
    n = distance_matrix.shape[0]
    # Initialize heuristic matrix
    eta = np.zeros((n, n), dtype=np.float64)
    
    # Extract depot distances for savings calculation
    # d0[i] is distance from depot (0) to node i
    d0 = distance_matrix[0, :]
    
    # Compute base heuristic: 1 / d^2
    # Handle division by zero (diagonal)
    # We'll compute for all i != j
    # Using broadcasting for efficiency
    # d_sq = distance_matrix ** 2
    # base = 1.0 / d_sq
    
    # To avoid divide by zero warnings and inf, we can use np.where or clip
    # But we need to apply the savings bonus only for i>0, j>0
    
    # Let's iterate or use vectorized ops
    # Vectorized approach:
    # 1. Compute base = 1 / d^2
    # 2. Compute savings_ratio for all pairs
    # 3. Apply bonus where i>0 and j>0
    
    # Safe division
    # Replace 0 with a small number for calculation, then fix diagonal
    d_safe = np.where(distance_matrix > 0, distance_matrix, 1e-9)
    base = 1.0 / (d_safe ** 2)
    
    # Calculate savings ratio: (d0i + d0j - dij) / (d0i + d0j)
    # d0i is d0[i], d0j is d0[j]
    # We need d0[i] for rows and d0[j] for cols
    d0_rows = d0[:, np.newaxis]  # shape (n, 1)
    d0_cols = d0[np.newaxis, :]  # shape (1, n)
    
    # Numerator: d0i + d0j - dij
    num = d0_rows + d0_cols - distance_matrix
    # Denominator: d0i + d0j
    den = d0_rows + d0_cols
    
    # Avoid division by zero in den (only if d0i=0 and d0j=0, i.e., i=0, j=0)
    den_safe = np.where(den > 0, den, 1e-9)
    
    savings_ratio = num / den_safe
    
    # Clip savings ratio to [0, 1] to handle any numerical weirdness or triangle inequality violations (shouldn't happen with Euclidean)
    savings_ratio = np.clip(savings_ratio, 0.0, 1.0)
    
    # Create a mask for customer-customer edges (i > 0 and j > 0)
    # Note: Node 0 is the depot.
    mask_customer = np.ones((n, n), dtype=bool)
    mask_customer[0, :] = False
    mask_customer[:, 0] = False
    
    # Apply bonus: eta = base * (1 + savings_ratio) for customers
    # For depot edges, eta = base
    bonus_factor = 1.0 + savings_ratio
    
    # Initialize eta with base values
    eta = base.copy()
    
    # Update customer-customer entries
    eta[mask_customer] = base[mask_customer] * bonus_factor[mask_customer]
    
    # Ensure diagonal is 0 or very small, though ants don't pick self
    np.fill_diagonal(eta, 1e-9)
    
    # Enforce positivity: non-positive becomes 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
