<<<<<<< SEARCH
def _evaluate_gap(cell, n_pw=13, n_bands=6, n_kpts=8):
    """Evaluate the gap-midgap ratio between bands 4 and 5."""
    # Irreducible BZ for square lattice: kx in [0, 0.5], ky in [0, kx] (or similar)
    # Key points: Gamma(0,0), M(0.5,0), X(0.5,0.5), and edges
    kpts = [
        (0.0, 0.0),      # Gamma
        (0.25, 0.0),
        (0.5, 0.0),      # M
        (0.25, 0.25),
        (0.5, 0.25),
        (0.5, 0.5),      # X
        (0.125, 0.0),
        (0.375, 0.0),
    ]
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in kpts:
        bands = _compute_bands(cell, kx, ky, n_pw=n_pw, n_bands=n_bands)
        if bands is None or len(bands) < n_bands:
            continue
        # bands[0] is lowest, bands[3] is 4th band, bands[4] is 5th band
        if len(bands) >= 5:
            max_omega4 = max(max_omega4, bands[3])
            min_omega5 = min(min_omega5, bands[4])
    
    if min_omega5 == float('inf') or max_omega4 == 0:
        return 0.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    if midgap <= 0:
        return 0.0
    
    return gap / midgap
=======
def _evaluate_gap(cell, n_pw=13, n_bands=6, n_kpts=8):
    """Evaluate the gap-midgap ratio between bands 4 and 5."""
    # Irreducible BZ for square lattice: kx in [0, 0.5], ky in [0, kx]
    # We sample points on the boundary and interior.
    # The evaluator uses 30 wave vectors. We should sample similarly.
    # Gamma (0,0), M (0.5, 0), X (0.5, 0.5)
    # Edges: Gamma-M, M-X, X-Gamma (by symmetry, Gamma-M and M-X cover the boundary if we consider the triangle)
    # Actually, the irreducible wedge is 0 <= ky <= kx <= 0.5.
    
    kpts = [
        (0.0, 0.0),      # Gamma
        (0.5, 0.0),      # M
        (0.5, 0.5),      # X
    ]
    
    # Add points along Gamma-M (ky=0, kx in [0, 0.5])
    for t in np.linspace(0.0, 0.5, 7)[1:-1]:
        kpts.append((t, 0.0))
        
    # Add points along M-X (kx=0.5, ky in [0, 0.5])
    for t in np.linspace(0.0, 0.5, 7)[1:-1]:
        kpts.append((0.5, t))
        
    # Add points along X-Gamma (kx=ky, kx in [0, 0.5])
    for t in np.linspace(0.0, 0.5, 7)[1:-1]:
        kpts.append((t, t))
        
    # Add some interior points
    kpts.append((0.25, 0.125))
    kpts.append((0.375, 0.25))
    kpts.append((0.25, 0.25))
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in kpts:
        bands = _compute_bands(cell, kx, ky, n_pw=n_pw, n_bands=n_bands)
        if bands is None or len(bands) < 5:
            continue
        # bands[3] is 4th band, bands[4] is 5th band
        max_omega4 = max(max_omega4, bands[3])
        min_omega5 = min(min_omega5, bands[4])
    
    if min_omega5 == float('inf') or max_omega4 == 0:
        return 0.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    if midgap <= 0:
        return 0.0
    
    return gap / midgap
>>>>>>> REPLACE

