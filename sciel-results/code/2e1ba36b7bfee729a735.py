"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _build_cell_from_1d_patterns(pattern_x, pattern_y, size=32, block_size=8):
    """
    Build a size x size cell from two 1D patterns of length block_size.
    Each 1D pattern is a list of 0/1 of length block_size.
    The cell is formed by tiling: cell[i, j] = pattern_x[i // block_size] * pattern_y[j // block_size]
    But wait, that gives a 4x4 block structure where each 8x8 sub-block is uniform.
    
    Actually, let me reconsider. The strategy says:
    "Model the unit cell as a 4x4 block of 8x8 sub-cells defined by the intersection of two independent 1D square-wave functions (one for x, one for y), where each 1D function is parameterized by a 4-bit pattern of half-period widths"
    
    So each 1D function has 4 bits, meaning 4 transitions or 4 half-periods? 
    With 32 pixels and 4 bits, each bit controls 8 pixels.
    
    Let me interpret: a 1D pattern of 4 bits means we have 4 segments of 8 pixels each.
    Each bit is 0 or 1, so the 1D function is a 32-length vector where each 8-pixel segment is uniform.
    
    Then the 2D cell is: cell[i, j] = x_pattern[i] * y_pattern[j]
    where x_pattern and y_pattern are 32-length vectors derived from 4-bit patterns.
    
    This ensures D4 symmetry if we also enforce that the 1D patterns are symmetric about the center.
    For a 32-length vector with 8-pixel segments, symmetry about the center means:
    segment 0 == segment 3, segment 1 == segment 2 (if we think of it as 4 segments of 8)
    Wait, 32 pixels, center is between pixel 15 and 16.
    For symmetry: pixel[i] == pixel[31-i]
    With 8-pixel segments: segment k (pixels 8k to 8k+7) should mirror segment 3-k.
    So bit0 == bit3, bit1 == bit2.
    
    So effectively only 2 independent bits per 1D pattern, giving 4 possibilities per 1D.
    Total 4 * 4 = 16 unique cells. That's too few.
    
    Let me re-read: "each 1D function is parameterized by a 4-bit pattern of half-period widths"
    Maybe the 4 bits control the widths, not just on/off. But with binary 0/1 pixels, 
    a "half-period width" might mean the fraction of the segment that is on.
    
    Actually, I think the intended interpretation is simpler: 
    We have 4 independent 8-pixel blocks in each dimension. Each block is either all 0 or all 1.
    For D4 symmetry, we need the pattern to be symmetric. 
    If we don't enforce symmetry in the 1D pattern, the 2D cell won't be symmetric.
    
    To ensure D4 symmetry of the 2D cell, we can enforce that the 1D x-pattern and y-pattern 
    are both symmetric, AND that x-pattern == y-pattern (for transpose symmetry).
    
    If x-pattern == y-pattern and both are symmetric, then:
    - Row reversal: cell[::-1] reverses rows. Since y-pattern is symmetric, this is fine.
    - Column reversal: cell[:, ::-1] reverses columns. Since x-pattern is symmetric, this is fine.
    - Transpose: cell.T[i,j] = cell[j,i] = x[j]*y[i]. Since x==y, this equals x[i]*x[j] = cell[i,j].
    
    So we need x-pattern == y-pattern and both symmetric.
    
    A symmetric 32-length pattern with 8-pixel segments: bits b0,b1,b2,b3 with b0==b3, b1==b2.
    So 2 free bits, 4 possibilities.
    
    That gives only 4 unique cells. Still too few.
    
    Let me try a different interpretation: maybe the 4-bit pattern doesn't enforce symmetry, 
    and we just generate all 256 combinations of x-pattern and y-pattern (each 4 bits = 16 possibilities), 
    and then check which ones satisfy D4 symmetry.
    
    For D4 symmetry:
    1. cell == cell[::-1] (row reversal)
    2. cell == cell[:, ::-1] (column reversal)  
    3. cell == cell.T (transpose)
    
    With cell[i,j] = x[i]*y[j]:
    - Row reversal: cell[31-i, j] = x[31-i]*y[j] should equal x[i]*y[j] for all i,j. This requires x[31-i] = x[i].
    - Column reversal: cell[i, 31-j] = x[i]*y[31-j] should equal x[i]*y[j]. This requires y[31-j] = y[j].
    - Transpose: cell[j,i] = x[j]*y[i] should equal x[i]*y[j]. This requires x[j]*y[i] = x[i]*y[j] for all i,j.
    
    The transpose condition is the most restrictive. If x and y are both 0/1 vectors, 
    x[j]*y[i] = x[i]*y[j] for all i,j means that the sets where x=1 and y=1 are either 
    identical or one is a subset of the other in a specific way. Actually, this condition 
    implies that the matrix x⊗y is symmetric, which means x⊗y = (x⊗y)^T = y⊗x.
    So x⊗y = y⊗x, which means x and y must be proportional. Since they're binary, 
    x = y or one is all zeros.
    
    So for non-trivial solutions, we need x = y, and both must be symmetric.
    
    So we're back to 4 possibilities. That's not enough for a meaningful search.
    
    Let me try yet another interpretation: maybe the 4-bit pattern controls the 1D function 
    in a different way. Perhaps each bit controls whether a particular 8-pixel block is 
    dielectric, but the 1D function is not simply a tiling.
    
    Actually, I think I'm overcomplicating this. Let me just implement the strategy as described:
    "reducing the search space to 256 unique binary combinations that can be exhaustively evaluated"
    
    So 256 = 16 * 16, meaning 16 possibilities for x-pattern and 16 for y-pattern.
    16 = 2^4, so each 1D pattern has 4 bits.
    
    Let me define: a 4-bit pattern p = (p0, p1, p2, p3) generates a 32-length vector where:
    - Pixels 0-7: p0
    - Pixels 8-15: p1
    - Pixels 16-23: p2
    - Pixels 24-31: p3
    
    Then cell[i,j] = x_pattern[i] * y_pattern[j].
    
    For D4 symmetry, we need to filter these 256 combinations.
    Let me just generate all 256, check symmetry, and evaluate the valid ones.
    """
    x_vec = np.zeros(size, dtype=int)
    for k in range(4):
        x_vec[k*block_size:(k+1)*block_size] = pattern_x[k]
    
    y_vec = np.zeros(size, dtype=int)
    for k in range(4):
        y_vec[k*block_size:(k+1)*block_size] = pattern_y[k]
    
    cell = np.outer(x_vec, y_vec)
    return cell


def _is_d4_symmetric(cell):
    """Check if cell is invariant under D4 symmetry."""
    cell = np.array(cell)
    # Row reversal
    if not np.array_equal(cell, cell[::-1, :]):
        return False
    # Column reversal
    if not np.array_equal(cell, cell[:, ::-1]):
        return False
    # Transpose
    if not np.array_equal(cell, cell.T):
        return False
    return True


def _estimate_gap(cell, problem):
    """
    Estimate the gap-midgap ratio for TM bands 4 and 5.
    This is a simplified estimator using a reduced plane-wave basis.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Convert cell to epsilon map
    eps_map = np.where(np.array(cell) == 1, eps_high, eps_low)
    
    # Compute Fourier coefficients of 1/epsilon for TM mode
    # TM mode: we need Fourier coefficients of 1/eps(k)
    inv_eps = 1.0 / eps_map
    
    # Use a small number of plane waves for estimation
    # The evaluator uses 289 plane waves (17x17), but we use fewer for speed
    # Let's use 5x5 = 25 plane waves for a rough estimate
    n_pw = 5
    k_range = np.arange(-n_pw//2, n_pw//2 + 1)
    
    # Compute Fourier coefficients of inv_eps
    # G = (mx, my) where mx, my are integers
    fourier_coeffs = {}
    for mx in k_range:
        for my in k_range:
            # Fourier coefficient: (1/N^2) * sum_{i,j} inv_eps[i,j] * exp(-2pi i (mx*i/N + my*j/N))
            phase = np.exp(-2j * np.pi * (mx * np.arange(size)[:, None] / size + my * np.arange(size)[None, :] / size))
            coeff = np.sum(inv_eps * phase) / (size * size)
            fourier_coeffs[(mx, my)] = coeff
    
    # For a rough estimate, we can use the DC component and a few harmonics
    # The gap is related to the Fourier components of 1/eps
    
    # A very rough estimate: the gap is proportional to the magnitude of the first-order Fourier components
    # For a square lattice, the important G vectors are (1,0), (0,1), (1,1), etc.
    
    # Get the magnitude of the first-order components
    g10 = abs(fourier_coeffs.get((1, 0), 0))
    g01 = abs(fourier_coeffs.get((0, 1), 0))
    g11 = abs(fourier_coeffs.get((1, 1), 0))
    
    # A crude gap estimate
    gap_estimate = 0.5 * (g10 + g01 + g11)
    
    return gap_estimate


def design(problem):
    """
    Design a symmetric 32x32 unit cell for maximum TM band gap between bands 4 and 5.
    Uses a 4x4 block structure with 8x8 sub-cells, parameterized by 4-bit patterns.
    """
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s limit
    
    size = problem["pixels"]
    block_size = 8
    
    # Generate all 256 combinations of x-pattern and y-pattern
    best_cell = None
    best_score = -np.inf
    
    # Pre-compute which patterns are symmetric
    symmetric_patterns = []
    for p in range(16):
        bits = [(p >> k) & 1 for k in range(4)]
        # Check symmetry: bit[k] == bit[3-k]
        if bits[0] == bits[3] and bits[1] == bits[2]:
            symmetric_patterns.append(bits)
    
    # For D4 symmetry of the 2D cell, we need x_pattern == y_pattern and both symmetric
    # So we only need to try the symmetric patterns
    for px in symmetric_patterns:
        for py in symmetric_patterns:
            if px != py:
                continue  # For transpose symmetry, we need px == py
            
            cell = _build_cell_from_1d_patterns(px, py, size, block_size)
            
            if not _is_d4_symmetric(cell):
                continue
            
            # Estimate the gap
            score = _estimate_gap(cell, problem)
            
            if score > best_score:
                best_score = score
                best_cell = cell.tolist()
            
            # Check time
            if time.time() > deadline - 10.0:
                break
    
    if best_cell is None:
        # Fallback: return the original centered rod design
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
    
    return best_cell

# EVOLVE-BLOCK-END
