"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Fourier-synthesis photonic crystal for large 4-5 TM band gap."""
    t0 = time.time()
    DEADLINE = 200.0  # safety margin before 230s kill
    
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    # Start with a base pattern and refine via simple local search in Fourier space
    # The key frequencies for 4-5 TM gap in square lattice are at G=(1,0), (0,1), (1,1)
    
    # Initialize with a pattern that has strong (1,0) and (1,1) Fourier components
    # Use a "checkerboard-like" pattern with modulated amplitude
    def make_cell_from_fourier(coeffs, size):
        """Convert Fourier coefficients to binary cell, preserving symmetry."""
        # coeffs: dict mapping (mx, my) -> amplitude
        cell_float = np.zeros((size, size))
        for (mx, my), amp in coeffs.items():
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size
                    y = (j + 0.5) / size
                    cell_float[i, j] += amp * np.cos(2 * np.pi * (mx * x + my * y))
        
        # Threshold to binary, ensuring symmetry
        # First, make symmetric by averaging with flips
        cell_sym = np.zeros((size, size))
        cell_sym += cell_float
        cell_sym += cell_float[::-1, :]       # vertical flip
        cell_sym += cell_float[:, ::-1]       # horizontal flip
        cell_sym += cell_float[::-1, ::-1]    # both flips
        cell_sym /= 4.0
        
        # Also enforce transpose symmetry
        cell_sym2 = (cell_sym + cell_sym.T) / 2.0
        
        # Threshold
        threshold = np.mean(cell_sym2)
        cell_bin = (cell_sym2 > threshold).astype(int)
        
        # Ensure exact symmetry
        cell_final = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                # Take the max over all symmetry operations to ensure consistency
                vals = [
                    cell_sym2[i, j],
                    cell_sym2[size-1-i, j],
                    cell_sym2[i, size-1-j],
                    cell_sym2[size-1-i, size-1-j],
                    cell_sym2[j, i],
                    cell_sym2[j, size-1-i],
                    cell_sym2[size-1-j, i],
                    cell_sym2[size-1-j, size-1-i],
                ]
                avg = np.mean(vals)
                cell_final[i, j] = 1 if avg > threshold else 0
        
        # Convert to nested list
        return cell_final.tolist()
    
    # Base Fourier coefficients: strong (1,0) and (1,1) components
    # These create the pattern that opens the 4-5 gap
    base_coeffs = {
        (1, 0): 1.0,
        (0, 1): 1.0,
        (1, 1): 0.7,
        (2, 0): 0.3,
        (0, 2): 0.3,
    }
    
    best_cell = make_cell_from_fourier(base_coeffs, size)
    best_score = -1.0  # We can't compute the actual score here, but we can try variations
    
    # Try several Fourier coefficient variations
    variations = [
        # (1,0) dominant
        {(1, 0): 1.0, (0, 1): 1.0, (1, 1): 0.5, (2, 0): 0.2, (0, 2): 0.2},
        # (1,1) dominant
        {(1, 0): 0.8, (0, 1): 0.8, (1, 1): 1.0, (2, 0): 0.4, (0, 2): 0.4},
        # Balanced
        {(1, 0): 1.0, (0, 1): 1.0, (1, 1): 0.8, (2, 0): 0.5, (0, 2): 0.5, (2, 1): 0.2, (1, 2): 0.2},
        # Higher harmonics
        {(1, 0): 1.0, (0, 1): 1.0, (1, 1): 0.6, (2, 0): 0.6, (0, 2): 0.6, (2, 2): 0.3},
        # Anisotropic
        {(1, 0): 1.2, (0, 1): 0.8, (1, 1): 0.7, (2, 0): 0.4, (0, 2): 0.2},
        # More complex
        {(1, 0): 1.0, (0, 1): 1.0, (1, 1): 0.9, (2, 0): 0.5, (0, 2): 0.5, (2, 1): 0.3, (1, 2): 0.3, (2, 2): 0.2},
    ]
    
    # Also try a "ring" pattern in real space as fallback
    def make_ring_cell(size, inner_r, outer_r):
        cell = []
        for i in range(size):
            row = []
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                r = np.sqrt(x*x + y*y)
                if inner_r < r < outer_r:
                    row.append(1)
                else:
                    row.append(0)
            cell.append(row)
        return cell
    
    # Try rings too
    ring_variations = [
        (0.15, 0.35),
        (0.1, 0.3),
        (0.2, 0.4),
        (0.1, 0.4),
    ]
    
    all_candidates = []
    for coeffs in variations:
        cell = make_cell_from_fourier(coeffs, size)
        all_candidates.append(cell)
    
    for ir, orr in ring_variations:
        cell = make_ring_cell(size, ir, orr)
        all_candidates.append(cell)
    
    # Also include the original seed
    seed_cell = []
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                row.append(1)
            else:
                row.append(0)
        seed_cell.append(row)
    all_candidates.append(seed_cell)
    
    # Simple heuristic: prefer structures with moderate fill fraction (40-60%)
    # and good symmetry
    def fill_fraction(cell):
        total = sum(sum(row) for row in cell)
        return total / (size * size)
    
    # Score candidates by a simple proxy: we want structures that are neither too empty nor too full
    # and have good Fourier content at critical frequencies
    def fourier_proxy(cell):
        """Compute a proxy score based on Fourier content at critical frequencies."""
        arr = np.array(cell, dtype=float)
        # Compute 2D FFT
        F = np.fft.fft2(arr)
        F_shifted = np.fft.fftshift(F)
        
        # Check amplitudes at (1,0), (0,1), (1,1) in pixel units
        # For a 32x32 grid, frequency 1 corresponds to index 16+1=17 or 16-1=15
        center = size // 2
        proxy = 0.0
        for (mx, my) in [(1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1)]:
            idx_i = center + mx
            idx_j = center + my
            if 0 <= idx_i < size and 0 <= idx_j < size:
                proxy += np.abs(F_shifted[idx_i, idx_j])
        
        ff = fill_fraction(cell)
        # Penalize extreme fill fractions
        ff_penalty = 1.0 - abs(ff - 0.5) * 2.0
        ff_penalty = max(0.0, ff_penalty)
        
        return proxy * ff_penalty
    
    scored = [(fourier_proxy(c), c) for c in all_candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Return the best candidate by proxy
    if scored:
        return scored[0][1]
    
    return seed_cell
# EVOLVE-BLOCK-END
