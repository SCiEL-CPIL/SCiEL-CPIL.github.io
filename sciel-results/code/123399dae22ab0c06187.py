"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
import random

def design(problem):
    """
    Design a 32x32 unit cell by optimizing the Fourier components of 1/epsilon
    using a fast FFT-based surrogate objective function, then refining with
    local search on the symmetric pixel configuration.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)
    
    # Fundamental domain indices for C4v symmetry on 32x32 grid
    # Independent pixels: i in [0, 15], j in [i, 15]
    # Total 136 bits.
    # We represent the state as a list of 136 bits.
    
    def bits_to_cell(bits):
        """Expand 136 bits to a full 32x32 symmetric cell."""
        cell = [[0 for _ in range(size)] for _ in range(size)]
        idx = 0
        for i in range(16):
            for j in range(i, 16):
                val = bits[idx]
                idx += 1
                # Symmetry:
                # 1. (i, j)
                cell[i][j] = val
                # 2. (j, i)
                cell[j][i] = val
                # 3. (31-i, j)
                cell[size - 1 - i][j] = val
                # 4. (i, 31-j)
                cell[i][size - 1 - j] = val
                # 5. (31-i, 31-j)
                cell[size - 1 - i][size - 1 - j] = val
                # 6. (31-j, i)
                cell[size - 1 - j][i] = val
                # 7. (31-j, 31-i)
                cell[size - 1 - j][size - 1 - i] = val
                # 8. (j, 31-i)
                cell[j][size - 1 - i] = val
        return cell

    def cell_to_eps(cell):
        """Convert 0/1 cell to epsilon array."""
        # 1 = dielectric (eps_high), 0 = air (eps_low)
        eps_arr = np.where(np.array(cell) == 1, eps_high, eps_low)
        return eps_arr

    def fitness(cell):
        """
        Surrogate objective: Maximize the magnitude of the 2nd order 
        Fourier coefficients of 1/epsilon.
        """
        eps_arr = cell_to_eps(cell)
        inv_eps = 1.0 / eps_arr
        
        # Compute 2D FFT
        # We care about reciprocal lattice vectors.
        # For a 32x32 grid, the FFT frequencies are 0, 1, ..., 15, -16, ..., -1.
        # The (2,0) vector corresponds to index 2 in the FFT output (if we shift).
        # np.fft.fft2 gives unshifted spectrum: [0, 1, ..., 15, -16, ..., -1]
        # We want the component corresponding to k=2.
        # Index 2 is 2. Index 30 (which is -2) is also magnitude 2.
        # Let's extract specific components.
        
        fft_inv_eps = np.fft.fft2(inv_eps)
        
        # Get magnitudes for (2,0), (0,2), (-2,0), (0,-2)
        # In unshifted FFT:
        # (2,0) is at [2, 0]
        # (0,2) is at [0, 2]
        # (-2,0) is at [30, 0] (since 32-2=30)
        # (0,-2) is at [0, 30]
        
        m_20 = np.abs(fft_inv_eps[2, 0])
        m_02 = np.abs(fft_inv_eps[0, 2])
        m_m20 = np.abs(fft_inv_eps[30, 0])
        m_0m2 = np.abs(fft_inv_eps[0, 30])
        
        # Also consider (1,1) and (1,-1) which might contribute to higher bands
        m_11 = np.abs(fft_inv_eps[1, 1])
        m_1m1 = np.abs(fft_inv_eps[1, 31])
        
        # Weighted sum. The 4-5 gap is strongly driven by (2,0) and (0,2).
        score = 2.0 * (m_20 + m_02 + m_m20 + m_0m2) + 0.5 * (m_11 + m_1m1)
        
        return score

    # Initialize bits
    # Start with the seed-like configuration or random?
    # Seed had 4 rods. Let's start with a random configuration to explore.
    # Or start with a checkerboard?
    # Let's try a few starting points.
    
    best_bits = None
    best_score = -1.0
    
    num_starts = 5
    steps_per_start = 20000
    
    for s in range(num_starts):
        if time.time() > deadline - 10.0:
            break
            
        if s == 0:
            # Try to approximate the seed: 4 rods at (8,8), (8,24), (24,8), (24,24) roughly?
            # Seed centers: -0.25 -> 0.25*32 - 16 = 8-16 = -8? No.
            # Seed x = (j+0.5)/32 - 0.5.
            # -0.25 -> (j+0.5)/32 = 0.25 -> j+0.5 = 8 -> j=7.5. So j=7 or 8.
            # Let's just use random for robustness, or a specific pattern.
            # Let's initialize with a "donut" or "holey" pattern which often has high harmonics.
            bits = [random.randint(0, 1) for _ in range(136)]
        else:
            bits = [random.randint(0, 1) for _ in range(136)]
            
        # Local Search: Simulated Annealing
        temp = 1.0
        cooling = 0.9999
        current_score = fitness(bits_to_cell(bits))
        
        for step in range(steps_per_start):
            if time.time() > deadline - 5.0:
                break
                
            # Pick a random bit to flip
            idx = random.randint(0, 135)
            old_val = bits[idx]
            bits[idx] = 1 - old_val
            
            new_cell = bits_to_cell(bits)
            new_score = fitness(new_cell)
            
            delta = new_score - current_score
            accept = False
            if delta > 0:
                accept = True
            else:
                # Accept with probability exp(delta / temp)
                if random.random() < math.exp(delta / temp):
                    accept = True
            
            if accept:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_bits = bits[:]
            else:
                # Revert
                bits[idx] = old_val
            
            temp *= cooling

    # Finalize
    if best_bits is None:
        # Fallback to seed-like
        bits = [0] * 136
        # Manually set some bits to form rods?
        # Just return random if no best found (unlikely)
        return bits_to_cell(bits)
    
    best_cell = bits_to_cell(best_bits)
    
    return best_cell

# EVOLVE-BLOCK-END
