"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def design(problem):
    """Return a symmetric 32x32 photonic crystal unit cell designed for a TM band gap."""
    size = 32
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin before 120s limit
    
    # Precompute symmetry orbits: each orbit is the set of pixels related
    # under the full square symmetry group (D4: 8 elements).
    # Flipping an entire orbit preserves all required symmetries.
    orbits = _precompute_orbits(size)
    
    # Precompute a local score function for fast evaluation
    # The 4th-5th TM band gap in a square photonic crystal is related to
    # the Fourier components of epsilon at reciprocal lattice vectors.
    # For a 32-pixel cell, the fundamental periodicity that opens the
    # gap is at ~16 pixels (half the cell). We approximate the gap by
    # measuring the 2-pixel-periodic modulation of epsilon.
    
    # Initialize with a reasonable starting pattern:
    # A pattern with moderate dielectric contrast at the relevant periodicity.
    # Start with a 16-pixel-periodic pattern: alternate 8-pixel blocks.
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            # 16-pixel periodic pattern with 8-pixel features
            pi = i % 16
            pj = j % 16
            # Create a pattern: dielectric in certain regions
            # Use a simple periodic pattern that has strong Fourier component
            # at the gap-opening periodicity
            if (pi < 8 and pj < 8) or (pi >= 8 and pj >= 8):
                cell[i][j] = 1
    
    # Ensure symmetry (should already be symmetric given the periodic construction)
    # But let's verify and fix if needed
    cell = _enforce_symmetry(cell, size)
    
    best_cell = [row[:] for row in cell]
    best_score = _local_score(cell, size)
    
    # Hill-climbing search: try flipping individual orbits
    # Use simulated annealing style: accept worse moves sometimes early on
    random.seed(42)
    orbit_order = list(range(len(orbits)))
    random.shuffle(orbit_order)
    
    temp = 0.05
    min_temp = 0.001
    cooling = 0.995
    
    iteration = 0
    no_improve = 0
    
    while time.time() < deadline - 5.0:
        if iteration % 200 == 0:
            temp = max(min_temp, temp * cooling)
            random.shuffle(orbit_order)
        
        # Try a few orbit flips per iteration
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            
            oi = orbit_order[iteration % len(orbit_order)]
            orbit = orbits[oi]
            
            # Flip this orbit
            for (i, j) in orbit:
                cell[i][j] = 1 - cell[i][j]
            
            new_score = _local_score(cell, size)
            
            delta = new_score - best_score
            if delta > 0 or (delta < 0 and random.random() < math.exp(delta / max(temp, 1e-9))):
                best_cell = [row[:] for row in cell]
                best_score = new_score
                no_improve = 0
            else:
                # Revert
                for (i, j) in orbit:
                    cell[i][j] = 1 - cell[i][j]
                no_improve += 1
            
            iteration += 1
        
        if no_improve > 5000:
            # Restart with a different initial pattern
            no_improve = 0
            temp = 0.05
            cell = _random_symmetric_cell(size, orbits, random)
            best_cell = [row[:] for row in cell]
            best_score = _local_score(cell, size)
            random.shuffle(orbit_order)
    
    return best_cell

def _precompute_orbits(size):
    """Precompute all symmetry orbits under D4 group for a size x size grid."""
    visited = set()
    orbits = []
    for i in range(size):
        for j in range(size):
            if (i, j) in visited:
                continue
            # Generate orbit under D4: identity, 90, 180, 270 rotations,
            # and 4 reflections
            orbit = set()
            coords = [
                (i, j),
                (size - 1 - i, j),      # reflect x
                (i, size - 1 - j),      # reflect y
                (size - 1 - i, size - 1 - j),  # 180 rot
                (j, i),                 # transpose (reflect diagonal)
                (size - 1 - j, i),      # reflect anti-diagonal
                (j, size - 1 - i),      # 90 rot CW
                (size - 1 - j, size - 1 - i),  # 270 rot CW
            ]
            for c in coords:
                orbit.add(c)
                visited.add(c)
            orbits.append(frozenset(orbit))
    return orbits

def _enforce_symmetry(cell, size):
    """Ensure the cell has full D4 symmetry by averaging over orbits."""
    orbits = _precompute_orbits(size)
    result = [[0] * size for _ in range(size)]
    for orbit in orbits:
        # Majority vote within orbit
        val = 0
        for (i, j) in orbit:
            val += cell[i][j]
        majority = 1 if val > len(orbit) // 2 else 0
        for (i, j) in orbit:
            result[i][j] = majority
    return result

def _random_symmetric_cell(size, orbits, rng):
    """Generate a random cell with D4 symmetry."""
    cell = [[0] * size for _ in range(size)]
    for orbit in orbits:
        val = rng.randint(0, 1)
        for (i, j) in orbit:
            cell[i][j] = val
    return cell

def _local_score(cell, size):
    """
    Fast local approximation of the TM 4-5 band gap quality.
    
    The key physics: the band gap between bands 4 and 5 in a square photonic
    crystal is opened by the Fourier component of epsilon at the reciprocal
    lattice vector G = (1,0) or (0,1) in units of 2pi/a, which corresponds
    to a spatial period of a/2 = 16 pixels in our 32-pixel cell.
    
    We compute the first Fourier coefficient of epsilon at this periodicity
    and use it as a proxy for gap strength. We also penalize extreme fill
    fractions since the gap is most effective at moderate contrast.
    """
    # Compute the 16-pixel-periodic Fourier coefficient
    # epsilon(x) = 1 + 11 * cell(x)  (air=1, dielectric=12)
    # The relevant Fourier component is at k = 2*pi/(16 pixels)
    # which means we look at the sum of cell(x) * exp(-i*pi*x/8)
    
    # More precisely, for a 32-pixel cell with period 32, the Fourier
    # coefficient at G=(1,0) is:
    # c_G = (1/N^2) * sum_{i,j} eps(i,j) * exp(-i*pi*(i+j)/16)
    # But the gap is related to |c_G|^2 * (contrast)^2
    
    # Simple proxy: compute the 2-coloring modulation
    # Pattern with period 16: sign changes every 8 pixels
    # mod(i,j) = (-1)^{floor(i/8)} * (-1)^{floor(j/8)}  ... no
    # Actually period 16 means: sign = (-1)^{floor(i/16)} but that's period 32
    # Period 16: sign alternates every 16 pixels -> (-1)^{floor(i/16)}
    # Wait, in a 32-pixel cell, period 16 means the pattern repeats every 16 pixels.
    # The Fourier component at that frequency: sum cell(i,j) * cos(pi*i/8)
    # since omega = 2*pi/16 = pi/8
    
    # Let me think more carefully. Cell size = 32 pixels = lattice constant a.
    # Reciprocal lattice vector G = 2*pi/a. In pixel units, G corresponds to
    # spatial frequency 1/32 cycles per pixel.
    # But the gap between bands 4 and 5 is at the M point of the BZ,
    # which involves G = (1,0)*2*pi/a. The periodicity in real space is a/2 = 16 pixels.
    
    # So we want the Fourier component of eps at spatial frequency 1/16 cycles/pixel.
    # That's: sum_{i,j} eps(i,j) * exp(-2*pi*i/(16)) = sum eps(i,j) * exp(-pi*i/8)
    
    # But actually for a 2D gap, we need the 2D Fourier component.
    # The simplest proxy: compute the correlation with a 16-pixel-periodic pattern.
    
    # Pattern A: period 16 in x, constant in y
    # mod_A(i,j) = cos(pi * i / 8)
    
    # Pattern B: period 16 in y, constant in x  
    # mod_B(i,j) = cos(pi * j / 8)
    
    # Pattern C: checkerboard with period 16
    # mod_C(i,j) = cos(pi*i/8) * cos(pi*j/8)
    
    sx = 0.0
    sy = 0.0
    sxy = 0.0
    total = 0.0
    
    for i in range(size):
        ci = math.cos(math.pi * i / 8.0)
        for j in range(size):
            eps = 1.0 + 11.0 * cell[i][j]
            cj = math.cos(math.pi * j / 8.0)
            sx += eps * ci
            sy += eps * cj
            sxy += eps * ci * cj
            total += eps
    
    mean_eps = total / (size * size)
    
    # The gap is proportional to |c_G|^2 where c_G is the Fourier coefficient
    # c_G = (1/N^2) * sum eps(i,j) * exp(-i*G.r)
    # |c_G|^2 ~ (sx^2 + sy^2 + 2*sxy^2) / N^4  (roughly)
    
    N2 = size * size
    c_x = sx / N2
    c_y = sy / N2
    c_xy = sxy / N2
    
    # Gap strength proxy
    gap_proxy = abs(c_x) + abs(c_y) + abs(c_xy)
    
    # Penalize being too far from 50% fill
    fill = (total - N2) / (11.0 * N2)  # fraction that is dielectric
    fill_penalty = (fill - 0.5) ** 2
    
    # Also want some variation at finer scales (not just pure 16-pixel period)
    # Add a small term for the 8-pixel period component
    s8x = 0.0
    s8y = 0.0
    for i in range(size):
        ci8 = math.cos(math.pi * i / 4.0)
        for j in range(size):
            eps = 1.0 + 11.0 * cell[i][j]
            cj8 = math.cos(math.pi * j / 4.0)
            s8x += eps * ci8
            s8y += eps * cj8
    
    c8x = s8x / N2
    c8y = s8y / N2
    fine_structure = abs(c8x) + abs(c8y)
    
    score = gap_proxy - 0.3 * fill_penalty + 0.1 * fine_structure
    
    return score
# EVOLVE-BLOCK-END
