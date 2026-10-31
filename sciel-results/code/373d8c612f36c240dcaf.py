"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import science_chemistry

def construct():
    # Generate candidate structures based on FCC truncated octahedron symmetry
    # 1 center, 12 inner shell, 24 outer shell
    
    # Inner shell: 12 atoms at permutations of (x, x, 0)
    inner_template = []
    for sx in [1, -1]:
        for sy in [1, -1]:
            inner_template.append((sx, sy, 0))
            inner_template.append((sx, 0, sy))
            inner_template.append((0, sx, sy))
    
    # Outer shell: 24 atoms
    # In FCC, the next shell after the 12 NNs consists of:
    # 6 atoms at (1, 0, 0) perms
    # 18 atoms? 
    # Actually, the 24 atoms of the truncated octahedron outer shell are:
    # 6 at (1, 0, 0) perms
    # 18 at (1, 1, 1) perms? No.
    
    # Let's use the explicit coordinates of the 38-atom truncated octahedron
    # which is a subset of the FCC lattice.
    # The 38 atoms are:
    # (0,0,0)
    # 12: (1,1,0) perms
    # 24: (2,0,0) perms (6) and (1,1,1) perms (8) and (2,1,1) perms (10)?
    
    # A simpler robust approach:
    # Use the 38 closest points in FCC, but enforce symmetry by relaxing.
    # However, the 38 closest points break symmetry.
    
    # Better: Use the known optimal structure coordinates.
    # The global minimum is a truncated octahedron.
    # We define it by two radii: r1 (inner), r2 (outer).
    # Inner 12: (r1/sqrt(2), r1/sqrt(2), 0) and perms.
    # Outer 24: (r2, 0, 0) and perms (6) AND (r2/sqrt(2), r2/sqrt(2), r2/sqrt(2)) and perms (8)?
    # 6 + 8 = 14. Not 24.
    
    # The 24 outer atoms are:
    # 6 at (r2, 0, 0)
    # 18 at (r2, r2, 0)? No.
    
    # Let's use the L-BFGS relaxation on a dense FCC cluster and pick the best.
    # We will generate 5 different FCC-like clusters with varying outer shell expansion.
    
    best_energy = np.inf
    best_positions = None
    
    # Template generation
    def get_fcc_cluster(r1, r2):
        pts = [(0, 0, 0)]
        # Inner 12
        for sx in [1, -1]:
            for sy in [1, -1]:
                pts.append((sx * r1 / np.sqrt(2), sy * r1 / np.sqrt(2), 0))
                pts.append((sx * r1 / np.sqrt(2), 0, sy * r1 / np.sqrt(2)))
                pts.append((0, sx * r1 / np.sqrt(2), sy * r1 / np.sqrt(2)))
        
        # Outer 24
        # 6 atoms on axes
        for axis in range(3):
            for s in [1, -1]:
                p = [0, 0, 0]
                p[axis] = s * r2
                pts.append(tuple(p))
        
        # 18 atoms? 
        # The remaining 18 are at (r2, r2, 0) type?
        # In FCC, the points at distance sqrt(2)*r2 from center are (1,1,0) type.
        # There are 12 such points.
        # We need 24 outer atoms total. We have 6. Need 18 more.
        # This suggests the outer shell is not just one FCC shell.
        
        # Actually, the 38-atom truncated octahedron has 24 outer atoms.
        # These are the 12 points at (1,1,0) and the 12 points at (1,0,1)?
        # No, (1,1,0) and (1,0,1) are the same set of 12 points?
        # Permutations of (1,1,0):
        # (1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0)
        # (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1)
        # (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1)
        # Total 12 points.
        
        # So we have 1 center + 12 inner + 6 axial + 12 diagonal = 31 atoms.
        # We need 7 more.
        
        # The next shell in FCC has 24 atoms.
        # It is likely that the "outer shell" of the 38-atom cluster is the 24 atoms of the second FCC shell.
        # Second FCC shell (after NN) has 6 at (1,0,0) and 12 at (1,1,0)? No.
        # NN: 12 at (1/2, 1/2, 0).
        # NNN: 6 at (1, 0, 0) and 12 at (1, 1, 0)?
        # Dist of (1,1,0) is sqrt(2). Dist of (1,0,0) is 1.
        # So they are different shells.
        
        # I will use a general relaxation starting from a symmetric guess.
        # Guess: 1 center, 12 inner at r=1, 24 outer at r=1.5.
        # The 24 outer are placed on the vertices of a truncated octahedron.
        
        # Vertices of truncated octahedron (24):
        # Permutations of (1, 1, 0) -> 12
        # Permutations of (1, 0, 1) -> Included
        # Permutations of (0, 1, 1) -> Included
        # This is only 12.
        
        # The other 12 are Permutations of (1, 1, 1)? No.
        # The other 12 are Permutations of (2, 1, 1)?
        
        # Let's use the 24 points:
        # (1, 1, 0) and perms (12)
        # (1, 0, 1) and perms (Included)
        # (0, 1, 1) and perms (Included)
        # (1, 1, 1) and perms (8)
        # (1, 0, 0) and perms (6)
        # Total 12+8+6 = 26.
        
        # I will simply use the 38 closest points in FCC, relax, and hope it finds the global min.
        # To improve odds, I will try multiple initial scales.
        
        return np.array(pts[:38], dtype=float)

    # Since the geometric construction is tricky, I will use a known good seed:
    # The 38 closest points in FCC lattice.
    # And then relax.
    
    # Generate FCC points
    fcc_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # Integer or half-integer
                if (i % 1 == 0 and j % 1 == 0 and k % 1 == 0) or \
                   (i % 1 == 0.5 and j % 1 == 0.5 and k % 1 == 0.5) or \
                   (i % 1 == -0.5 and j % 1 == -0.5 and k % 1 == -0.5):
                    fcc_points.append((i, j, k))
    
    # Sort by distance
    fcc_points.sort(key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    
    # Take 38 closest
    seed_points = np.array(fcc_points[:38], dtype=float)
    
    # Relax
    # The task provides science_chemistry.relax
    # I will try a few scales
    best_energy = np.inf
    best_pos = None
    
    for scale in [0.9, 1.0, 1.1, 1.2]:
        pos = seed_points * scale
        try:
            relaxed = science_chemistry.relax(pos)
            # Calculate energy
            # 4 * sum(r^-12 - r^-6)
            dists = np.sqrt(np.sum((relaxed[:, np.newaxis] - relaxed[np.newaxis, :]) ** 2, axis=-1))
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            pair_dists = dists[mask]
            energy = 4 * np.sum(pair_dists ** -12 - pair_dists ** -6)
            
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed
        except Exception:
            pass
            
    if best_pos is None:
        return seed_points * 1.0
        
    return best_pos
# EVOLVE-BLOCK-END
