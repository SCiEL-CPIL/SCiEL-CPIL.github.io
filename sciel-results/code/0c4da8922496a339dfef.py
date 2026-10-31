import numpy as np
import science_chemistry

N = 38

def compute_energy(pos):
    """
    Compute Lennard-Jones energy.
    """
    n = len(pos)
    if n < 2:
        return 0.0
    
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    
    # Mask diagonal to avoid division by zero
    # We only need upper triangle
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    
    # LJ potential: 4 * (1/r^12 - 1/r^6)
    # r^6 = (r^2)^3
    inv_r6 = 1.0 / (d_sq ** 3)
    inv_r12 = inv_r6 ** 2
    
    e = 4.0 * (inv_r12 - inv_r6)
    return np.sum(e)

def get_fcc_fragment():
    """
    Generate 38 atoms from an FCC lattice fragment.
    """
    points = []
    # Shell 1: r^2 = 0 (1 atom)
    points.append([0.0, 0.0, 0.0])
    
    # Shell 2: r^2 = 2 (12 atoms)
    # Permutations of (1, 1, 0)
    for i in range(3):
        for j in range(3):
            if i != j:
                k = 3 - i - j
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        p = [0.0, 0.0, 0.0]
                        p[i] = s1
                        p[j] = s2
                        p[k] = 0.0
                        points.append(p)
                        
    # Shell 3: r^2 = 4 (6 atoms)
    # Permutations of (2, 0, 0)
    for i in range(3):
        for s in [1, -1]:
            p = [0.0, 0.0, 0.0]
            p[i] = s * 2.0
            points.append(p)
            
    # Shell 4: r^2 = 6 (24 atoms)
    # Permutations of (2, 1, 1)
    for i in range(3):
        for j in range(3):
            if i != j:
                k = 3 - i - j
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        for s3 in [1, -1]:
                            p = [0.0, 0.0, 0.0]
                            p[i] = s1 * 2.0
                            p[j] = s2
                            p[k] = s3
                            points.append(p)
                            
    # We have 1 + 12 + 6 + 24 = 43 points.
    # We need 38.
    # Take the first 38.
    # The first 19 are from shells 1-3.
    # The next 19 are from shell 4.
    # This selection is arbitrary but valid.
    
    selected = points[:38]
    
    # Scale to match LJ minimum distance
    # Current NN distance is sqrt(2^2 + 1^2 + 0^2) = sqrt(2) ? No.
    # Nearest neighbors are (1,1,0) type. Distance = sqrt(1+1) = sqrt(2).
    # Target NN distance = 2^(1/6).
    # Scale factor = 2^(1/6) / sqrt(2) = 2^(1/6) * 2^(-1/2) = 2^(1/6 - 3/6) = 2^(-2/6) = 2^(-1/3).
    
    scale = 2.0 ** (-1.0 / 3.0)
    return np.array(selected, dtype=float) * scale

def construct():
    # 1. Initial Guess
    pos = get_fcc_fragment()
    
    # 2. Local Relaxation
    pos = science_chemistry.relax(pos)
    energy = compute_energy(pos)
    
    # 3. Basin Hopping (if needed)
    # Target energy is approx -173.928
    # If we are close, we might skip or do fewer steps.
    # But to be safe, run a short basin hopping.
    
    rng = np.random.default_rng(42)
    best_pos = pos
    best_energy = energy
    
    max_steps = 30
    step_size = 0.5
    
    for i in range(max_steps):
        # Perturb
        noise = rng.normal(0, step_size, pos.shape)
        new_pos = pos + noise
        
        # Relax
        new_pos = science_chemistry.relax(new_pos)
        new_energy = compute_energy(new_pos)
        
        # Accept if better
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos
            pos = new_pos
            # Optional: reduce step size if we found a new minimum
            # step_size *= 0.95
            
        # Early exit if we are very close to the theoretical minimum
        if best_energy < -173.92:
            break
            
    return best_pos
