"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

# ---------------------------------------------------------------------------
# Fast TM plane-wave scorer (kept from the current program, vectorised).
# ---------------------------------------------------------------------------
def _build_fourier(inv_eps, size, indices):
    """Fourier coefficients of inv_eps for the needed G vectors."""
    x_coords = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x_coords, x_coords, indexing='ij')
    coeffs = {}
    for m in indices:
        for n in indices:
            phase = np.exp(-1j * (2 * np.pi * m * X + 2 * np.pi * n * Y))
            coeffs[(m, n)] = np.sum(inv_eps * phase) / (size * size)
    return coeffs

def _pw_solver(cell, n_pw=13, k_points=None):
    """Fast TM band solver. Returns (max_omega4, min_omega5)."""
    size = cell.shape[0]
    inv_eps = np.where(cell > 0.5, 1.0 / 12.0, 1.0)

    indices = list(range(-n_pw // 2, n_pw // 2 + 1))
    fourier_coeffs = _build_fourier(inv_eps, size, indices)

    if k_points is None:
        k_points = []
        for t in np.linspace(0, 1, 6):
            k_points.append((t * np.pi / 2, t * np.pi / 2))          # Gamma->X
        for t in np.linspace(0, 1, 6):
            k_points.append((np.pi / 2 + t * np.pi / 2,
                             np.pi / 2 - t * np.pi / 2))             # X->M
        for t in np.linspace(0, 1, 6):
            k_points.append((np.pi - t * np.pi, 0.0))                # M->Gamma
        k_points += [(0.3 * np.pi, 0.5 * np.pi),
                     (0.5 * np.pi, 0.7 * np.pi),
                     (0.2 * np.pi, 0.3 * np.pi),
                     (0.4 * np.pi, 0.15 * np.pi)]

    G_list = [(m, n) for m in indices for n in indices]
    N = len(G_list)

    max_omega4 = 0.0
    min_omega5 = np.inf

    for kx, ky in k_points:
        kxG = kx - 2 * np.pi * np.array([m for m, _ in G_list])
        kyG = ky - 2 * np.pi * np.array([n for _, n in G_list])
        A = (kxG ** 2 + kyG ** 2)
        B = np.empty((N, N), dtype=np.complex128)
        for i, (mi, ni) in enumerate(G_list):
            for j, (mj, nj) in enumerate(G_list):
                B[i, j] = fourier_coeffs.get((mi - mj, ni - nj), 0.0)
        try:
            from scipy.linalg import eig
            vals, _ = eig(A, B)
            vals_real = np.real(vals)
            vals_pos = vals_real[vals_real > 1e-10]
            vals_pos.sort()
            if len(vals_pos) >= 5:
                o4 = math.sqrt(vals_pos[3])
                o5 = math.sqrt(vals_pos[4])
                if o4 > max_omega4:
                    max_omega4 = o4
                if o5 < min_omega5:
                    min_omega5 = o5
        except Exception:
            continue

    return max_omega4, min_omega5

def _score_cell(cell):
    try:
        o4, o5 = _pw_solver(cell, n_pw=13)
        if o5 > o4 > 0:
            gap = o5 - o4
            mid = (o5 + o4) / 2.0
            if mid > 0:
                return gap / mid
    except Exception:
        pass
    return -1.0

# ---------------------------------------------------------------------------
# STRUCTURAL CHANGE: Binary Search with D4 Symmetry.
# We represent the cell as a 32x32 binary matrix.
# We exploit D4 symmetry to reduce the search space to 128 independent bits.
# We use a local search (Hill Climbing / Simulated Annealing) to optimize these bits.
# ---------------------------------------------------------------------------

def _get_orbits(size=32):
    """Precompute the orbits of pixels under D4 symmetry."""
    orbits = []
    visited = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in visited:
                continue
            orbit = set()
            # Identity
            orbit.add((i, j))
            # Reflection x
            orbit.add((i, size - 1 - j))
            # Reflection y
            orbit.add((size - 1 - i, j))
            # Reflection xy
            orbit.add((size - 1 - i, size - 1 - j))
            # Rotation 90
            orbit.add((j, size - 1 - i))
            # Rotation 90 + Ref X
            orbit.add((size - 1 - j, size - 1 - i)) # Wait, let's derive properly
            # R90: (i,j) -> (j, size-1-i)
            # R90(RefX(i,j)) = R90(i, size-1-j) = (size-1-j, size-1-i)
            # R90(RefY(i,j)) = R90(size-1-i, j) = (j, i) -- This is Transpose
            # R90(RefXY(i,j)) = R90(size-1-i, size-1-j) = (size-1-j, i)
            
            # Let's just generate all 8 explicitly to be safe
            def transform(i, j, op):
                if op == 0: return (i, j)
                if op == 1: return (i, size - 1 - j)
                if op == 2: return (size - 1 - i, j)
                if op == 3: return (size - 1 - i, size - 1 - j)
                if op == 4: return (j, size - 1 - i)
                if op == 5: return (size - 1 - j, size - 1 - i)
                if op == 6: return (size - 1 - j, i)
                if op == 7: return (j, i)
                
            for op in range(8):
                orbit.add(transform(i, j, op))
            
            for p in orbit:
                visited.add(p)
            orbits.append(list(orbit))
    return orbits

def _apply_symmetry(cell, orbits):
    """Enforce D4 symmetry by averaging the orbit values and thresholding."""
    # For binary search, we just set all pixels in an orbit to the value of the first pixel in the orbit
    # But since we are flipping whole orbits, this is consistent.
    # This function is for safety if we start from an arbitrary state.
    sym_cell = np.zeros_like(cell)
    for orbit in orbits:
        val = cell[orbit[0][0], orbit[0][1]]
        for i, j in orbit:
            sym_cell[i, j] = val
    return sym_cell

def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin

    orbits = _get_orbits(size)
    n_orbits = len(orbits)
    
    # Initialize state: List of 0/1 for each orbit
    # Strategy: Start with Dielectric Background (1s) and carve Air Holes (0s)
    # This is often better for TM gaps.
    
    best_cell = None
    best_score = -1.0
    
    # Pre-compute a few initial candidates
    # 1. All Dielectric (1)
    # 2. All Air (0)
    # 3. Central Hole (Orbits near center set to 0)
    # 4. Random
    
    rng = np.random.RandomState(42)
    
    initial_states = []
    # All 1
    initial_states.append([1] * n_orbits)
    # All 0
    initial_states.append([0] * n_orbits)
    
    # Central Hole: Set orbits where min(i,j) is close to size/2 to 0
    center = size / 2.0
    for k, orbit in enumerate(orbits):
        i, j = orbit[0]
        dist = math.sqrt((i - center + 0.5)**2 + (j - center + 0.5)**2)
        if dist < 4.0: # Rough central region
             # We need to be careful not to break symmetry, but the orbit logic handles it.
             # If ANY pixel in the orbit is close to center, it's a central orbit.
             # Actually, just check the representative.
             pass
             
    # Let's just generate random states
    for _ in range(10):
        initial_states.append([rng.randint(0, 2) for _ in range(n_orbits)])
        
    # One specific "Cross" or "Grid" like pattern?
    # For now, rely on search.

    def state_to_cell(state):
        cell = np.zeros((size, size), dtype=np.float64)
        for k, orbit in enumerate(orbits):
            val = state[k]
            for i, j in orbit:
                cell[i, j] = val
        return cell

    def evaluate_state(state):
        cell = state_to_cell(state)
        return _score_cell(cell)

    # Evaluate initial states
    for state in initial_states:
        if time.time() > deadline:
            break
        try:
            sc = evaluate_state(state)
            if sc > best_score:
                best_score = sc
                best_cell = state_to_cell(state)
        except Exception:
            continue

    # Local Search (Simulated Annealing)
    # Start from the best initial state
    if best_cell is None:
        # Fallback: All 1s
        state = [1] * n_orbits
        best_cell = state_to_cell(state)
        best_score = evaluate_state(state)
    
    # Find the state corresponding to best_cell
    # This is tricky if we just kept the cell. Let's keep the best state.
    best_state = None
    # Re-evaluate initial states to find the state index
    for state in initial_states:
        sc = evaluate_state(state)
        if sc > best_score - 1e-9: # Allow for floating point noise
             # It's possible two states have same score.
             # We should track the state that gave the best score in the first loop.
             # Let's refactor the first loop to store best_state.
             pass

    # Refactor: Track best_state in the first loop
    best_state = [1] * n_orbits # Default
    best_score = -1.0
    for state in initial_states:
        if time.time() > deadline:
            break
        try:
            sc = evaluate_state(state)
            if sc > best_score:
                best_score = sc
                best_state = state
                best_cell = state_to_cell(state)
        except Exception:
            continue

    # Annealing / Hill Climbing
    current_state = list(best_state)
    current_score = best_score
    T = 0.05 # Initial temperature
    T_min = 0.001
    
    while time.time() < deadline - 5.0:
        # Pick a random orbit to flip
        idx = rng.randint(n_orbits)
        new_state = list(current_state)
        new_state[idx] = 1 - new_state[idx]
        
        new_score = evaluate_state(new_state)
        
        delta = new_score - current_score
        
        if delta > 0 or (T > T_min and rng.random() < math.exp(delta / T)):
            current_state = new_state
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = state_to_cell(current_state)
                best_state = list(current_state)
            T *= 0.9995 # Cool down
        else:
            T *= 0.9995 # Cool down even on rejection? Or keep T?
            # Standard SA cools regardless.
            
        # Periodic restart if stuck?
        # For now, simple cooling.

    # Final safety: guarantee a valid, symmetric output.
    # best_cell is already symmetric by construction.
    result = []
    for i in range(size):
        row = [1 if best_cell[i, j] > 0.5 else 0 for j in range(size)]
        result.append(row)
    return result
# EVOLVE-BLOCK-END
