"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 unit cell maximizing the TM band-4/5 gap-midgap ratio."""
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12)
    
    t0 = time.time()
    deadline = t0 + 220.0  # Safety margin before 230s limit

    half = size // 2  # 16
    N = size  # 32

    # --- Symmetric cell construction from 16x16 upper-triangle quadrant ---
    def build_cell_from_quadrant(quadrant):
        """Build full 32x32 cell from 16x16 upper-triangle quadrant (i<=j)."""
        cell = [[0]*N for _ in range(N)]
        for i in range(half):
            for j in range(i, half):
                val = quadrant[i][j]
                cell[i][j] = val
                cell[i][N-1-j] = val
                cell[N-1-i][j] = val
                cell[N-1-i][N-1-j] = val
                cell[j][i] = val
                cell[j][N-1-i] = val
                cell[N-1-j][i] = val
                cell[N-1-j][N-1-i] = val
        return cell

    # --- Fast 5x5 PWE solver (25 plane waves) for rapid evaluation ---
    max_g = 2  # 5x5 = 25 plane waves
    g_indices = [(mx, my) for mx in range(-max_g, max_g+1) for my in range(-max_g, max_g+1)]
    num_g = len(g_indices)
    G_vecs = np.array([(2*math.pi*mx, 2*math.pi*my) for mx, my in g_indices])
    
    # Key k-points in irreducible BZ (Gamma-M-X-Gamma path)
    k_points = [
        (0.0, 0.0),           # Gamma
        (math.pi/4, math.pi/4),  # M
        (0.0, math.pi),       # X
        (math.pi/2, 0.0),     # X'
        (math.pi/6, math.pi/3),
        (0.0, math.pi/2),
        (math.pi/3, math.pi/3),
        (math.pi/8, math.pi/8),
        (0.0, 2*math.pi/3),
        (math.pi/4, 0.0),
    ]

    def compute_eps_inv_fft(cell):
        """Compute 2D FFT of 1/epsilon for fast Fourier coefficient lookup."""
        eps_inv = np.ones((N, N))
        for i in range(N):
            for j in range(N):
                if cell[i][j] == 1:
                    eps_inv[i, j] = 1.0 / eps_high
        return np.fft.fft2(eps_inv) / (N * N)

    def evaluate_gap_fast(cell, fft_coeff):
        """Fast evaluation using 5x5 PWE basis."""
        # Precompute eps_inv matrix for this cell
        eps_inv_mat = np.zeros((num_g, num_g))
        for p in range(num_g):
            mx_p, my_p = g_indices[p]
            for q in range(num_g):
                mx_q, my_q = g_indices[q]
                dm = (mx_p - mx_q) % N
                dn = (my_p - my_q) % N
                eps_inv_mat[p, q] = fft_coeff[dn, dm]
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            kG_sq = (kx + G_vecs[:, 0])**2 + (ky + G_vecs[:, 1])**2
            M = eps_inv_mat * kG_sq[None, :]
            
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                continue
            
            eigvals = np.sort(eigvals)
            if len(eigvals) < 5:
                continue
            
            omega4 = math.sqrt(max(eigvals[3], 0))
            omega5 = math.sqrt(max(eigvals[4], 0))
            
            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == float('inf') or max_omega4 <= 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0
        
        return gap / midgap

    def evaluate_gap_medium(cell, fft_coeff):
        """Medium-accuracy evaluation using 7x7 PWE basis (49 plane waves)."""
        mg = 3
        gi = [(mx, my) for mx in range(-mg, mg+1) for my in range(-mg, mg+1)]
        ng = len(gi)
        gv = np.array([(2*math.pi*mx, 2*math.pi*my) for mx, my in gi])
        
        eps_mat = np.zeros((ng, ng))
        for p in range(ng):
            mx_p, my_p = gi[p]
            for q in range(ng):
                mx_q, my_q = gi[q]
                dm = (mx_p - mx_q) % N
                dn = (my_p - my_q) % N
                eps_mat[p, q] = fft_coeff[dn, dm]
        
        max_o4 = 0.0
        min_o5 = float('inf')
        
        for kx, ky in k_points:
            kG_sq = (kx + gv[:, 0])**2 + (ky + gv[:, 1])**2
            M = eps_mat * kG_sq[None, :]
            try:
                ev = np.linalg.eigvalsh(M)
            except:
                continue
            ev = np.sort(ev)
            if len(ev) < 5:
                continue
            o4 = math.sqrt(max(ev[3], 0))
            o5 = math.sqrt(max(ev[4], 0))
            max_o4 = max(max_o4, o4)
            min_o5 = min(min_o5, o5)
        
        if min_o5 == float('inf') or max_o4 <= 0:
            return -1.0
        gap = min_o5 - max_o4
        midgap = (min_o5 + max_o4) / 2.0
        if midgap <= 0:
            return -1.0
        return gap / midgap

    # --- Initial designs: various ring patterns ---
    def make_ring_quadrant(r_inner, r_outer, thickness=1.0):
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / N
                y = (i + 0.5) / N
                r = math.sqrt(x*x + y*y)
                if r_inner <= r <= r_outer:
                    q[i][j] = 1
        return q

    def make_cross_quadrant(width):
        """Cross/plus pattern in the quadrant."""
        q = [[0]*half for _ in range(half)]
        center = half - 1
        for i in range(half):
            for j in range(i, half):
                # Horizontal bar
                if abs(i - center) <= width // 2:
                    q[i][j] = 1
                # Vertical bar (only for j >= i in upper triangle)
                elif abs(j - center) <= width // 2 and i >= center:
                    q[i][j] = 1
        return q

    def make_checkerboard_quadrant(period, offset=0):
        q = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                if (i + j + offset) % period < period // 2:
                    q[i][j] = 1
        return q

    # --- Genetic Algorithm ---
    rng = random.Random(42)
    
    # Initialize population with diverse designs
    initial_designs = []
    for ri, ro in [(0.1, 0.3), (0.15, 0.35), (0.2, 0.4), (0.25, 0.45), 
                   (0.12, 0.28), (0.18, 0.38), (0.22, 0.42), (0.08, 0.25)]:
        initial_designs.append(make_ring_quadrant(ri, ro))
    for w in [1, 2, 3]:
        initial_designs.append(make_cross_quadrant(w))
    for p in [4, 6, 8]:
        for off in [0, 1]:
            initial_designs.append(make_checkerboard_quadrant(p, off))
    
    # Add some random designs
    for _ in range(10):
        q = [[rng.randint(0, 1) for j in range(i, half)] for i in range(half)]
        initial_designs.append(q)

    pop_size = 40
    population = initial_designs[:pop_size]
    while len(population) < pop_size:
        q = [[rng.randint(0, 1) for j in range(i, half)] for i in range(half)]
        population.append(q)

    # Evaluate initial population
    def eval_population(pop):
        scores = []
        for q in pop:
            if time.time() > deadline - 60:
                scores.append(-1.0)
                continue
            cell = build_cell_from_quadrant(q)
            fft = compute_eps_inv_fft(cell)
            score = evaluate_gap_fast(cell, fft)
            scores.append(score)
        return scores

    scores = eval_population(population)
    
    best_score = max(scores)
    best_idx = scores.index(best_score)
    best_quadrant = [row[:] for row in population[best_idx]]
    
    # Track best with medium-accuracy verification
    best_medium_score = -1.0
    best_medium_quadrant = [row[:] for row in best_quadrant]
    
    def verify_medium(quadrant):
        """Verify a design with medium-accuracy PWE."""
        cell = build_cell_from_quadrant(quadrant)
        fft = compute_eps_inv_fft(cell)
        return evaluate_gap_medium(cell, fft)

    best_medium_score = verify_medium(best_quadrant)
    best_medium_quadrant = [row[:] for row in best_quadrant]

    # --- GA operators ---
    def crossover(q1, q2):
        """Uniform crossover on the upper-triangle quadrant."""
        child = [[0]*half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                if rng.random() < 0.5:
                    child[i][j] = q1[i][j]
                else:
                    child[i][j] = q2[i][j]
        return child

    def mutate(quadrant, rate=0.05):
        """Bit-flip mutation on the upper-triangle quadrant."""
        child = [row[:] for row in quadrant]
        for i in range(half):
            for j in range(i, half):
                if rng.random() < rate:
                    child[i][j] = 1 - child[i][j]
        return child

    def tournament_select(pop, scores, k=3):
        """Tournament selection."""
        indices = rng.sample(range(len(pop)), min(k, len(pop)))
        best_i = max(indices, key=lambda x: scores[x])
        return [row[:] for row in pop[best_i]]

    # --- Main GA loop ---
    gen = 0
    max_gens = 50
    elite_size = 5
    
    while gen < max_gens:
        if time.time() > deadline - 80:
            break
        
        gen += 1
        
        # Sort population by score
        sorted_idx = sorted(range(len(population)), key=lambda x: scores[x], reverse=True)
        population = [population[i] for i in sorted_idx]
        scores = [scores[i] for i in sorted_idx]
        
        # Keep elites
        new_pop = [row[:] for row in population[:elite_size]]
        new_scores = scores[:elite_size]
        
        # Fill rest with offspring
        while len(new_pop) < pop_size:
            if time.time() > deadline - 75:
                break
            
            p1 = tournament_select(population, scores)
            p2 = tournament_select(population, scores)
            child = crossover(p1, p2)
            child = mutate(child, rate=0.03 + 0.02 * (gen / max_gens))
            new_pop.append(child)
        
        # Evaluate new population
        if time.time() > deadline - 70:
            break
        
        new_scores = eval_population(new_pop)
        population = new_pop
        scores = new_scores
        
        # Check for improvement
        current_best = max(scores)
        current_best_idx = scores.index(current_best)
        
        if current_best > best_score:
            best_score = current_best
            best_quadrant = [row[:] for row in population[current_best_idx]]
            
            # Verify with medium accuracy periodically
            if gen % 5 == 0 or current_best > best_medium_score + 0.01:
                med_score = verify_medium(best_quadrant)
                if med_score > best_medium_score:
                    best_medium_score = med_score
                    best_medium_quadrant = [row[:] for row in best_quadrant]

    # --- Final local search refinement on the best design ---
    current_quadrant = [row[:] for row in best_medium_quadrant]
    current_score = best_medium_score
    current_cell = build_cell_from_quadrant(current_quadrant)
    current_fft = compute_eps_inv_fft(current_cell)
    
    positions = [(i, j) for i in range(half) for j in range(i, half)]
    
    for sweep in range(10):
        if time.time() > deadline - 20:
            break
        
        shuffled_pos = positions[:]
        rng.shuffle(shuffled_pos)
        
        improved = False
        for (i, j) in shuffled_pos:
            if time.time() > deadline - 15:
                break
            
            old_val = current_quadrant[i][j]
            current_quadrant[i][j] = 1 - old_val
            new_cell = build_cell_from_quadrant(current_quadrant)
            new_fft = compute_eps_inv_fft(new_cell)
            new_score = evaluate_gap_medium(new_cell, new_fft)
            
            if new_score > current_score:
                current_score = new_score
                current_cell = new_cell
                current_fft = new_fft
                improved = True
                if new_score > best_medium_score:
                    best_medium_score = new_score
                    best_medium_quadrant = [row[:] for row in current_quadrant]
            else:
                current_quadrant[i][j] = old_val
        
        if not improved:
            # Try random perturbations
            for _ in range(3):
                if time.time() > deadline - 15:
                    break
                for _ in range(3):
                    i, j = rng.choice(positions)
                    current_quadrant[i][j] = 1 - current_quadrant[i][j]
                new_cell = build_cell_from_quadrant(current_quadrant)
                new_fft = compute_eps_inv_fft(new_cell)
                new_score = evaluate_gap_medium(new_cell, new_fft)
                if new_score > current_score:
                    current_score = new_score
                    current_cell = new_cell
                    current_fft = new_fft
                    if new_score > best_medium_score:
                        best_medium_score = new_score
                        best_medium_quadrant = [row[:] for row in current_quadrant]
                else:
                    for _ in range(3):
                        i, j = rng.choice(positions)
                        current_quadrant[i][j] = 1 - current_quadrant[i][j]

    # Build final cell from the best design
    final_cell = build_cell_from_quadrant(best_medium_quadrant)
    
    # Verify symmetry
    for i in range(N):
        for j in range(N):
            if final_cell[i][j] != final_cell[N-1-i][j]:
                final_cell[N-1-i][j] = final_cell[i][j]
            if final_cell[i][j] != final_cell[i][N-1-j]:
                final_cell[i][N-1-j] = final_cell[i][j]
            if final_cell[i][j] != final_cell[j][i]:
                final_cell[j][i] = final_cell[i][j]
    
    return final_cell
# EVOLVE-BLOCK-END
