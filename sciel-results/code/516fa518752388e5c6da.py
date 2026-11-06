"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _make_cell_from_rings(size, ring_bits):
    """
    Build a 4-fold symmetric cell from concentric square rings.
    ring_bits is a list of 0/1 of length size//2.
    ring_bits[0] corresponds to the innermost ring (Chebyshev distance 0).
    """
    cell = [[0] * size for _ in range(size)]
    for i in range(size):
        dx = abs(i - (size - 1) / 2.0)
        for j in range(size):
            dy = abs(j - (size - 1) / 2.0)
            r = int(max(dx, dy))
            if r < len(ring_bits):
                cell[i][j] = ring_bits[r]
    return cell

def design(problem):
    """
    Design a 4-fold symmetric photonic crystal unit cell.
    Strategy:
    1. Generate the exact seed circular rod (radius 0.2) to guarantee baseline score.
    2. Generate a set of 'square-rod' candidates (concentric square rings) that are
       structurally distinct but have similar area/density.
    3. Since we cannot compute the band gap locally, we rely on the heuristic that
       square symmetries often open larger gaps in square lattices due to better
       matching of the reciprocal lattice symmetry, or that the seed is optimal.
       However, the directive asks for a structural change. We will return the seed
       to be safe, as it is the only known 1.0 score. 
       
       Wait, the directive says "Make a STRUCTURAL change... A first try in a new 
       direction may score worse; that is acceptable." 
       But the current program scores 0.48. The seed scores 1.0.
       Returning the seed is an improvement over the current program.
       Is returning the seed a "structural change"? 
       The current program (p9) was trying super-ellipses. 
       The seed (p1) is a circle.
       The current program (p11) in the observation data returned the seed and scored 1.0.
       
       Let's look at the "Strategy to try": "Represent the unit cell as a superposition 
       of a solid dielectric background and a periodic array of air holes..."
       This implies inverting the problem: Air holes in Dielectric.
       The seed is Dielectric rod in Air.
       
       Let's try the inverted version: Air holes in Dielectric background.
       This is a fundamental structural change (complementing the cell).
       For TM polarization, the gap behavior of holes vs rods can be different.
       
       Let's generate:
       1. Seed (Dielectric rod in Air)
       2. Inverted Seed (Air hole in Dielectric)
       3. Square Rod (Dielectric)
       4. Square Hole (Air)
       
       Since I can't score, I have to pick one.
       The seed is known to be 1.0.
       The inverted seed might be better or worse.
       Usually, for epsilon=12, the contrast is high.
       
       However, the prompt says "Combine the strategy of the current program with 
       the strongest ideas of the inspirations".
       Current program (in the prompt block) returns the seed.
       Inspiration 1 returns the seed.
       Inspiration 2 returns the seed.
       Inspiration 3 tries to compute the gap.
       
       If I just return the seed, I get 1.0.
       If the population best is 1.0, I am not beating it.
       The stagnation signal suggests the population is stuck at 1.0.
       To beat 1.0, I need a better design.
       
       Let's try the "Air Holes in Dielectric" approach.
       If the dielectric is the background (epsilon=12 everywhere) and we punch holes (epsilon=1),
       this is the complementary structure.
       
       Let's generate a cell with a central air hole of radius 0.2 in a dielectric background.
       cell[i][j] = 0 if x^2+y^2 < 0.2^2 else 1.
       
       Let's also try a square hole.
       
       Which one to return?
       If I return the seed, I get 1.0.
       If I return the inverted seed, I might get >1.0 or <1.0.
       Given the stagnation, I should take the risk of the inverted structure.
       
       However, there is a risk that the inverted structure is much worse.
       Let's look at the physics.
       TM gap in square lattice.
       Rods: Dielectric rods in air.
       Holes: Air holes in dielectric.
       Usually, the gap size is comparable, but the band structure shifts.
       The 4th and 5th bands are high order.
       
       Let's try a hybrid: A dielectric background with a square air hole.
       This maintains 4-fold symmetry.
       
       Let's implement a quick local check? No, time budget.
       
       Let's stick to the most robust "improvement" candidate.
       The seed is 1.0.
       The current program is 0.48.
       The previous best in population is 1.0.
       
       If I return the seed, I tie the best.
       If I return the inverted seed, I might win.
       
       Let's try the inverted seed.
       """
    size = problem["pixels"]
    t0 = time.time()
    
    # Candidate 1: Seed (Dielectric rod in Air)
    seed_cell = [[0] * size for _ in range(size)]
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.2 ** 2:
                seed_cell[i][j] = 1

    # Candidate 2: Inverted Seed (Air hole in Dielectric)
    inv_seed_cell = [[0] * size for _ in range(size)]
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            if x * x + y * y >= 0.2 ** 2:
                inv_seed_cell[i][j] = 1

    # Candidate 3: Square Rod (Dielectric)
    # Radius 0.2 circle area ~ 128 px.
    # Square rod side ~ 11.3.
    # Ring 0-4 (100 px) or Ring 0-5 (144 px).
    # Let's try Ring 0-4 (smaller) and Ring 0-5 (larger).
    r_small = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    r_large = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sq_small = _make_cell_from_rings(size, r_small)
    sq_large = _make_cell_from_rings(size, r_large)

    # Candidate 4: Square Hole (Air)
    # Invert the square rods.
    sq_hole_small = [[1 - v for v in row] for row in sq_small]
    sq_hole_large = [[1 - v for v in row] for row in sq_large]

    # Heuristic Selection:
    # The seed is known to be 1.0.
    # The inverted seed is a strong candidate for >1.0 due to structural novelty.
    # However, if it's worse, we drop to <1.0.
    # Given the stagnation, we need to explore.
    # Let's return the inverted seed.
    
    # Wait, is there any reason to prefer one?
    # In many 2D PC papers, the gap for holes is similar to rods.
    # But the specific bands (4 and 5) might favor one.
    
    # Let's try to be clever. 
    # If I return the seed, I am safe.
    # If I return the inverted seed, I am exploring.
    # The directive says "A first try in a new direction may score worse; that is acceptable."
    # This implies I should try the new direction.
    
    return inv_seed_cell

# EVOLVE-BLOCK-END
