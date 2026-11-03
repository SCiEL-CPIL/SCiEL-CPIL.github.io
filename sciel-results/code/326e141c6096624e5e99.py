import math
import time
import random
import numpy as np

N = 11
APOTHEM = math.sqrt(3) / 2.0
SQRT3 = math.sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

# Axial coordinate basis for unit hexagons (flat-top orientation)
# Adjacent hexagons in axial coords (q, r):
#   east:  (+1,  0)  -> dx = sqrt(3), dy = 0
#   NE:    (+1, -1)  -> dx = sqrt(3)/2, dy = 3/2
#   NW:    ( 0, -1)  -> dx = -sqrt(3)/2, dy = 3/2
#   west:  (-1,  0)  -> dx = -sqrt(3), dy = 0
#   SW:    (-1, +1)  -> dx = -sqrt(3)/2, dy = -3/2
#   SE:    ( 0, +1)  -> dx = sqrt(3)/2, dy = -3/2

AXIAL_DIRS = [
    (1, 0), (1, -1), (0, -1),
    (-1, 0), (-1, 1), (0, 1)
]

def axial_to_xy(q, r):
    x = SQRT3 * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def xy_to_axial(x, y):
    r = y / 1.5
    q = (x / SQRT3) - r / 2.0
    return round(q), round(r)

def valid_packing(cells):
    """Check that all cells are distinct and form a connected set."""
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False
    if not cells:
        return True
    # BFS connectivity
    visited = {cells[0]}
    frontier = [cells[0]]
    while frontier:
        q, r = frontier.pop()
        for dq, dr in AXIAL_DIRS:
            nb = (q + dq, r + dr)
            if nb in cell_set and nb not in visited:
                visited.add(nb)
                frontier.append(nb)
    return len(visited) == len(cells)

def compute_S(cells, beta):
    """Compute the enclosing hexagon side length for a set of axial cells at rotation beta.
    The inner hexagons are in standard orientation (angle 0). The outer hexagon is rotated by beta.
    S = (max_proj + APOTHEM) / APOTHEM where max_proj is the support function of the cluster
    relative to the outer hexagon's center."""
    pts = np.array([axial_to_xy(q, r) for q, r in cells])
    # Center of mass
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    pts_c = pts - np.array([cx, cy])
    
    # Outer hexagon normals at rotation beta
    # For a regular hexagon with side S and rotation beta, the apothem is S * sqrt(3)/2
    # The six outward normals are at angles beta + pi/6 + k*pi/3
    normals = np.array([
        [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
        for k in range(6)
    ])
    
    projs = pts_c @ normals.T
    max_abs = np.max(np.abs(projs))
    # The outer hexagon must contain all inner hexagons.
    # Each inner hexagon has apothem APOTHEM. So the outer hexagon's apothem must be >= max_proj + APOTHEM.
    # Outer apothem = S * APOTHEM (since S is side length and apothem = S*sqrt(3)/2)
    # So S * APOTHEM >= max_proj + APOTHEM
    # S >= (max_proj + APOTHEM) / APOTHEM
    S = (max_abs + APOTHEM) / APOTHEM
    return S, cx, cy

def golden_section_minimize(f, a, b, tol=1e-8, max_iter=100):
    """Golden section search for minimum of f on [a, b]."""
    gr = (math.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
    return min(fc, fd), (a + b) / 2

def best_S_for_packing(cells):
    """Find the best rotation angle for a given packing using golden section search."""
    def f(beta):
        S, _, _ = compute_S(cells, beta)
        return S
    
    # Search over one period of the hexagonal symmetry: [0, pi/6)
    # But the function is periodic with period pi/6 for the outer hexagon
    # Actually the outer hexagon has 6-fold symmetry, so period is pi/3
    # But we also need to consider that the inner hexagons are fixed orientation
    # The combined system has period pi/6 (since inner has 6-fold, outer has 6-fold,
    # but they're offset)
    # Let's search [0, pi/6]
    best_S = float('inf')
    best_beta = 0.0
    
    # Coarse search first
    n_coarse = 36
    for i in range(n_coarse):
        beta = i * (math.pi / 6.0) / n_coarse
        S, _, _ = compute_S(cells, beta)
        if S < best_S:
            best_S = S
            best_beta = beta
    
    # Fine golden section around the best
    step = (math.pi / 6.0) / n_coarse
    a = max(0.0, best_beta - step)
    b = min(math.pi / 6.0, best_beta + step)
    if b > a:
        S_gs, beta_gs = golden_section_minimize(f, a, b, tol=1e-10, max_iter=50)
        if S_gs < best_S:
            best_S = S_gs
            best_beta = beta_gs
    
    return best_S, best_beta

def initial_populations():
    """Generate diverse initial packings of 11 connected hexagons on axial lattice."""
    pop = []
    
    # Star / flower: center + 6 neighbors + 4 more
    # Center (0,0), ring 1: all 6 dirs
    # Ring 2: pick 4 from 12 positions
    star = [(0, 0)]
    for dq, dr in AXIAL_DIRS:
        star.append((dq, dr))
    # Add 4 from ring 2
    ring2_candidates = []
    for dq, dr in [(2, 0), (2, -1), (1, -2), (0, -2), (-1, -1), (-2, 0),
                   (-2, 1), (-1, 2), (0, 2), (1, 1), (2, 1), (1, 0)]:
        # Only add if adjacent to existing
        ring2_candidates.append((dq, dr))
    # Pick 4 spread out
    for idx in [0, 2, 4, 6]:
        if idx < len(ring2_candidates):
            star.append(ring2_candidates[idx])
    if valid_packing(star) and len(star) == 11:
        pop.append(star)
    
    # Linear-ish: 3-4-3-1 pattern
    linear = [(0, 0), (1, 0), (2, 0),
              (1, -1), (2, -1), (3, -1), (4, -1),
              (1, -2), (2, -2), (3, -2),
              (2, -3)]
    if valid_packing(linear) and len(linear) == 11:
        pop.append(linear)
    
    # 4-3-4
    rect = [(0, 0), (1, 0), (2, 0), (3, 0),
            (1, -1), (2, -1), (3, -1),
            (0, -2), (1, -2), (2, -2), (3, -2)]
    if valid_packing(rect) and len(rect) == 11:
        pop.append(rect)
    
    # 2-3-3-3
    diamond = [(0, 0), (1, 0),
               (0, -1), (1, -1), (2, -1),
               (0, -2), (1, -2), (2, -2),
               (0, -3), (1, -3), (2, -3)]
    if valid_packing(diamond) and len(diamond) == 11:
        pop.append(diamond)
    
    # 5-3-3
    wide = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
            (1, -1), (2, -1), (3, -1),
            (1, -2), (2, -2), (3, -2)]
    if valid_packing(wide) and len(wide) == 11:
        pop.append(wide)
    
    # 3-5-3
    tri = [(0, 0), (1, 0), (2, 0),
           (0, -1), (1, -1), (2, -1), (3, -1), (4, -1),
           (0, -2), (1, -2), (2, -2)]
    if valid_packing(tri) and len(tri) == 11:
        pop.append(tri)
    
    # 4-4-3
    block = [(0, 0), (1, 0), (2, 0), (3, 0),
             (0, -1), (1, -1), (2, -1), (3, -1),
             (0, -2), (1, -2), (2, -2)]
    if valid_packing(block) and len(block) == 11:
        pop.append(block)
    
    # 5-4-2
    asym = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
            (0, -1), (1, -1), (2, -1), (3, -1),
            (0, -2), (1, -2)]
    if valid_packing(asym) and len(asym) == 11:
        pop.append(asym)
    
    # 3-4-4
    asym2 = [(0, 0), (1, 0), (2, 0),
             (0, -1), (1, -1), (2, -1), (3, -1),
             (0, -2), (1, -2), (2, -2), (3, -2)]
    if valid_packing(asym2) and len(asym2) == 11:
        pop.append(asym2)
    
    # 2-5-4
    asym3 = [(0, 0), (1, 0),
             (0, -1), (1, -1), (2, -1), (3, -1), (4, -1),
             (0, -2), (1, -2), (2, -2), (3, -2)]
    if valid_packing(asym3) and len(asym3) == 11:
        pop.append(asym3)
    
    # 1-5-5
    row = [(0, 0),
           (0, -1), (1, -1), (2, -1), (3, -1), (4, -1),
           (0, -2), (1, -2), (2, -2), (3, -2), (4, -2)]
    if valid_packing(row) and len(row) == 11:
        pop.append(row)
    
    # 6-5
    two_row = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
               (0, -1), (1, -1), (2, -1), (3, -1), (4, -1)]
    if valid_packing(two_row) and len(two_row) == 11:
        pop.append(two_row)
    
    # 7-4
    seven = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
             (1, -1), (2, -1), (3, -1), (4, -1)]
    if valid_packing(seven) and len(seven) == 11:
        pop.append(seven)
    
    # 8-3
    eight = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
             (2, -1), (3, -1), (4, -1)]
    if valid_packing(eight) and len(eight) == 11:
        pop.append(eight)
    
    # 9-2
    nine = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0),
            (3, -1), (4, -1)]
    if valid_packing(nine) and len(nine) == 11:
        pop.append(nine)
    
    # 10-1
    ten = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),
           (4, -1)]
    if valid_packing(ten) and len(ten) == 11:
        pop.append(ten)
    
    # 11 in a line
    line = [(i, 0) for i in range(11)]
    if valid_packing(line) and len(line) == 11:
        pop.append(line)
    
    # Add some random connected shapes
    for _ in range(20):
        random_shape = [(0, 0)]
        attempts = 0
        while len(random_shape) < N and attempts < 500:
            attempts += 1
            base = random.choice(random_shape)
            dq, dr = random.choice(AXIAL_DIRS)
            new_cell = (base[0] + dq, base[1] + dr)
            if new_cell not in random_shape:
                random_shape.append(new_cell)
        if len(random_shape) == N and valid_packing(random_shape):
            pop.append(random_shape)
    
    # Remove duplicates
    unique_pop = []
    seen = set()
    for p in pop:
        key = frozenset(p)
        if key not in seen:
            seen.add(key)
            unique_pop.append(p)
    
    return unique_pop

def mutate(cells):
    """Mutate a packing by moving one cell to a new adjacent position."""
    cells = list(cells)
    idx = random.randrange(len(cells))
    old_cell = cells[idx]
    # Try to move to a random neighbor
    for _ in range(20):
        dq, dr = random.choice(AXIAL_DIRS)
        new_cell = (old_cell[0] + dq, old_cell[1] + dr)
        if new_cell not in cells:
            cells[idx] = new_cell
            if valid_packing(cells) and len(set(cells)) == len(cells):
                return cells
    # If can't move, try adding a new cell and removing another
    for _ in range(20):
        base = random.choice(cells)
        dq, dr = random.choice(AXIAL_DIRS)
        new_cell = (base[0] + dq, base[1] + dr)
        if new_cell not in cells:
            remove_idx = random.randrange(len(cells))
            if remove_idx != idx:
                cells[remove_idx] = new_cell
                if valid_packing(cells) and len(set(cells)) == len(cells):
                    return cells
    return cells

def crossover(cells1, cells2):
    """Simple crossover: take a prefix from one and suffix from another, then repair."""
    # This is tricky for connected shapes; just return a random mix
    mixed = list(cells1)
    for _ in range(random.randint(1, 3)):
        idx = random.randrange(len(mixed))
        dq, dr = random.choice(AXIAL_DIRS)
        new_cell = (mixed[idx][0] + dq, mixed[idx][1] + dr)
        if new_cell not in mixed:
            mixed[idx] = new_cell
    if valid_packing(mixed) and len(set(mixed)) == len(mixed):
        return mixed
    return list(cells1)

def genetic_search(initial_pop, pop_size=50, generations=200):
    """Run a genetic algorithm to find the best packing."""
    # Evaluate initial population
    scored = []
    for cells in initial_pop:
        S, beta = best_S_for_packing(cells)
        scored.append((S, beta, cells))
    scored.sort(key=lambda x: x[0])
    
    # Keep top pop_size
    pop = scored[:pop_size]
    best_S = pop[0][0]
    best_beta = pop[0][1]
    best_cells = pop[0][2]
    
    for gen in range(generations):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        new_pop = []
        
        # Elitism: keep top 20%
        elite_count = max(1, pop_size // 5)
        new_pop.extend(pop[:elite_count])
        
        # Fill rest with offspring
        while len(new_pop) < pop_size:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Tournament selection
            parent1 = random.choice(pop[:max(1, pop_size // 2)])
            parent2 = random.choice(pop[:max(1, pop_size // 2)])
            
            if random.random() < 0.3:
                child = crossover(parent1[2], parent2[2])
            else:
                child = mutate(parent1[2])
            
            if child is None or len(child) != N or not valid_packing(child):
                continue
            
            S, beta = best_S_for_packing(child)
            new_pop.append((S, beta, child))
            
            if S < best_S:
                best_S = S
                best_beta = beta
                best_cells = child
        
        # Update population
        new_pop.sort(key=lambda x: x[0])
        pop = new_pop[:pop_size]
    
    return best_S, best_beta, best_cells

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: return a simple valid configuration
        cells = [(0, 0)] + [(dq, dr) for dq, dr in AXIAL_DIRS] + [(2, 0), (0, -2), (-2, 0), (0, 2)]
        if not valid_packing(cells) or len(cells) != 11:
            cells = [(i, 0) for i in range(11)]
        S, beta = best_S_for_packing(cells)
        pts = [axial_to_xy(q, r) for q, r in cells]
        cx = np.mean([p[0] for p in pts])
        cy = np.mean([p[1] for p in pts])
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [float(cx), float(cy)], float(S), float(math.degrees(beta))
    
    # Generate initial populations
    initial_pop = initial_populations()
    
    # Run genetic search
    best_S, best_beta, best_cells = genetic_search(initial_pop, pop_size=40, generations=150)
    
    # Also try some direct evaluations of well-known patterns
    known_patterns = [
        # Flower + 4
        [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
         (2, 0), (2, -1), (1, -2), (0, -2)],
        # 3-4-3-1
        [(0, 0), (1, 0), (2, 0),
         (1, -1), (2, -1), (3, -1), (4, -1),
         (1, -2), (2, -2), (3, -2),
         (2, -3)],
        # 4-3-4
        [(0, 0), (1, 0), (2, 0), (3, 0),
         (1, -1), (2, -1), (3, -1),
         (0, -2), (1, -2), (2, -2), (3, -2)],
        # 2-3-3-3
        [(0, 0), (1, 0),
         (0, -1), (1, -1), (2, -1),
         (0, -2), (1, -2), (2, -2),
         (0, -3), (1, -3), (2, -3)],
        # 5-3-3
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
         (1, -1), (2, -1), (3, -1),
         (1, -2), (2, -2), (3, -2)],
        # 3-5-3
        [(0, 0), (1, 0), (2, 0),
         (0, -1), (1, -1), (2, -1), (3, -1), (4, -1),
         (0, -2), (1, -2), (2, -2)],
        # 4-4-3
        [(0, 0), (1, 0), (2, 0), (3, 0),
         (0, -1), (1, -1), (2, -1), (3, -1),
         (0, -2), (1, -2), (2, -2)],
        # 5-4-2
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
         (0, -1), (1, -1), (2, -1), (3, -1),
         (0, -2), (1, -2)],
        # 3-4-4
        [(0, 0), (1, 0), (2, 0),
         (0, -1), (1, -1), (2, -1), (3, -1),
         (0, -2), (1, -2), (2, -2), (3, -2)],
        # 2-5-4
        [(0, 0), (1, 0),
         (0, -1), (1, -1), (2, -1), (3, -1), (4, -1),
         (0, -2), (1, -2), (2, -2), (3, -2)],
        # 1-5-5
        [(0, 0),
         (0, -1), (1, -1), (2, -1), (3, -1), (4, -1),
         (0, -2), (1, -2), (2, -2), (3, -2), (4, -2)],
        # 6-5
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
         (0, -1), (1, -1), (2, -1), (3, -1), (4, -1)],
        # 7-4
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
         (1, -1), (2, -1), (3, -1), (4, -1)],
        # 8-3
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
         (2, -1), (3, -1), (4, -1)],
        # 9-2
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0),
         (3, -1), (4, -1)],
        # 10-1
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),
         (4, -1)],
        # 11-line
        [(i, 0) for i in range(11)],
    ]
    
    for cells in known_patterns:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        if valid_packing(cells) and len(cells) == N:
            S, beta = best_S_for_packing(cells)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_cells = cells
    
    # Final: compute the result
    pts = [axial_to_xy(q, r) for q, r in best_cells]
    cx = np.mean([p[0] for p in pts])
    cy = np.mean([p[1] for p in pts])
    
    inner = [[float(x), float(y), 0.0] for x, y in pts]
    angle_deg = float(math.degrees(best_beta))
    
    return inner, [float(cx), float(cy)], float(best_S), angle_deg

