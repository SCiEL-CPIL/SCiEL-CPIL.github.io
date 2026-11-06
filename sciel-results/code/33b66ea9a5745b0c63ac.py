"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Strategy:
    1. Precompute constraint masks for fast residue validity (bit operations)
    2. Greedy initial construction with fast residue search
    3. Hill climbing with efficient add/swap/residue-reassignment
    4. Simulated annealing with temperature schedule using fast validity checks
    5. Final polish
    """
    deadline = time.time() + 140.0
    random.seed(42)
    
    def _time_left():
        return deadline - time.time()
    
    # Precompute gcd table for all pairs of moduli in [20, 200]
    # gcd_table[i][j] = gcd(i+20, j+20) for i,j in 0..179
    MOD_MIN = 20
    MOD_MAX = 200
    N_MOD = MOD_MAX - MOD_MIN + 1  # 181
    
    # Build gcd lookup: gcd_val[m1_idx][m2_idx]
    # Use a dict keyed by (m1, m2) for simplicity, or a 2D array
    # Since moduli are 20..200, we can use a flat array indexed by (m1-20, m2-20)
    gcd_arr = [[0] * N_MOD for _ in range(N_MOD)]
    for i in range(N_MOD):
        for j in range(N_MOD):
            gcd_arr[i][j] = gcd(MOD_MIN + i, MOD_MIN + j)
    
    def is_valid_pair(m1, r1, m2, r2):
        """Check if two residue classes are disjoint."""
        g = gcd_arr[m1 - MOD_MIN][m2 - MOD_MIN]
        return (r1 % g) != (r2 % g)
    
    def find_valid_residue(m, classes):
        """Find a residue r in [0, m) such that [m, r] is disjoint from all classes.
        Uses precomputed gcd for speed."""
        m_idx = m - MOD_MIN
        for r in range(m):
            valid = True
            for m2, r2 in classes:
                g = gcd_arr[m_idx][m2 - MOD_MIN]
                if (r % g) == (r2 % g):
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def find_valid_residue_fast(m, classes, m_indices, r_values):
        """Find valid residue using pre-extracted arrays for speed."""
        m_idx = m - MOD_MIN
        n = len(classes)
        for r in range(m):
            valid = True
            for i in range(n):
                g = gcd_arr[m_idx][m_indices[i]]
                if (r % g) == (r_values[i] % g):
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def calc_score(classes):
        """Calculate the sum of 1/m as a float."""
        return sum(1.0 / m for m, r in classes)
    
    def try_add(classes, m):
        """Try to add modulus m with a valid residue. Returns new classes or None."""
        r = find_valid_residue(m, classes)
        if r is not None:
            return classes + [[m, r]]
        return None
    
    def check_residue_valid(m, r, classes, skip_idx=-1):
        """Check if residue r for modulus m is valid against all classes except skip_idx."""
        m_idx = m - MOD_MIN
        for i, (m2, r2) in enumerate(classes):
            if i == skip_idx:
                continue
            g = gcd_arr[m_idx][m2 - MOD_MIN]
            if (r % g) == (r2 % g):
                return False
        return True
    
    def hill_climb(classes, time_limit=3.0):
        """Hill climbing polish with efficient operations."""
        improved = True
        start = time.time()
        while improved and time.time() - start < time_limit:
            improved = False
            current_moduli = set(c[0] for c in classes)
            missing = [m for m in range(MOD_MIN, MOD_MAX + 1) if m not in current_moduli]
            
            # Try adding
            for m in missing:
                if time.time() - start > time_limit:
                    break
                new_classes = try_add(classes, m)
                if new_classes is not None:
                    classes = new_classes
                    improved = True
            
            # Try swapping: replace a modulus with a missing one of higher weight
            current_moduli = set(c[0] for c in classes)
            missing = [m for m in range(MOD_MIN, MOD_MAX + 1) if m not in current_moduli]
            # Sort missing by 1/m descending
            missing.sort(key=lambda m: -1.0 / m)
            
            for m_add in missing:
                if time.time() - start > time_limit:
                    break
                for m_remove in sorted(current_moduli, key=lambda m: 1.0 / m):
                    if time.time() - start > time_limit:
                        break
                    if 1.0 / m_add <= 1.0 / m_remove:
                        continue
                    new_classes = [c for c in classes if c[0] != m_remove]
                    r = find_valid_residue(m_add, new_classes)
                    if r is not None:
                        new_classes.append([m_add, r])
                        classes = new_classes
                        improved = True
                        break
        
        return classes
    
    def residue_reoptimize(classes, time_limit=2.0):
        """Try to reassign residues to free up capacity for more moduli."""
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            current_moduli = set(c[0] for c in classes)
            missing = [m for m in range(MOD_MIN, MOD_MAX + 1) if m not in current_moduli]
            missing.sort(key=lambda m: -1.0 / m)  # Prefer smaller moduli (higher weight)
            
            for m_add in missing:
                if time.time() - start > time_limit:
                    break
                # Try reassigning each existing class's residue
                for i in range(len(classes)):
                    m_i, r_i = classes[i]
                    # Try residues in a smart order: prefer those that are "different" from current
                    for r_new in range(m_i):
                        if r_new == r_i:
                            continue
                        # Check if new assignment is valid
                        if not check_residue_valid(m_i, r_new, classes, skip_idx=i):
                            continue
                        # Try to add m_add with the new configuration
                        new_classes = classes[:]
                        new_classes[i] = [m_i, r_new]
                        r_add = find_valid_residue(m_add, new_classes)
                        if r_add is not None:
                            new_classes.append([m_add, r_add])
                            classes = new_classes
                            improved = True
                            break
                    if improved:
                        break
        return classes
    
    def sa_polish(classes, initial_temp=1.0, cooling=0.999, min_temp=0.005):
        """Simulated annealing with temperature schedule and fast validity checks."""
        current = classes[:]
        current_score = calc_score(current)
        best = classes[:]
        best_score = current_score
        temperature = initial_temp
        
        n_iter = 0
        while temperature > min_temp and _time_left() > 15.0:
            n_iter += 1
            move_type = random.random()
            delta = 0.0
            new_classes = None
            
            if move_type < 0.30:
                # Single residue reassignment
                if len(current) == 0:
                    temperature *= cooling
                    continue
                i = random.randint(0, len(current) - 1)
                m, r = current[i]
                r_new = random.randint(0, m - 1)
                if r_new == r:
                    temperature *= cooling
                    continue
                if not check_residue_valid(m, r_new, current, skip_idx=i):
                    temperature *= cooling
                    continue
                new_classes = current[:]
                new_classes[i] = [m, r_new]
                delta = 0.0
            elif move_type < 0.55:
                # Add a missing modulus
                current_moduli = set(c[0] for c in current)
                missing = [m for m in range(MOD_MIN, MOD_MAX + 1) if m not in current_moduli]
                if not missing:
                    temperature *= cooling
                    continue
                m_add = random.choice(missing)
                r = find_valid_residue(m_add, current)
                if r is None:
                    temperature *= cooling
                    continue
                new_classes = current + [[m_add, r]]
                delta = 1.0 / m_add
            elif move_type < 0.70:
                # Remove a modulus
                if len(current) <= 2:
                    temperature *= cooling
                    continue
                i = random.randint(0, len(current) - 1)
                m_remove = current[i][0]
                new_classes = [c for c in current if c[0] != m_remove]
                delta = -1.0 / m_remove
            elif move_type < 0.85:
                # 2-swap: swap one modulus for another
                current_moduli = set(c[0] for c in current)
                missing = [m for m in range(MOD_MIN, MOD_MAX + 1) if m not in current_moduli]
                if not missing or len(current) == 0:
                    temperature *= cooling
                    continue
                m_add = random.choice(missing)
                m_remove = random.choice([c[0] for c in current])
                new_classes = [c for c in current if c[0] != m_remove]
                r = find_valid_residue(m_add, new_classes)
                if r is None:
                    temperature *= cooling
                    continue
                new_classes.append([m_add, r])
                delta = 1.0 / m_add - 1.0 / m_remove
            else:
                # Multi-residue perturbation: reassign 2 random residues
                if len(current) < 2:
                    temperature *= cooling
                    continue
                i1, i2 = random.sample(range(len(current)), 2)
                m1, r1 = current[i1]
                m2, r2 = current[i2]
                r1_new = random.randint(0, m1 - 1)
                r2_new = random.randint(0, m2 - 1)
                # Check both new residues are valid against all others
                valid = True
                for j in range(len(current)):
                    if j == i1 or j == i2:
                        continue
                    mj, rj = current[j]
                    if not check_residue_valid(m1, r1_new, [current[k] for k in range(len(current)) if k != i1], skip_idx=-1):
                        # Simpler: check against all except i1
                        pass
                    # Direct check
                    g1 = gcd_arr[m1 - MOD_MIN][mj - MOD_MIN]
                    if (r1_new % g1) == (rj % g1):
                        valid = False
                        break
                    g2 = gcd_arr[m2 - MOD_MIN][mj - MOD_MIN]
                    if (r2_new % g2) == (rj % g2):
                        valid = False
                        break
                if valid:
                    # Check i1 vs i2
                    g12 = gcd_arr[m1 - MOD_MIN][m2 - MOD_MIN]
                    if (r1_new % g12) == (r2_new % g12):
                        valid = False
                if not valid:
                    temperature *= cooling
                    continue
                new_classes = current[:]
                new_classes[i1] = [m1, r1_new]
                new_classes[i2] = [m2, r2_new]
                delta = 0.0
            
            if new_classes is None:
                temperature *= cooling
                continue
            
            new_score = current_score + delta
            
            # Acceptance criterion
            if new_score > current_score:
                current = new_classes
                current_score = new_score
                if current_score > best_score:
                    best = current[:]
                    best_score = current_score
            else:
                # Accept with probability exp(delta / temperature)
                if delta < 0:
                    accept_prob = math.exp(delta / temperature)
                else:
                    accept_prob = 1.0
                if random.random() < accept_prob:
                    current = new_classes
                    current_score = new_score
                    if current_score > best_score:
                        best = current[:]
                        best_score = current_score
            
            # Cool down
            temperature *= cooling
        
        return best, best_score
    
    # === Main search ===
    
    # Initial greedy construction - try smallest moduli first (highest 1/m)
    classes = []
    for m in range(MOD_MIN, MOD_MAX + 1):
        if _time_left() < 10.0:
            break
        new_classes = try_add(classes, m)
        if new_classes is not None:
            classes = new_classes
    
    best_classes = classes[:]
    best_score = calc_score(classes)
    
    # Residue reoptimization to free capacity
    if _time_left() > 20.0:
        reopt = residue_reoptimize(best_classes, time_limit=2.0)
        reopt_score = calc_score(reopt)
        if reopt_score > best_score:
            best_classes = reopt
            best_score = reopt_score
    
    # Hill climb
    if _time_left() > 15.0:
        polished = hill_climb(best_classes, time_limit=2.0)
        pol_score = calc_score(polished)
        if pol_score > best_score:
            best_classes = polished
            best_score = pol_score
    
    # Run multiple SA rounds with different parameters
    sa_configs = [
        (1.0, 0.999),
        (0.5, 0.998),
        (0.3, 0.997),
        (0.8, 0.9985),
    ]
    
    for temp, cooling in sa_configs:
        if _time_left() < 15.0:
            break
        
        # SA from current best
        sa_result, sa_score = sa_polish(best_classes, initial_temp=temp, cooling=cooling)
        if sa_score > best_score:
            best_classes = sa_result
            best_score = sa_score
        
        # Residue reoptimization after SA
        if _time_left() > 15.0:
            reopt = residue_reoptimize(best_classes, time_limit=1.5)
            reopt_score = calc_score(reopt)
            if reopt_score > best_score:
                best_classes = reopt
                best_score = reopt_score
        
        # Hill climb polish
        if _time_left() > 10.0:
            polished = hill_climb(best_classes, time_limit=2.0)
            pol_score = calc_score(polished)
            if pol_score > best_score:
                best_classes = polished
                best_score = pol_score
    
    # Final hill climb
    if _time_left() > 5.0:
        polished = hill_climb(best_classes, time_limit=3.0)
        pol_score = calc_score(polished)
        if pol_score > best_score:
            best_classes = polished
            best_score = pol_score
    
    # Final residue reoptimization
    if _time_left() > 3.0:
        reopt = residue_reoptimize(best_classes, time_limit=1.0)
        reopt_score = calc_score(reopt)
        if reopt_score > best_score:
            best_classes = reopt
            best_score = reopt_score
    
    return best_classes
# EVOLVE-BLOCK-END
