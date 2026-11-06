"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd
from fractions import Fraction

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Strategy:
    1. Precompute conflict graph using bitsets for O(1) intersection checks
    2. Greedy initial construction
    3. Branch-and-bound on conflict graph with weight/capacity ratio heuristic
    4. Simulated annealing polish
    5. Final hill climbing
    """
    deadline = time.time() + 145.0
    random.seed(42)
    
    def _time_left():
        return deadline - time.time()
    
    # Precompute all possible (m, r) pairs and their conflicts
    all_moduli = list(range(20, 201))
    # Map each (m, r) to an index
    pair_to_idx = {}
    idx_to_pair = []
    for m in all_moduli:
        for r in range(m):
            pair_to_idx[(m, r)] = len(idx_to_pair)
            idx_to_pair.append((m, r))
    
    n_pairs = len(idx_to_pair)
    weights = [Fraction(1, m) for m, r in idx_to_pair]
    
    # Precompute conflict bitsets: conflict[i] is a set of indices that conflict with i
    # Two pairs (m1,r1) and (m2,r2) conflict if r1 % gcd(m1,m2) == r2 % gcd(m1,m2)
    # We'll use Python integers as bitsets
    conflicts = [0] * n_pairs
    for i in range(n_pairs):
        m1, r1 = idx_to_pair[i]
        for j in range(i+1, n_pairs):
            m2, r2 = idx_to_pair[j]
            g = gcd(m1, m2)
            if (r1 % g) == (r2 % g):
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    # Also precompute: for each modulus m, which indices have that modulus
    moduli_indices = {}
    for idx, (m, r) in enumerate(idx_to_pair):
        if m not in moduli_indices:
            moduli_indices[m] = []
        moduli_indices[m].append(idx)
    
    def is_valid_pair(m1, r1, m2, r2):
        g = gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    def calc_score(classes):
        return sum(1.0 / m for m, r in classes)
    
    def find_valid_residue(m, classes):
        for r in range(m):
            valid = True
            for m2, r2 in classes:
                g = gcd(m, m2)
                if (r % g) == (r2 % g):
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def try_add(classes, m):
        r = find_valid_residue(m, classes)
        if r is not None:
            return classes + [[m, r]]
        return None
    
    def branch_and_bound(initial_selection, time_limit=60.0):
        """
        Branch-and-bound on the conflict graph.
        initial_selection: set of indices already selected
        Returns best selection as list of (m, r) pairs.
        """
        start_time = time.time()
        best_selection = set(initial_selection)
        best_score = sum(weights[i] for i in best_selection)
        
        # Precompute: for each index, its weight
        # Available: indices not conflicting with any selected
        def get_available(selected_set):
            """Get bitmask of available indices (not conflicting with selected)."""
            # Start with all indices
            available = (1 << n_pairs) - 1
            for idx in selected_set:
                available &= ~conflicts[idx]
            available &= ~(1 << 0)  # placeholder, will fix
            # Remove selected indices themselves
            for idx in selected_set:
                available &= ~(1 << idx)
            return available
        
        # Better approach: maintain available as a set
        def get_available_set(selected_set):
            """Get set of available indices."""
            blocked = set()
            for idx in selected_set:
                # Get all conflicts of idx
                c = conflicts[idx]
                while c:
                    low_bit = c & (-c)
                    j = low_bit.bit_length() - 1
                    blocked.add(j)
                    c ^= low_bit
            available = set(range(n_pairs)) - blocked - selected_set
            return available
        
        # For efficiency, use a different approach:
        # Track which moduli are already used
        def get_available_by_moduli(used_moduli, selected_set):
            """Get available indices given used moduli and selected set."""
            available = []
            for idx in range(n_pairs):
                m, r = idx_to_pair[idx]
                if m in used_moduli:
                    continue
                if idx in selected_set:
                    continue
                # Check if conflicts with any selected
                conflict = False
                c = conflicts[idx]
                for s in selected_set:
                    if (c >> s) & 1:
                        conflict = True
                        break
                if not conflict:
                    available.append(idx)
            return available
        
        # Branch and bound
        def bb(selected_set, used_moduli, available_list, current_score):
            nonlocal best_selection, best_score
            
            if time.time() - start_time > time_limit:
                return
            
            # Update best
            if current_score > best_score:
                best_score = current_score
                best_selection = set(selected_set)
            
            if not available_list:
                return
            
            # Upper bound: current_score + sum of weights of available
            # (optimistic: assume we can pick all available)
            upper_bound = current_score + sum(weights[i] for i in available_list)
            if upper_bound <= best_score:
                return
            
            # Sort available by weight descending (greedy heuristic)
            # Actually, sort by weight / (1 + number of conflicts with other available)
            # For simplicity, sort by weight descending
            available_sorted = sorted(available_list, key=lambda i: weights[i], reverse=True)
            
            # Limit branching factor for speed
            max_branch = min(len(available_sorted), 50)
            
            for idx in available_sorted[:max_branch]:
                if time.time() - start_time > time_limit:
                    return
                
                m, r = idx_to_pair[idx]
                new_selected = selected_set | {idx}
                new_used = used_moduli | {m}
                
                # Get new available
                new_available = []
                for j in available_list:
                    if j == idx:
                        continue
                    # Check if j conflicts with idx
                    c = conflicts[idx]
                    if (c >> j) & 1:
                        continue
                    # Check if j's modulus is already used
                    mj, rj = idx_to_pair[j]
                    if mj in new_used:
                        continue
                    new_available.append(j)
                
                new_score = current_score + weights[idx]
                bb(new_selected, new_used, new_available, new_score)
        
        # Initial available
        initial_available = get_available_by_moduli(set(m for m, r in idx_to_pair if False), set(initial_selection))
        # Actually, let's recompute properly
        used_moduli = set()
        for idx in initial_selection:
            m, r = idx_to_pair[idx]
            used_moduli.add(m)
        
        initial_available = []
        for idx in range(n_pairs):
            m, r = idx_to_pair[idx]
            if m in used_moduli:
                continue
            if idx in initial_selection:
                continue
            conflict = False
            c = conflicts[idx]
            for s in initial_selection:
                if (c >> s) & 1:
                    conflict = True
                    break
            if not conflict:
                initial_available.append(idx)
        
        bb(set(initial_selection), used_moduli, initial_available, sum(weights[i] for i in initial_selection))
        
        # Convert back to (m, r) pairs
        return [(idx_to_pair[i][0], idx_to_pair[i][1]) for i in best_selection]
    
    def hill_climb(classes, time_limit=3.0):
        """Hill climbing polish."""
        improved = True
        start = time.time()
        while improved and time.time() - start < time_limit:
            improved = False
            current_moduli = set(c[0] for c in classes)
            missing = [m for m in range(20, 201) if m not in current_moduli]
            
            # Try adding
            for m in missing:
                if time.time() - start > time_limit:
                    break
                new_classes = try_add(classes, m)
                if new_classes is not None:
                    classes = new_classes
                    improved = True
            
            # Try swapping
            current_moduli = set(c[0] for c in classes)
            missing = [m for m in range(20, 201) if m not in current_moduli]
            for m_add in missing:
                if time.time() - start > time_limit:
                    break
                for m_remove in current_moduli:
                    if 1.0/m_add <= 1.0/m_remove:
                        continue
                    new_classes = [c for c in classes if c[0] != m_remove]
                    r = find_valid_residue(m_add, new_classes)
                    if r is not None:
                        new_classes.append([m_add, r])
                        classes = new_classes
                        improved = True
                        break
        return classes
    
    def sa_polish(classes, initial_temp=1.0, cooling=0.999, min_temp=0.01):
        """Proper simulated annealing with temperature schedule."""
        current = classes[:]
        current_score = calc_score(current)
        best = classes[:]
        best_score = current_score
        temperature = initial_temp
        
        while temperature > min_temp and _time_left() > 15.0:
            move_type = random.random()
            delta = 0.0
            new_classes = None
            
            if move_type < 0.25:
                if len(current) == 0:
                    temperature *= cooling
                    continue
                i = random.randint(0, len(current) - 1)
                m, r = current[i]
                r_new = random.randint(0, m - 1)
                if r_new == r:
                    temperature *= cooling
                    continue
                new_classes = current[:]
                new_classes[i] = [m, r_new]
                valid = True
                for j in range(len(new_classes)):
                    if j == i:
                        continue
                    if not is_valid_pair(m, r_new, new_classes[j][0], new_classes[j][1]):
                        valid = False
                        break
                if not valid:
                    temperature *= cooling
                    continue
                delta = 0.0
            elif move_type < 0.5:
                current_moduli = set(c[0] for c in current)
                missing = [m for m in range(20, 201) if m not in current_moduli]
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
            elif move_type < 0.7:
                if len(current) <= 2:
                    temperature *= cooling
                    continue
                i = random.randint(0, len(current) - 1)
                m_remove = current[i][0]
                new_classes = [c for c in current if c[0] != m_remove]
                delta = -1.0 / m_remove
            else:
                current_moduli = set(c[0] for c in current)
                missing = [m for m in range(20, 201) if m not in current_moduli]
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
                delta = 1.0/m_add - 1.0/m_remove
            
            if new_classes is None:
                temperature *= cooling
                continue
            
            new_score = current_score + delta
            
            if new_score > current_score:
                current = new_classes
                current_score = new_score
                if current_score > best_score:
                    best = current[:]
                    best_score = current_score
            else:
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
            
            temperature *= cooling
        
        return best, best_score
    
    # Initial greedy construction
    classes = []
    for m in range(20, 201):
        if _time_left() < 30.0:
            break
        new_classes = try_add(classes, m)
        if new_classes is not None:
            classes = new_classes
    
    best_classes = classes[:]
    best_score = calc_score(classes)
    
    # Convert to indices for B&B
    best_indices = set()
    for m, r in best_classes:
        if (m, r) in pair_to_idx:
            best_indices.add(pair_to_idx[(m, r)])
    
    # Run branch-and-bound
    if _time_left() > 40.0:
        bb_result = branch_and_bound(best_indices, time_limit=min(50.0, _time_left() - 20.0))
        bb_score = calc_score(bb_result)
        if bb_score > best_score:
            best_classes = bb_result
            best_score = bb_score
    
    # SA polish
    if _time_left() > 20.0:
        sa_result, sa_score = sa_polish(best_classes, initial_temp=0.5, cooling=0.998)
        if sa_score > best_score:
            best_classes = sa_result
            best_score = sa_score
    
    # Final hill climb
    if _time_left() > 5.0:
        polished = hill_climb(best_classes, time_limit=min(3.0, _time_left() - 3.0))
        pol_score = calc_score(polished)
        if pol_score > best_score:
            best_classes = polished
            best_score = pol_score
    
    return best_classes
# EVOLVE-BLOCK-END
