import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill
    
    def is_compatible(m_new, r_new, selected):
        """Check if (m_new, r_new) is disjoint from all in selected."""
        for m_i, r_i in selected:
            g = math.gcd(m_new, m_i)
            if r_new % g == r_i % g:
                return False
        return True
    
    def score_float(selected):
        """Compute score as float."""
        return sum(1.0 / m for m, r in selected)
    
    def score_fraction(selected):
        """Compute exact rational score."""
        total = Fraction(0, 1)
        for m, r in selected:
            total += Fraction(1, m)
        return total
    
    def count_available_residues(m, selected):
        """Count how many residues of m are compatible with selected."""
        count = 0
        for r in range(m):
            if is_compatible(m, r, selected):
                count += 1
        return count
    
    def best_residue(m, selected):
        """Find the best residue for m given selected classes.
        Heuristic: pick residue that minimizes 'tightness' - i.e., maximizes
        the minimum gcd-distance to existing residues."""
        valid_residues = []
        for r in range(m):
            if is_compatible(m, r, selected):
                valid_residues.append(r)
        
        if not valid_residues:
            return None
        
        if len(valid_residues) == 1:
            return valid_residues[0]
        
        # Score each valid residue by how "spread out" it is
        # For each residue r, compute min over selected of (r - r_i) mod gcd(m, m_i)
        # Pick the one with the largest minimum distance
        best_r = valid_residues[0]
        best_score_val = -1
        
        for r in valid_residues:
            min_dist = m  # start with max possible
            for m_i, r_i in selected:
                g = math.gcd(m, m_i)
                if g == 1:
                    continue
                # Distance in the cyclic group Z_g
                diff = (r - r_i) % g
                dist = min(diff, g - diff)
                min_dist = min(min_dist, dist)
            
            # We want to maximize the minimum distance
            if min_dist > best_score_val:
                best_score_val = min_dist
                best_r = r
        
        return best_r
    
    def greedy_construct(order, use_best_residue=True):
        """Greedy construction following given modulus order."""
        selected = []
        for m in order:
            if use_best_residue:
                r = best_residue(m, selected)
            else:
                r = None
                for ri in range(m):
                    if is_compatible(m, ri, selected):
                        r = ri
                        break
            
            if r is not None:
                selected.append([m, r])
        
        return selected
    
    def try_add_to_set(selected, m_new):
        """Try to add modulus m_new to selected. Returns new set or None."""
        r = best_residue(m_new, selected)
        if r is not None:
            return selected + [[m_new, r]]
        return None
    
    # Precompute divisor counts
    def divisor_count(m):
        count = 0
        for d in range(1, int(math.isqrt(m)) + 1):
            if m % d == 0:
                count += 1
                if d != m // d:
                    count += 1
        return count
    
    div_counts = {m: divisor_count(m) for m in range(20, 201)}
    
    best_score = 0.0
    best_selected = []
    
    # Phase 1: Multiple greedy orderings
    orderings = [
        list(range(20, 201)),  # natural
        sorted(range(20, 201), key=lambda m: div_counts[m]),  # fewest divisors first
        sorted(range(20, 201), key=lambda m: -div_counts[m]),  # most divisors first
        sorted(range(20, 201), key=lambda m: (div_counts[m], m)),  # divisors then natural
        sorted(range(20, 201), reverse=True),  # large to small
        sorted(range(20, 201), key=lambda m: (m % 2, m)),  # evens first
        sorted(range(20, 201), key=lambda m: (m % 3, m)),  # multiples of 3 first
    ]
    
    for order in orderings:
        selected = greedy_construct(order, use_best_residue=True)
        s = score_float(selected)
        if s > best_score:
            best_score = s
            best_selected = [x[:] for x in selected]
        
        if time.time() - start_time > DEADLINE - 30:
            break
    
    # Phase 2: Randomized orderings
    random.seed(42)
    for _ in range(50):
        if time.time() - start_time > DEADLINE - 20:
            break
        order = list(range(20, 201))
        random.shuffle(order)
        selected = greedy_construct(order, use_best_residue=True)
        s = score_float(selected)
        if s > best_score:
            best_score = s
            best_selected = [x[:] for x in selected]
    
    # Phase 3: Local search - try to improve the best solution
    improved = True
    while improved and time.time() - start_time < DEADLINE - 10:
        improved = False
        current = [x[:] for x in best_selected]
        current_score = score_float(current)
        
        # Try adding any missing modulus
        current_mods = set(x[0] for x in current)
        for m_try in range(20, 201):
            if m_try in current_mods:
                continue
            new_set = try_add_to_set(current, m_try)
            if new_set is not None:
                new_score = score_float(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    current_mods.add(m_try)
                    improved = True
        
        # Try swapping: remove one modulus, add another
        for i, (m_i, r_i) in enumerate(current):
            remaining = [x[:] for x in current if x[0] != m_i]
            rem_score = score_float(remaining)
            rem_mods = set(x[0] for x in remaining)
            
            for m_try in range(20, 201):
                if m_try in rem_mods:
                    continue
                new_set = try_add_to_set(remaining, m_try)
                if new_set is not None:
                    new_score = score_float(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break  # restart outer loop
        
        if current_score > best_score:
            best_score = current_score
            best_selected = [x[:] for x in current]
    
    # Phase 4: Final polish - try all possible single-residue changes
    for i, (m_i, r_i) in enumerate(best_selected):
        if time.time() - start_time > DEADLINE - 5:
            break
        other = [x[:] for x in best_selected if x[0] != m_i]
        # Try all valid residues for m_i
        valid_rs = []
        for r in range(m_i):
            if is_compatible(m_i, r, other):
                valid_rs.append(r)
        
        # For each valid residue, check if it enables adding new moduli
        for r_new in valid_rs:
            if r_new == r_i:
                continue
            test_set = other + [[m_i, r_new]]
            test_mods = set(x[0] for x in test_set)
            
            # Try to add any missing modulus
            for m_try in range(20, 201):
                if m_try in test_mods:
                    continue
                new_set = try_add_to_set(test_set, m_try)
                if new_set is not None:
                    new_score = score_float(new_set)
                    if new_score > best_score:
                        best_score = new_score
                        best_selected = [x[:] for x in new_set]
                    break
    
    return best_selected
