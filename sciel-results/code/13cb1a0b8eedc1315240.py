"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
from itertools import combinations
import random

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Key insight: If gcd(m1, m2) == 1, the classes can NEVER be disjoint.
    So we only consider moduli that share factors with others in the set.
    
    Strategy:
    1. Precompute gcd relationships between all moduli in [20, 200].
    2. Multiple construction strategies with local search:
       a. Greedy by ascending m (smaller = higher weight)
       b. Group by common prime factor, pick best multiples
       c. Local search with add, swap, and remove-readd moves
    3. Return best found within time budget.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    all_moduli = list(range(20, 201))
    
    # Precompute gcd table for all pairs
    gcd_table = {}
    for i, m1 in enumerate(all_moduli):
        for j, m2 in enumerate(all_moduli):
            if j > i:
                gcd_table[(m1, m2)] = gcd(m1, m2)
                gcd_table[(m2, m1)] = gcd(m1, m2)
    
    def find_valid_residue(m, current_classes):
        """Find a residue r in [0, m) such that (m, r) is disjoint from all current classes."""
        for r in range(m):
            ok = True
            for cm, cr in current_classes:
                g = gcd(m, cm)
                if g == 1:
                    return None  # Can never be disjoint
                if r % g == cr % g:
                    ok = False
                    break
            if ok:
                return r
        return None
    
    def find_valid_residue_fast(m, current_classes):
        """Optimized residue finding using precomputed gcd table."""
        # Build constraints: for each existing class, r mod g != cr mod g
        constraints = []
        for cm, cr in current_classes:
            g = gcd(m, cm)
            if g == 1:
                return None
            forbidden = cr % g
            constraints.append((g, forbidden))
        
        # Try residues in order of least constraint
        for r in range(m):
            ok = True
            for g, forbidden in constraints:
                if r % g == forbidden:
                    ok = False
                    break
            if ok:
                return r
        return None
    
    def calc_score(cls):
        """Calculate sum of 1/m for the given classes."""
        total = Fraction(0, 1)
        for m, r in cls:
            total += Fraction(1, m)
        return total
    
    def calc_score_float(cls):
        """Calculate sum of 1/m as float for quick comparisons."""
        return sum(1.0 / m for m, r in cls)
    
    def is_pairwise_valid(cls):
        """Check if all pairs are disjoint."""
        n = len(cls)
        for i in range(n):
            for j in range(i+1, n):
                m1, r1 = cls[i]
                m2, r2 = cls[j]
                g = gcd(m1, m2)
                if g == 1:
                    return False
                if r1 % g == r2 % g:
                    return False
        return True
    
    def local_search(initial_classes, time_limit):
        """Improve a set of classes using local search moves with better heuristics."""
        current_classes = [c[:] for c in initial_classes]
        current_score = calc_score(current_classes)
        best_score = current_score
        best_classes = [c[:] for c in current_classes]
        
        if not is_pairwise_valid(current_classes):
            return best_classes, best_score
        
        in_set = set(c[0] for c in current_classes)
        
        # Precompute weight for quick comparison
        weights = {m: Fraction(1, m) for m in all_moduli}
        
        while time.time() < time_limit:
            improved = False
            
            # Move 1: Try adding a missing modulus (try all, not just sorted)
            add_candidates = [m for m in all_moduli if m not in in_set]
            # Sort by weight descending
            add_candidates.sort(key=lambda x: -float(weights[x]))
            
            for m in add_candidates:
                r = find_valid_residue_fast(m, current_classes)
                if r is not None:
                    new_classes = current_classes + [[m, r]]
                    new_score = current_score + weights[m]
                    if new_score > current_score:
                        current_classes = new_classes
                        current_score = new_score
                        in_set.add(m)
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_classes = [c[:] for c in current_classes]
            
            # Move 2: Try swapping - replace a low-weight modulus with a higher-weight one
            # Sort current classes by weight ascending (least valuable first)
            sorted_indices = sorted(range(len(current_classes)), 
                                   key=lambda i: float(weights[current_classes[i][0]]))
            
            for i in sorted_indices:
                m_old = current_classes[i][0]
                # Try replacing with higher weight moduli
                for m_new in sorted(all_moduli, key=lambda x: -float(weights[x])):
                    if m_new in in_set or m_new == m_old:
                        continue
                    if float(weights[m_new]) <= float(weights[m_old]):
                        continue  # Only try if clearly better weight
                    
                    candidate = [c[:] for c in current_classes]
                    candidate.pop(i)
                    r_new = find_valid_residue_fast(m_new, candidate)
                    if r_new is not None:
                        candidate.append([m_new, r_new])
                        new_score = current_score - weights[m_old] + weights[m_new]
                        if new_score > current_score:
                            current_classes = candidate
                            current_score = new_score
                            in_set.remove(m_old)
                            in_set.add(m_new)
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_classes = [c[:] for c in current_classes]
                            break  # Restart swap loop with new state
                if improved:
                    break
            
            # Move 3: Try 2-opt style - remove two, add two different ones
            if not improved and time.time() < time_limit:
                n = len(current_classes)
                for i in range(n):
                    for j in range(i+1, n):
                        m_remove1 = current_classes[i][0]
                        m_remove2 = current_classes[j][0]
                        
                        candidate = [c[:] for c in current_classes]
                        candidate.pop(j)
                        candidate.pop(i)
                        
                        # Try adding two new moduli
                        add_options = [m for m in all_moduli 
                                      if m not in set(c[0] for c in candidate)]
                        add_options.sort(key=lambda x: -float(weights[x]))
                        
                        found = False
                        for m_add1 in add_options[:10]:  # Limit to top 10 for speed
                            r_add1 = find_valid_residue_fast(m_add1, candidate)
                            if r_add1 is None:
                                continue
                            temp_candidate = candidate + [[m_add1, r_add1]]
                            
                            for m_add2 in add_options[:10]:
                                if m_add2 == m_add1:
                                    continue
                                r_add2 = find_valid_residue_fast(m_add2, temp_candidate)
                                if r_add2 is None:
                                    continue
                                
                                new_candidate = temp_candidate + [[m_add2, r_add2]]
                                new_score = (current_score 
                                            - weights[m_remove1] 
                                            - weights[m_remove2]
                                            + weights[m_add1] 
                                            + weights[m_add2])
                                if new_score > current_score:
                                    current_classes = new_candidate
                                    current_score = new_score
                                    in_set = set(c[0] for c in current_classes)
                                    improved = True
                                    found = True
                                    if current_score > best_score:
                                        best_score = current_score
                                        best_classes = [c[:] for c in current_classes]
                                    break
                            if found:
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_classes, best_score
    
    def greedy_ascending():
        """Greedy: add moduli in ascending order (smallest first = highest weight)."""
        classes = []
        in_set = set()
        for m in all_moduli:  # Already ascending
            r = find_valid_residue_fast(m, classes)
            if r is not None:
                classes.append([m, r])
                in_set.add(m)
        return classes
    
    def greedy_by_prime_groups():
        """Group moduli by their smallest prime factor, pick best from each group."""
        from math import isqrt
        
        def smallest_prime_factor(n):
            if n % 2 == 0:
                return 2
            for i in range(3, isqrt(n) + 1, 2):
                if n % i == 0:
                    return i
            return n
        
        # Group moduli by smallest prime factor
        groups = {}
        for m in all_moduli:
            spf = smallest_prime_factor(m)
            if spf not in groups:
                groups[spf] = []
            groups[spf].append(m)
        
        # Try different strategies for combining groups
        best_classes = []
        best_score = Fraction(0, 1)
        
        # Strategy: pick from each prime group the smallest moduli
        for strategy in range(3):
            classes = []
            in_set = set()
            
            if strategy == 0:
                # Pick smallest from each group
                for spf in sorted(groups.keys()):
                    for m in sorted(groups[spf]):
                        if m in in_set:
                            continue
                        r = find_valid_residue_fast(m, classes)
                        if r is not None:
                            classes.append([m, r])
                            in_set.add(m)
                            break  # Only one per group for now
            elif strategy == 1:
                # Pick up to 2 from each group
                for spf in sorted(groups.keys()):
                    count = 0
                    for m in sorted(groups[spf]):
                        if m in in_set:
                            continue
                        r = find_valid_residue_fast(m, classes)
                        if r is not None:
                            classes.append([m, r])
                            in_set.add(m)
                            count += 1
                            if count >= 2:
                                break
            else:
                # Greedy overall, but prefer moduli with more shared factors
                def factor_count(m):
                    cnt = 0
                    for other in all_moduli:
                        if other != m and gcd(m, other) > 1:
                            cnt += 1
                    return cnt
                
                for m in sorted(all_moduli, key=lambda x: (-factor_count(x), x)):
                    if m in in_set:
                        continue
                    r = find_valid_residue_fast(m, classes)
                    if r is not None:
                        classes.append([m, r])
                        in_set.add(m)
            
            if classes:
                score = calc_score(classes)
                if score > best_score:
                    best_score = score
                    best_classes = [c[:] for c in classes]
        
        return best_classes
    
    # Run multiple strategies
    best_overall = []
    best_overall_score = Fraction(0, 1)
    
    strategies = [
        ("greedy_ascending", greedy_ascending),
        ("greedy_by_prime_groups", greedy_by_prime_groups),
    ]
    
    # Add a randomized greedy strategy
    random.seed(42)
    
    def greedy_randomized():
        """Greedy with randomized order, multiple restarts."""
        best = []
        best_score = Fraction(0, 1)
        
        for restart in range(5):
            classes = []
            in_set = set()
            order = all_moduli[:]
            random.shuffle(order)
            # But bias toward smaller moduli
            order.sort(key=lambda m: (random.random() * 0.3 + 1.0/m, m))
            
            for m in order:
                if m in in_set:
                    continue
                r = find_valid_residue_fast(m, classes)
                if r is not None:
                    classes.append([m, r])
                    in_set.add(m)
            
            if classes:
                score = calc_score(classes)
                if score > best_score:
                    best_score = score
                    best = [c[:] for c in classes]
        
        return best
    
    strategies.append(("greedy_randomized", greedy_randomized))
    
    for name, func in strategies:
        if time.time() > deadline - 30:
            break
        try:
            classes = func()
            if classes and is_pairwise_valid(classes):
                score = calc_score(classes)
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall = [c[:] for c in classes]
        except Exception:
            pass
    
    # Local search on the best initial solution
    if best_overall and time.time() < deadline - 10:
        improved_classes, improved_score = local_search(
            best_overall, 
            time.time() + (deadline - time.time() - 5)
        )
        if improved_score > best_overall_score:
            best_overall = improved_classes
            best_overall_score = improved_score
    
    # Final polish pass: try to add any remaining moduli with best residues
    if best_overall and time.time() < deadline - 2:
        in_set = set(c[0] for c in best_overall)
        for m in sorted(all_moduli, key=lambda x: -1.0/x):
            if m in in_set:
                continue
            r = find_valid_residue_fast(m, best_overall)
            if r is not None:
                best_overall.append([m, r])
                in_set.add(m)
    
    # Final validation
    if best_overall:
        if not is_pairwise_valid(best_overall):
            # Fallback to simple valid solution
            best_overall = greedy_ascending()
            if not is_pairwise_valid(best_overall):
                best_overall = []
    
    return [c[:] for c in best_overall]

# EVOLVE-BLOCK-END
