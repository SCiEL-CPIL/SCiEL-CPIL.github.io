"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from math import gcd
from fractions import Fraction

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Key constraint: For two classes (m1, r1) and (m2, r2) to be disjoint,
    we need r1 != r2 mod gcd(m1, m2). If gcd(m1, m2) == 1, this is impossible
    (since any two residues are congruent mod 1). So we can only include moduli
    that share factors with all other moduli in the set.
    
    Strategy:
    1. Identify compatible moduli (those that can coexist with many others).
    2. Greedy construction with multiple orderings.
    3. Local search with add, swap, 2-swap, and residue reassignment moves.
    4. Simulated annealing for escaping local optima.
    5. Residue optimization to maximize addition capacity.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    all_moduli = list(range(20, 201))
    
    # Precompute gcd table
    gcd_table = {}
    for i, m1 in enumerate(all_moduli):
        for j, m2 in enumerate(all_moduli):
            if j > i:
                g = gcd(m1, m2)
                gcd_table[(m1, m2)] = g
                gcd_table[(m2, m1)] = g
    
    # Precompute compatibility: which pairs of moduli CAN be disjoint
    # (i.e., gcd > 1)
    compatible_pairs = {}
    for m1 in all_moduli:
        compatible_pairs[m1] = set()
    for m1 in all_moduli:
        for m2 in all_moduli:
            if m1 != m2 and gcd_table.get((m1, m2), gcd(m1, m2)) > 1:
                compatible_pairs[m1].add(m2)
    
    def find_valid_residue(m, current_classes):
        """Find a residue r in [0, m) such that (m, r) is disjoint from all current classes."""
        if not current_classes:
            return 0
        
        for cm, cr in current_classes:
            g = gcd_table.get((m, cm), gcd(m, cm))
            if g == 1:
                return None
        
        forbidden = [False] * m
        for cm, cr in current_classes:
            g = gcd_table.get((m, cm), gcd(m, cm))
            forbidden_residue = cr % g
            for r in range(forbidden_residue, m, g):
                forbidden[r] = True
        
        for r in range(m):
            if not forbidden[r]:
                return r
        return None
    
    def find_all_valid_residues(m, current_classes):
        """Find ALL valid residues for m given current_classes."""
        if not current_classes:
            return list(range(m))
        
        for cm, cr in current_classes:
            g = gcd_table.get((m, cm), gcd(m, cm))
            if g == 1:
                return []
        
        forbidden = [False] * m
        for cm, cr in current_classes:
            g = gcd_table.get((m, cm), gcd(m, cm))
            forbidden_residue = cr % g
            for r in range(forbidden_residue, m, g):
                forbidden[r] = True
        
        return [r for r in range(m) if not forbidden[r]]
    
    def calc_score_float(cls):
        """Calculate sum of 1/m as float."""
        return sum(1.0 / m for m, r in cls)
    
    def calc_score_frac(cls):
        """Calculate sum of 1/m as Fraction for exact comparison."""
        total = Fraction(0, 1)
        for m, r in cls:
            total += Fraction(1, m)
        return total
    
    def is_pairwise_valid(cls):
        """Check if all pairs are disjoint."""
        n = len(cls)
        for i in range(n):
            for j in range(i+1, n):
                m1, r1 = cls[i]
                m2, r2 = cls[j]
                g = gcd_table.get((m1, m2), gcd(m1, m2))
                if g == 1:
                    return False
                if r1 % g == r2 % g:
                    return False
        return True
    
    def count_addable(cls, in_set):
        """Count how many additional moduli can be added to cls."""
        count = 0
        for m in all_moduli:
            if m in in_set:
                continue
            r = find_valid_residue(m, cls)
            if r is not None:
                count += 1
        return count
    
    def local_search(initial_classes, time_limit):
        """Improve a set of classes using local search moves."""
        current_classes = [c[:] for c in initial_classes]
        current_score = calc_score_float(current_classes)
        best_score = current_score
        best_classes = [c[:] for c in current_classes]
        
        if not is_pairwise_valid(current_classes):
            return best_classes, best_score
        
        in_set = set(c[0] for c in current_classes)
        moduli_by_weight_desc = sorted(all_moduli, key=lambda x: -1.0/x)
        
        while time.time() < time_limit:
            improved = False
            
            # Move 1: Try adding a missing modulus (smallest first)
            for m in moduli_by_weight_desc:
                if m in in_set:
                    continue
                r = find_valid_residue(m, current_classes)
                if r is not None:
                    current_classes.append([m, r])
                    current_score += 1.0 / m
                    in_set.add(m)
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
                    break
            
            # Move 2: Try 1-swap (replace a larger modulus with a smaller one)
            if not improved and time.time() < time_limit:
                sorted_indices = sorted(range(len(current_classes)),
                                       key=lambda i: 1.0/current_classes[i][0])
                
                for i in sorted_indices:
                    m_old = current_classes[i][0]
                    
                    for m_new in moduli_by_weight_desc:
                        if m_new in in_set or m_new == m_old:
                            continue
                        if 1.0/m_new <= 1.0/m_old:
                            break
                        
                        candidate = [c[:] for c in current_classes]
                        candidate.pop(i)
                        r_new = find_valid_residue(m_new, candidate)
                        if r_new is not None:
                            candidate.append([m_new, r_new])
                            new_score = current_score - 1.0/m_old + 1.0/m_new
                            if new_score > current_score:
                                current_classes = candidate
                                current_score = new_score
                                in_set.remove(m_old)
                                in_set.add(m_new)
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_classes = [c[:] for c in current_classes]
                                break
                    if improved:
                        break
            
            # Move 3: Try 2-swap (replace two moduli with two others)
            if not improved and time.time() < time_limit:
                n = len(current_classes)
                found = False
                for i in range(n):
                    if found or time.time() >= time_limit:
                        break
                    for j in range(i+1, n):
                        if found or time.time() >= time_limit:
                            break
                        m_remove1 = current_classes[i][0]
                        m_remove2 = current_classes[j][0]
                        
                        candidate = [c[:] for c in current_classes]
                        candidate.pop(j)
                        candidate.pop(i)
                        
                        cand_in_set = set(c[0] for c in candidate)
                        add_options = [m for m in moduli_by_weight_desc
                                      if m not in cand_in_set][:25]
                        
                        for m_add1 in add_options:
                            if found or time.time() >= time_limit:
                                break
                            r_add1 = find_valid_residue(m_add1, candidate)
                            if r_add1 is None:
                                continue
                            temp_candidate = candidate + [[m_add1, r_add1]]
                            temp_in_set = cand_in_set | {m_add1}
                            
                            for m_add2 in add_options:
                                if m_add2 == m_add1 or m_add2 in temp_in_set:
                                    continue
                                r_add2 = find_valid_residue(m_add2, temp_candidate)
                                if r_add2 is None:
                                    continue
                                
                                new_candidate = temp_candidate + [[m_add2, r_add2]]
                                new_score = (current_score
                                            - 1.0/m_remove1
                                            - 1.0/m_remove2
                                            + 1.0/m_add1
                                            + 1.0/m_add2)
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
                    if found:
                        break
            
            if not improved:
                break
        
        return best_classes, best_score
    
    def simulated_annealing(initial_classes, time_limit):
        """SA with temperature schedule to escape local optima."""
        current_classes = [c[:] for c in initial_classes]
        current_score = calc_score_float(current_classes)
        best_score = current_score
        best_classes = [c[:] for c in current_classes]
        
        in_set = set(c[0] for c in current_classes)
        moduli_by_weight_desc = sorted(all_moduli, key=lambda x: -1.0/x)
        
        T_start = 0.12
        T_end = 0.0005
        start_time = time.time()
        remaining = time_limit - start_time
        est_iters = int(remaining * 2000)  # ~500us per iter
        est_iters = max(est_iters, 500)
        cooling_rate = (T_end / T_start) ** (1.0 / est_iters)
        
        T = T_start
        
        while time.time() < time_limit - 0.5:
            T *= cooling_rate
            
            move_type = random.random()
            
            if move_type < 0.35:
                # Add a random missing modulus
                missing = [m for m in all_moduli if m not in in_set]
                if not missing:
                    continue
                # Bias toward smaller moduli
                if random.random() < 0.6:
                    m = min(missing)
                else:
                    m = random.choice(missing[:40]) if len(missing) > 40 else random.choice(missing)
                
                r = find_valid_residue(m, current_classes)
                if r is None:
                    continue
                
                old_score = current_score
                current_classes.append([m, r])
                new_score = old_score + 1.0/m
                in_set.add(m)
                
                delta = new_score - old_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
                else:
                    current_classes.pop()
                    in_set.discard(m)
                    current_score = old_score
                    
            elif move_type < 0.55:
                # Remove a random modulus
                if not current_classes:
                    continue
                idx = random.randint(0, len(current_classes) - 1)
                m_old = current_classes[idx][0]
                
                old_score = current_score
                current_classes.pop(idx)
                new_score = old_score - 1.0/m_old
                in_set.remove(m_old)
                
                delta = new_score - old_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
                else:
                    current_classes.insert(idx, [m_old, 0])  # placeholder, will fix
                    # Actually, we need to restore properly
                    # Let's just re-add with a valid residue
                    r = find_valid_residue(m_old, current_classes)
                    if r is not None:
                        current_classes.append([m_old, r])
                        current_classes.remove([m_old, 0])
                    else:
                        current_classes.append([m_old, 0])
                    in_set.add(m_old)
                    current_score = old_score
                    
            elif move_type < 0.85:
                # Swap: remove one, add another
                if not current_classes:
                    continue
                idx = random.randint(0, len(current_classes) - 1)
                m_old = current_classes[idx][0]
                
                missing = [m for m in all_moduli if m not in in_set]
                if not missing:
                    continue
                candidates = sorted(missing, key=lambda x: x)[:30]
                m_new = random.choice(candidates)
                
                candidate = [c[:] for c in current_classes]
                candidate.pop(idx)
                r_new = find_valid_residue(m_new, candidate)
                if r_new is None:
                    continue
                candidate.append([m_new, r_new])
                
                old_score = current_score
                new_score = old_score - 1.0/m_old + 1.0/m_new
                
                delta = new_score - old_score
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_classes = candidate
                    current_score = new_score
                    in_set.remove(m_old)
                    in_set.add(m_new)
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
                        
            else:
                # Residue reassignment
                if not current_classes:
                    continue
                idx = random.randint(0, len(current_classes) - 1)
                m = current_classes[idx][0]
                others = [c for i, c in enumerate(current_classes) if i != idx]
                
                valid_residues = find_all_valid_residues(m, others)
                if len(valid_residues) <= 1:
                    continue
                
                old_r = current_classes[idx][1]
                new_r = random.choice(valid_residues)
                if new_r == old_r:
                    continue
                
                # Accept with probability based on temperature
                if random.random() < 0.3 + 0.7 * T / T_start:
                    current_classes[idx] = [m, new_r]
        
        return best_classes, best_score
    
    def residue_reoptimization(classes, time_limit):
        """Try different residues for each modulus to maximize addition capacity."""
        current_classes = [c[:] for c in classes]
        best_classes = [c[:] for c in current_classes]
        best_score = calc_score_float(current_classes)
        
        for idx in range(len(current_classes)):
            if time.time() >= time_limit:
                break
            m = current_classes[idx][0]
            others = [c for i, c in enumerate(current_classes) if i != idx]
            
            valid_residues = find_all_valid_residues(m, others)
            if len(valid_residues) <= 1:
                continue
            
            best_r = current_classes[idx][1]
            best_add_count = -1
            
            # Try a subset of valid residues
            step = max(1, len(valid_residues) // 30)
            for r in valid_residues[::step][:30]:
                test_classes = [c[:] for c in current_classes]
                test_classes[idx] = [m, r]
                
                in_set = set(c[0] for c in test_classes)
                add_count = 0
                for m_add in all_moduli:
                    if m_add in in_set:
                        continue
                    r_add = find_valid_residue(m_add, test_classes)
                    if r_add is not None:
                        add_count += 1
                        if add_count > best_add_count + 5:
                            break
                
                if add_count > best_add_count:
                    best_add_count = add_count
                    best_r = r
            
            if best_r != current_classes[idx][1]:
                current_classes[idx] = [m, best_r]
        
        # After reoptimization, try to add more moduli
        in_set = set(c[0] for c in current_classes)
        for m in sorted(all_moduli):
            if m in in_set:
                continue
            r = find_valid_residue(m, current_classes)
            if r is not None:
                current_classes.append([m, r])
                in_set.add(m)
        
        score = calc_score_float(current_classes)
        if score > best_score:
            best_classes = [c[:] for c in current_classes]
            best_score = score
        
        return best_classes, best_score
    
    def greedy_ascending():
        """Greedy: add moduli in ascending order."""
        classes = []
        in_set = set()
        for m in all_moduli:
            r = find_valid_residue(m, classes)
            if r is not None:
                classes.append([m, r])
                in_set.add(m)
        return classes
    
    def greedy_by_compatibility():
        """Greedy: prioritize moduli with high compatibility (can coexist with many)."""
        def compat_score(m):
            return len(compatible_pairs[m])
        
        classes = []
        in_set = set()
        for m in sorted(all_moduli, key=lambda x: (-compat_score(x), x)):
            r = find_valid_residue(m, classes)
            if r is not None:
                classes.append([m, r])
                in_set.add(m)
        return classes
    
    def greedy_by_factor_richness():
        """Greedy: prioritize moduli with many small prime factors."""
        def factor_richness(m):
            count = 0
            for p in [2, 3, 5, 7, 11, 13]:
                if m % p == 0:
                    count += 1
            return (-count, m)
        
        classes = []
        in_set = set()
        for m in sorted(all_moduli, key=factor_richness):
            r = find_valid_residue(m, classes)
            if r is not None:
                classes.append([m, r])
                in_set.add(m)
        return classes
    
    def greedy_randomized(num_restarts=20):
        """Greedy with randomized order."""
        best = []
        best_score = 0.0
        
        for restart in range(num_restarts):
            classes = []
            in_set = set()
            order = all_moduli[:]
            random.shuffle(order)
            # Bias toward smaller moduli
            order.sort(key=lambda m: (random.random() * 0.2 + 1.0/m, m))
            
            for m in order:
                if m in in_set:
                    continue
                r = find_valid_residue(m, classes)
                if r is not None:
                    classes.append([m, r])
                    in_set.add(m)
            
            if classes:
                score = calc_score_float(classes)
                if score > best_score:
                    best_score = score
                    best = [c[:] for c in classes]
        
        return best
    
    random.seed(42)
    
    # Run multiple strategies
    best_overall = []
    best_overall_score = 0.0
    
    strategies = [
        ("greedy_ascending", greedy_ascending),
        ("greedy_by_compatibility", greedy_by_compatibility),
        ("greedy_by_factor_richness", greedy_by_factor_richness),
        ("greedy_randomized", greedy_randomized),
    ]
    
    for name, func in strategies:
        if time.time() > deadline - 50:
            break
        try:
            classes = func()
            if classes and is_pairwise_valid(classes):
                score = calc_score_float(classes)
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall = [c[:] for c in classes]
        except Exception:
            pass
    
    # Local search on the best initial solution
    if best_overall and time.time() < deadline - 20:
        ls_time = min(time.time() + 12, deadline - 15)
        improved_classes, improved_score = local_search(best_overall, ls_time)
        if improved_score > best_overall_score:
            best_overall = improved_classes
            best_overall_score = improved_score
    
    # Simulated annealing to escape local optima
    if best_overall and time.time() < deadline - 10:
        sa_time = min(time.time() + 10, deadline - 6)
        sa_classes, sa_score = simulated_annealing(best_overall, sa_time)
        if sa_score > best_overall_score:
            best_overall = sa_classes
            best_overall_score = sa_score
    
    # Residue reoptimization
    if best_overall and time.time() < deadline - 4:
        ro_classes, ro_score = residue_reoptimization(best_overall, deadline - 3)
        if ro_score > best_overall_score:
            best_overall = ro_classes
            best_overall_score = ro_score
    
    # Final local search pass
    if best_overall and time.time() < deadline - 2:
        ls_time2 = min(time.time() + 2, deadline - 1)
        improved_classes2, improved_score2 = local_search(best_overall, ls_time2)
        if improved_score2 > best_overall_score:
            best_overall = improved_classes2
            best_overall_score = improved_score2
    
    # Final polish: try to add any remaining moduli
    if best_overall and time.time() < deadline - 0.5:
        in_set = set(c[0] for c in best_overall)
        changed = True
        while changed and time.time() < deadline - 0.5:
            changed = False
            for m in sorted(all_moduli):
                if m in in_set:
                    continue
                r = find_valid_residue(m, best_overall)
                if r is not None:
                    best_overall.append([m, r])
                    in_set.add(m)
                    changed = True
    
    # Final validation
    if best_overall:
        if not is_pairwise_valid(best_overall):
            best_overall = greedy_ascending()
            if not is_pairwise_valid(best_overall):
                best_overall = []
    
    return [c[:] for c in best_overall]

# EVOLVE-BLOCK-END
