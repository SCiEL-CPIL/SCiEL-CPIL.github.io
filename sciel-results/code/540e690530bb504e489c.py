"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin for evaluator's 150s limit
    
    best_score = Fraction(0, 1)
    best_classes = []
    
    def find_valid_residues(moduli_list):
        """Try to assign residues to make pairwise disjoint. Return residues or None."""
        n = len(moduli_list)
        if n == 0:
            return []
        
        residues = [0] * n
        for i in range(n):
            m = moduli_list[i]
            found = False
            for r in range(m):
                ok = True
                for j in range(i):
                    m2 = moduli_list[j]
                    g = gcd(m, m2)
                    if (r - residues[j]) % g == 0:
                        ok = False
                        break
                if ok:
                    residues[i] = r
                    found = True
                    break
            if not found:
                return None
        return residues
    
    def score_of(moduli_list):
        return sum(Fraction(1, m) for m in moduli_list)
    
    # Strategy 1: Systematic search over subsets of small moduli
    # Focus on moduli that are multiples of small numbers (2,3,4,5,6) to ensure shared factors
    candidate_moduli = []
    for m in range(20, 201):
        # Prefer moduli with many small factors
        small_factors = sum(1 for d in range(2, 10) if m % d == 0)
        candidate_moduli.append((m, small_factors))
    
    # Sort by: fewer small factors first? No, we want moduli that CAN share residues.
    # Actually, moduli that are multiples of the same small number can be made disjoint.
    # Let's group by their smallest prime factor or by divisibility patterns.
    
    # Strategy 2: Greedy with backtracking - try adding moduli in order of decreasing 1/m
    all_moduli = list(range(20, 201))
    all_moduli.sort()  # ascending m = descending 1/m
    
    # Try multiple greedy orders
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
        
        # Different greedy strategies
        if attempt == 0:
            # Pure ascending
            order = all_moduli
        elif attempt == 1:
            # Ascending but skip some to allow more flexibility
            order = [m for m in all_moduli if m % 2 == 0] + [m for m in all_moduli if m % 2 != 0]
        elif attempt == 2:
            # Group by gcd with 20
            groups = {}
            for m in all_moduli:
                g = gcd(m, 20)
                groups.setdefault(g, []).append(m)
            order = []
            for g in sorted(groups.keys(), reverse=True):
                order.extend(sorted(groups[g]))
        elif attempt == 3:
            # Prioritize multiples of 4, then 6, then others
            mult4 = [m for m in all_moduli if m % 4 == 0]
            mult6 = [m for m in all_moduli if m % 6 == 0 and m % 4 != 0]
            rest = [m for m in all_moduli if m % 4 != 0 and m % 6 != 0]
            order = sorted(mult4) + sorted(mult6) + sorted(rest)
        else:
            # Random-ish: interleave small and large
            order = []
            for i in range(0, len(all_moduli), 2):
                if i < len(all_moduli):
                    order.append(all_moduli[i])
                if i+1 < len(all_moduli):
                    order.append(all_moduli[i+1])
        
        greedy_set = []
        greedy_residues = []
        for m in order:
            if time.time() > deadline - 10:
                break
            residues = find_valid_residues(greedy_set + [m])
            if residues is not None:
                greedy_set.append(m)
                greedy_residues = residues
        
        if greedy_set:
            s = score_of(greedy_set)
            if s > best_score:
                best_score = s
                best_classes = [[greedy_set[i], greedy_residues[i]] for i in range(len(greedy_set))]
    
    # Strategy 3: Local search - try swapping out moduli
    if best_classes and time.time() < deadline - 20:
        current_moduli = [c[0] for c in best_classes]
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            # Try removing one modulus and adding a different one
            for remove_idx in range(len(current_moduli)):
                if time.time() > deadline - 20:
                    break
                trial_set = [m for i, m in enumerate(current_moduli) if i != remove_idx]
                for add_m in all_moduli:
                    if add_m in trial_set:
                        continue
                    trial_set2 = trial_set + [add_m]
                    trial_set2.sort()
                    residues = find_valid_residues(trial_set2)
                    if residues is not None:
                        s = score_of(trial_set2)
                        if s > current_score:
                            current_score = s
                            current_moduli = trial_set2
                            best_score = s
                            best_classes = [[trial_set2[i], residues[i]] for i in range(len(trial_set2))]
                            improved = True
                            break
                if time.time() > deadline - 20:
                    break
    
    # Ensure we have at least the seed result
    if not best_classes:
        best_classes = [[20 * j, j - 1] for j in range(1, 11)]
        best_score = sum(Fraction(1, 20*j) for j in range(1, 11))
    
    return best_classes
# EVOLVE-BLOCK-END
