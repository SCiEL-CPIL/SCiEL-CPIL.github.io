"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    def time_left():
        return deadline - time.time()

    best_score = Fraction(0, 1)
    best_classes = []

    def assign_residues_greedy(moduli_list):
        """Assign residues greedily to make pairwise disjoint. Return residues or None."""
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

    def try_update(moduli_list):
        nonlocal best_score, best_classes
        residues = assign_residues_greedy(moduli_list)
        if residues is not None:
            s = score_of(moduli_list)
            if s > best_score:
                best_score = s
                best_classes = [[moduli_list[i], residues[i]] for i in range(len(moduli_list))]

    def can_assign(moduli_list):
        """Check if moduli_list can have pairwise disjoint residues."""
        return assign_residues_greedy(moduli_list) is not None

    # Strategy 1: Systematic search over common factor base d
    # For each d, take the smallest k multiples of d in [20,200]
    for d in range(2, 201):
        if time_left() < 5:
            break
        multiples = sorted([m for m in range(20, 201) if m % d == 0])
        if len(multiples) < 2:
            continue
        # Try increasing sizes
        for k in range(2, len(multiples) + 1):
            if time_left() < 5:
                break
            subset = multiples[:k]
            try_update(subset)

    # Strategy 2: Greedy addition with backtracking-like refinement
    all_moduli = list(range(20, 201))
    all_moduli.sort()

    greedy_set = []
    for m in all_moduli:
        if time_left() < 10:
            break
        test_set = greedy_set + [m]
        residues = assign_residues_greedy(test_set)
        if residues is not None:
            greedy_set = test_set

    if greedy_set:
        try_update(greedy_set)

    # Strategy 3: Local search / hill climbing
    if best_classes:
        current_moduli = [c[0] for c in best_classes]
        improved = True
        while improved and time_left() > 15:
            improved = False
            # Try adding a modulus not in current set
            for m in all_moduli:
                if m in current_moduli:
                    continue
                test_set = current_moduli + [m]
                test_set.sort()
                residues = assign_residues_greedy(test_set)
                if residues is not None:
                    s = score_of(test_set)
                    if s > best_score:
                        best_score = s
                        best_classes = [[test_set[i], residues[i]] for i in range(len(test_set))]
                        current_moduli = test_set
                        improved = True
                        break
            # Try removing a modulus and replacing with a smaller one
            for i, m in enumerate(current_moduli):
                if time_left() < 15:
                    break
                for m2 in all_moduli:
                    if m2 >= m or m2 in current_moduli:
                        continue
                    test_set = [x for j, x in enumerate(current_moduli) if j != i]
                    test_set.append(m2)
                    test_set.sort()
                    residues = assign_residues_greedy(test_set)
                    if residues is not None:
                        s = score_of(test_set)
                        if s > best_score:
                            best_score = s
                            best_classes = [[test_set[i], residues[i]] for i in range(len(test_set))]
                            current_moduli = test_set
                            improved = True
                            break
                if improved:
                    break

    # Strategy 4: Randomized restarts with different orderings
    random.seed(42)
    for restart in range(30):
        if time_left() < 5:
            break
        shuffled = list(range(20, 201))
        random.shuffle(shuffled)
        greedy_set = []
        for m in shuffled:
            test_set = greedy_set + [m]
            residues = assign_residues_greedy(test_set)
            if residues is not None:
                greedy_set = test_set
        if greedy_set:
            try_update(greedy_set)

    # Strategy 5: Focused search on small moduli with high factor overlap
    small_moduli = list(range(20, 51))
    for size in range(3, min(7, len(small_moduli) + 1)):
        if time_left() < 3:
            break
        for combo in combinations(small_moduli, size):
            if time_left() < 3:
                break
            try_update(list(combo))

    # Strategy 6: Branch and bound on small moduli
    # Focus on moduli 20-60 where reciprocal sum is highest
    candidate_moduli = list(range(20, 61))
    
    def branch_and_bound(candidates, current_set, current_score):
        nonlocal best_score, best_classes
        if time_left() < 3:
            return
        if not candidates:
            if current_score > best_score:
                best_score = current_score
                residues = assign_residues_greedy(current_set)
                if residues is not None:
                    best_classes = [[current_set[i], residues[i]] for i in range(len(current_set))]
            return
        
        # Upper bound: current score + sum of reciprocals of remaining candidates
        upper_bound = current_score + sum(Fraction(1, m) for m in candidates)
        if upper_bound <= best_score:
            return
        
        # Branch on first candidate
        m = candidates[0]
        rest = candidates[1:]
        
        # Option 1: Include m
        if can_assign(current_set + [m]):
            branch_and_bound(rest, current_set + [m], current_score + Fraction(1, m))
        
        # Option 2: Exclude m
        branch_and_bound(rest, current_set, current_score)
    
    if time_left() > 10:
        branch_and_bound(candidate_moduli, [], Fraction(0, 1))

    # Ensure we have at least a valid result
    if not best_classes:
        multiples = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        residues = assign_residues_greedy(multiples)
        if residues is not None:
            best_classes = [[multiples[i], residues[i]] for i in range(len(multiples))]
            best_score = score_of(multiples)
        else:
            best_classes = [[20, 0]]
            best_score = Fraction(1, 20)

    return best_classes
# EVOLVE-BLOCK-END
