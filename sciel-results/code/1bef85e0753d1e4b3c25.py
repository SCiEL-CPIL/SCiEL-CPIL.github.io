"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd
from itertools import combinations
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    N = 512
    
    # Precompute exact rational values
    exact_values = [Fraction(0, 1)] * N
    float_values = [0.0] * N
    for a in range(1, N):
        exact_values[a] = Fraction(1, N - a)
        float_values[a] = float(exact_values[a])
        
    # Precompute prime factors
    def get_prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    all_factors = [set() for _ in range(N)]
    for i in range(2, N):
        all_factors[i] = get_prime_factors(i)
        
    # State
    chosen = []
    chosen_set = set()
    prime_count = [0] * N  # prime_count[p] is count of chosen numbers divisible by p
    current_score = Fraction(0, 1)
    
    def add_num(a):
        nonlocal current_score
        chosen.append(a)
        chosen_set.add(a)
        current_score += exact_values[a]
        for p in all_factors[a]:
            prime_count[p] += 1
            
    def remove_num(a):
        nonlocal current_score
        """Properly remove a number from the state."""
        if a in chosen:
            idx = chosen.index(a)
            chosen.pop(idx)
            chosen_set.discard(a)
            current_score -= exact_values[a]
            for p in all_factors[a]:
                prime_count[p] -= 1

    def is_coprime_to_state(a):
        """Checks if a is coprime to all numbers currently in 'chosen' (reflected in prime_count)"""
        fa = all_factors[a]
        if not fa:
            return True
        for p in fa:
            if prime_count[p] > 0:
                return False
        return True

    # Phase 1: Greedy Initialization with lookahead
    # Try multiple greedy orders and keep the best
    greedy_orders = [
        list(range(511, 0, -1)),  # descending
        list(range(1, N)),  # ascending
        [511 - i for i in range(511)],  # same as descending
        # Custom order: prioritize numbers close to 511 but with fewer prime factors
        sorted(range(1, N), key=lambda x: (-(N - x), len(all_factors[x]))),
    ]
    
    best_greedy_score = Fraction(0, 1)
    best_greedy_chosen = []
    
    for order in greedy_orders:
        # Reset state
        chosen.clear()
        chosen_set.clear()
        prime_count = [0] * N
        current_score = Fraction(0, 1)
        
        for a in order:
            if is_coprime_to_state(a):
                add_num(a)
        
        if current_score > best_greedy_score:
            best_greedy_score = current_score
            best_greedy_chosen = chosen[:]
    
    # Restore best greedy solution
    chosen = best_greedy_chosen[:]
    chosen_set = set(chosen)
    prime_count = [0] * N
    current_score = Fraction(0, 1)
    for a in chosen:
        current_score += exact_values[a]
        for p in all_factors[a]:
            prime_count[p] += 1
    
    best_score = current_score
    best_chosen = chosen[:]
    
    def get_top_candidates(L):
        """
        Returns top L candidates coprime to current state, sorted by exact value descending.
        """
        cands = []
        for a in range(511, 0, -1):
            if a in chosen_set:
                continue
            if is_coprime_to_state(a):
                cands.append(a)
                if len(cands) >= L:
                    break
        return cands

    def solve_k_opt(cands, k):
        """
        Finds best k numbers from cands that are pairwise coprime.
        cands is sorted by value descending.
        Returns list of selected numbers and their total exact value.
        """
        if k == 1:
            if not cands: return [], Fraction(0, 1)
            return [cands[0]], exact_values[cands[0]]
        
        if k == 2:
            best_val = Fraction(0, 1)
            best_pair = []
            for i in range(len(cands)):
                a = cands[i]
                fa = all_factors[a]
                for j in range(i + 1, len(cands)):
                    b = cands[j]
                    if not (fa & all_factors[b]):
                        val = exact_values[a] + exact_values[b]
                        if val > best_val:
                            best_val = val
                            best_pair = [a, b]
            return best_pair, best_val
            
        if k == 3:
            best_val = Fraction(0, 1)
            best_triple = []
            search_cands = cands[:20] 
            for i in range(len(search_cands)):
                a = search_cands[i]
                fa = all_factors[a]
                for j in range(i + 1, len(search_cands)):
                    b = search_cands[j]
                    fb = all_factors[b]
                    if fa & fb: continue
                    for k_idx in range(j + 1, len(search_cands)):
                        c = search_cands[k_idx]
                        fc = all_factors[c]
                        if (fa & fc) or (fb & fc): continue
                        val = exact_values[a] + exact_values[b] + exact_values[c]
                        if val > best_val:
                            best_val = val
                            best_triple = [a, b, c]
            return best_triple, best_val
        return [], Fraction(0, 1)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try Add 1
        cands = get_top_candidates(10)
        if cands:
            a = cands[0]
            if is_coprime_to_state(a):
                add_num(a)
                if current_score > best_score:
                    best_score = current_score
                    best_chosen = chosen[:]
                    improved = True
                else:
                    remove_num(a)

        # 2. Try 1-Opt Swap
        for i in range(len(chosen)):
            if time.time() > deadline:
                return best_chosen
            
            old_a = chosen[i]
            
            # Temporarily remove old_a
            remove_num(old_a)
            
            # Find best replacement
            cands = get_top_candidates(30)
            new_subset, new_val = solve_k_opt(cands, 1)
            
            old_val = exact_values[old_a]
            
            if new_val > old_val and new_subset:
                for a in new_subset:
                    add_num(a)
                best_score = current_score
                best_chosen = chosen[:]
                improved = True
            else:
                add_num(old_a)
                
        # 3. Try 2-Opt Swap
        if len(chosen) >= 2:
            # Try more pairs if we have time
            max_pairs = 10 if time.time() < deadline - 30 else 5
            pairs_to_try = list(combinations(range(len(chosen)), 2))[:max_pairs]
            for i, j in pairs_to_try:
                if time.time() > deadline:
                    return best_chosen
                
                old_a = chosen[i]
                old_b = chosen[j]
                
                # Remove both
                remove_num(old_a)
                remove_num(old_b)
                
                cands = get_top_candidates(30)
                new_subset, new_val = solve_k_opt(cands, 2)
                
                old_total_val = exact_values[old_a] + exact_values[old_b]
                
                if new_val > old_total_val and len(new_subset) == 2:
                    for a in new_subset:
                        add_num(a)
                    best_score = current_score
                    best_chosen = chosen[:]
                    improved = True
                else:
                    add_num(old_a)
                    add_num(old_b)
                    
        # 4. Try 3-Opt Swap (Limited)
        if len(chosen) >= 3:
            max_triples = 3 if time.time() < deadline - 20 else 1
            triples_to_try = list(combinations(range(len(chosen)), 3))[:max_triples]
            for i, j, k in triples_to_try:
                if time.time() > deadline:
                    return best_chosen
                
                old_a = chosen[i]
                old_b = chosen[j]
                old_c = chosen[k]
                
                remove_num(old_a)
                remove_num(old_b)
                remove_num(old_c)
                
                cands = get_top_candidates(20)
                new_subset, new_val = solve_k_opt(cands, 3)
                
                old_total_val = exact_values[old_a] + exact_values[old_b] + exact_values[old_c]
                
                if new_val > old_total_val and len(new_subset) == 3:
                    for a in new_subset:
                        add_num(a)
                    best_score = current_score
                    best_chosen = chosen[:]
                    improved = True
                else:
                    add_num(old_a)
                    add_num(old_b)
                    add_num(old_c)

    # Final polish: try to add any remaining candidates
    for a in range(511, 0, -1):
        if a not in best_chosen and is_coprime_to_state(a):
            # Check if adding improves
            temp_score = best_score + exact_values[a]
            if temp_score > best_score:
                best_chosen.append(a)
                best_score = temp_score

    return best_chosen
# EVOLVE-BLOCK-END
