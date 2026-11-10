"""Optimized approach: Prime-factor-based conflict detection + First-Improvement local search + SA."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, isqrt, log
from fractions import Fraction
from collections import defaultdict

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    if n < 25:
        return True
    if n % 5 == 0:
        return False
    if n < 49:
        return True
    if n % 7 == 0:
        return False
    if n < 121:
        return True
    i = 11
    while i <= isqrt(n):
        if n % i == 0:
            return False
        i += 2
    return True

def _get_prime_factors(n):
    """Return list of unique prime factors of n."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    N = 512
    
    # Precompute values and prime factors for all numbers 1..511
    values = [Fraction(0)] * N
    prime_factors = [[] for _ in range(N)]
    is_prime_arr = [False] * N
    
    for a in range(1, N):
        values[a] = Fraction(1, N - a)
        prime_factors[a] = _get_prime_factors(a)
        is_prime_arr[a] = _is_prime(a)
        
    # Identify primes
    primes = [a for a in range(1, N) if is_prime_arr[a]]
    
    # Initialize with all primes
    selected = set(primes)
    
    # Build mapping: prime -> set of selected elements that have this prime as a factor
    prime_to_selected = defaultdict(set)
    for b in selected:
        for p in prime_factors[b]:
            prime_to_selected[p].add(b)
    
    def get_conflicts(candidates):
        """Get all selected elements that conflict with any candidate."""
        conflict_set = set()
        for c in candidates:
            for p in prime_factors[c]:
                if p in prime_to_selected:
                    conflict_set.update(prime_to_selected[p])
        return conflict_set

    def apply_move(add_candidates, remove_elements):
        """Apply a move: remove elements, add candidates. Update prime_to_selected."""
        for b in remove_elements:
            for p in prime_factors[b]:
                if p in prime_to_selected:
                    prime_to_selected[p].discard(b)
                    if not prime_to_selected[p]:
                        del prime_to_selected[p]
        for c in add_candidates:
            for p in prime_factors[c]:
                prime_to_selected[p].add(c)
        selected.difference_update(remove_elements)
        selected.update(add_candidates)

    def get_score():
        return sum(values[a] for a in selected)
    
    current_score = get_score()
    
    # Phase 1: 1-for-k First-Improvement Search
    improved = True
    while improved:
        if time.time() > deadline - 30:
            break
            
        improved = False
        for c in range(1, N):
            if c in selected:
                continue
            
            conflict_set = set()
            for p in prime_factors[c]:
                if p in prime_to_selected:
                    conflict_set.update(prime_to_selected[p])
            
            conflict_cost = sum(values[b] for b in conflict_set)
            gain = values[c] - conflict_cost
            
            if gain > 0:
                apply_move([c], list(conflict_set))
                current_score += gain
                improved = True
                break
    
    # Phase 2: 2-for-k Search - optimized with precomputed conflict costs
    if time.time() < deadline - 20:
        # Build list of unselected composites
        unselected = [a for a in range(1, N) if a not in selected and a != 1]
        
        # Precompute conflict set and cost for each unselected element
        conflict_info = {}
        for a in unselected:
            conf = set()
            for p in prime_factors[a]:
                if p in prime_to_selected:
                    conf.update(prime_to_selected[p])
            conflict_info[a] = (conf, sum(values[b] for b in conf))
        
        # Sort by value descending, take top candidates
        unselected.sort(key=lambda x: values[x], reverse=True)
        top_candidates = unselected[:80]
        
        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
                
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None
            
            # Try pairs from top_candidates
            for i in range(len(top_candidates)):
                c1 = top_candidates[i]
                if c1 in selected:
                    continue
                conf1, cost1 = conflict_info[c1]
                
                for j in range(i + 1, len(top_candidates)):
                    c2 = top_candidates[j]
                    if c2 in selected:
                        continue
                    if gcd(c1, c2) != 1:
                        continue
                    
                    conf2, cost2 = conflict_info[c2]
                    combined_conf = conf1 | conf2
                    combined_cost = sum(values[b] for b in combined_conf)
                    gain = values[c1] + values[c2] - combined_cost
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_add = [c1, c2]
                        best_remove = list(combined_conf)
            
            if best_add is not None:
                apply_move(best_add, best_remove)
                current_score += best_gain
                improved = True
                # Refresh conflict info after move
                for a in list(conflict_info.keys()):
                    if a in selected:
                        del conflict_info[a]
                    else:
                        conf = set()
                        for p in prime_factors[a]:
                            if p in prime_to_selected:
                                conf.update(prime_to_selected[p])
                        conflict_info[a] = (conf, sum(values[b] for b in conf))
                # Add newly unselected elements
                for a in range(1, N):
                    if a not in selected and a != 1 and a not in conflict_info:
                        conf = set()
                        for p in prime_factors[a]:
                            if p in prime_to_selected:
                                conf.update(prime_to_selected[p])
                        conflict_info[a] = (conf, sum(values[b] for b in conf))
    
    # Phase 3: 3-for-k Search (lightweight)
    if time.time() < deadline - 15:
        unselected = [a for a in range(1, N) if a not in selected and a != 1]
        unselected.sort(key=lambda x: values[x], reverse=True)
        top3 = unselected[:40]
        
        improved = True
        while improved:
            if time.time() > deadline - 8:
                break
                
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None
            
            # Try triples from top3
            for i in range(len(top3)):
                c1 = top3[i]
                if c1 in selected:
                    continue
                for j in range(i + 1, len(top3)):
                    c2 = top3[j]
                    if c2 in selected or gcd(c1, c2) != 1:
                        continue
                    for k in range(j + 1, len(top3)):
                        c3 = top3[k]
                        if c3 in selected or gcd(c1, c3) != 1 or gcd(c2, c3) != 1:
                            continue
                        
                        conflict_set = set()
                        for c in (c1, c2, c3):
                            for p in prime_factors[c]:
                                if p in prime_to_selected:
                                    conflict_set.update(prime_to_selected[p])
                        
                        conflict_cost = sum(values[b] for b in conflict_set)
                        gain = values[c1] + values[c2] + values[c3] - conflict_cost
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_add = [c1, c2, c3]
                            best_remove = list(conflict_set)
            
            if best_add is not None:
                apply_move(best_add, best_remove)
                current_score += best_gain
                improved = True
    
    # Phase 4: Simulated Annealing with precomputed unselected list
    if time.time() < deadline - 5:
        T = 0.05
        T_min = 0.001
        cooling = 0.995
        
        sa_deadline = deadline - 2
        
        # Maintain unselected list efficiently
        unselected_set = set(a for a in range(1, N) if a not in selected)
        
        while T > T_min and time.time() < sa_deadline:
            if not selected or not unselected_set:
                break
            a_out = random.choice(list(selected))
            a_in = random.choice(list(unselected_set))
            
            # Check if a_in is coprime to all selected except a_out
            ok = True
            for p in prime_factors[a_in]:
                if p in prime_to_selected:
                    conflicts = prime_to_selected[p]
                    if len(conflicts) == 1 and a_out in conflicts:
                        continue
                    else:
                        ok = False
                        break
            
            if not ok:
                continue
            
            gain = values[a_in] - values[a_out]
            
            if gain > 0 or random.random() < pow(2.718281828, gain / T):
                apply_move([a_in], [a_out])
                current_score += gain
                unselected_set.discard(a_in)
                unselected_set.add(a_out)
            
            T *= cooling
    
    # Final Verification
    result = list(selected)
    for a in result:
        if a < 1 or a > 511:
            raise ValueError(f"Out of range: {a}")
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                raise ValueError(f"Not coprime: {result[i]}, {result[j]}")
    
    return result
# EVOLVE-BLOCK-END
