import time
import math
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for 150s limit
    
    N = 512
    
    # Precompute prime factors for 1..511
    prime_factors = [set() for _ in range(N)]
    for i in range(2, N):
        if not prime_factors[i]:
            for j in range(i, N, i):
                prime_factors[j].add(i)
    
    # Precompute scores as Fractions
    scores = [Fraction(0, 1) for _ in range(N)]
    for a in range(1, N):
        scores[a] = Fraction(1, N - a)
    
    # Greedy Initialization
    selected = []
    used_primes = set()
    
    for a in range(N - 1, 0, -1):
        if not prime_factors[a].intersection(used_primes):
            selected.append(a)
            used_primes.update(prime_factors[a])
    
    current_score = sum(scores[a] for a in selected)
    best_score = current_score
    best_selected = selected[:]
    
    # Local Search
    # We will try to improve by swapping.
    # To avoid infinite loops, we only accept if score strictly increases.
    # Since scores are discrete fractions, this will terminate.
    
    def get_score(sel):
        return sum(scores[a] for a in sel)
    
    def is_valid_addition(a, sel_without_removed, used_primes_without_removed):
        if not prime_factors[a].intersection(used_primes_without_removed):
            return True
        return False

    # Helper to check if a set of candidates is pairwise coprime and coprime to base
    def check_candidates(candidates, base_selected, base_used_primes):
        # Check pairwise coprime among candidates
        cand_primes = [prime_factors[a] for a in candidates]
        # Check against base
        for i in range(len(candidates)):
            if cand_primes[i].intersection(base_used_primes):
                return False
            for j in range(i + 1, len(candidates)):
                if cand_primes[i].intersection(cand_primes[j]):
                    return False
        return True

    # We iterate until no improvement or deadline
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try 1-for-1 swaps
        for i, a_i in enumerate(selected):
            if time.time() > deadline:
                break
            # Calculate score without a_i
            score_without_i = current_score - scores[a_i]
            # Primes without a_i
            used_primes_without_i = set(used_primes)
            used_primes_without_i.difference_update(prime_factors[a_i])
            
            # Find best replacement
            best_gain = Fraction(0, 1)
            best_j = -1
            
            # We can optimize by only checking unselected numbers
            # But iterating all 1..511 is fast enough (511 iterations * |selected| ~ 100 * 511 = 50k ops)
            for a_j in range(1, N):
                if a_j in selected:
                    continue
                if not prime_factors[a_j].intersection(used_primes_without_i):
                    gain = scores[a_j] - scores[a_i]
                    if gain > best_gain:
                        best_gain = gain
                        best_j = a_j
            
            if best_j != -1 and best_gain > 0:
                # Apply swap
                selected[i] = best_j
                used_primes.difference_update(prime_factors[a_i])
                used_primes.update(prime_factors[best_j])
                current_score += best_gain
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_selected = selected[:]
                break # Restart search from top
        
        # 2. Try 1-for-2 swaps
        # Remove a_i, add a_j, a_k
        # This is more expensive: O(|selected| * N^2)
        # N=511, N^2=260k. |selected|~100. 26M ops. Might be too slow in Python if not optimized.
        # We can restrict candidates to those with high score (large a).
        # Or just try it if time permits.
        
        if time.time() < deadline - 10.0: # Leave 10s buffer
            for i, a_i in enumerate(selected):
                if time.time() > deadline:
                    break
                score_without_i = current_score - scores[a_i]
                used_primes_without_i = set(used_primes)
                used_primes_without_i.difference_update(prime_factors[a_i])
                
                # Candidates for replacement: unselected numbers
                # We want to pick two that are coprime to each other and to the rest.
                # Sort unselected by score descending?
                # Just iterate pairs of unselected numbers.
                # To speed up, we can pre-filter unselected numbers that are coprime to used_primes_without_i
                candidates = []
                for a in range(1, N):
                    if a in selected:
                        continue
                    if not prime_factors[a].intersection(used_primes_without_i):
                        candidates.append(a)
                
                # Now find best pair in candidates
                # Sort candidates by score descending to find best quickly?
                # Or just check all pairs.
                # Number of candidates might be small.
                if len(candidates) < 2:
                    continue
                
                # Check pairs
                best_gain = Fraction(0, 1)
                best_pair = (-1, -1)
                
                # Optimization: only check pairs where sum of scores > score[a_i]
                # f(a_j) + f(a_k) > f(a_i)
                
                for idx_j in range(len(candidates)):
                    a_j = candidates[idx_j]
                    # If a_j alone doesn't beat a_i, and a_k < a_j, sum might still beat?
                    # f is decreasing. So f(a_j) + f(a_k) <= 2*f(a_j).
                    # If 2*f(a_j) < f(a_i), skip this j?
                    if scores[a_j] * 2 < scores[a_i]:
                        continue
                        
                    pf_j = prime_factors[a_j]
                    
                    for idx_k in range(idx_j + 1, len(candidates)):
                        a_k = candidates[idx_k]
                        
                        # Check coprime with j
                        if pf_j.intersection(prime_factors[a_k]):
                            continue
                        
                        # Check gain
                        gain = scores[a_j] + scores[a_k] - scores[a_i]
                        if gain > best_gain:
                            best_gain = gain
                            best_pair = (a_j, a_k)
                
                if best_pair[0] != -1 and best_gain > 0:
                    # Apply swap
                    selected[i] = best_pair[0]
                    selected.append(best_pair[1])
                    used_primes.difference_update(prime_factors[a_i])
                    used_primes.update(prime_factors[best_pair[0]])
                    used_primes.update(prime_factors[best_pair[1]])
                    current_score += best_gain
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = selected[:]
                    break # Restart
        
        # 3. Try 2-for-1 swaps
        # Remove a_i, a_j, add a_k
        if time.time() < deadline - 5.0:
            # Iterate pairs in selected
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                a_i = selected[i]
                for j in range(i + 1, len(selected)):
                    a_j = selected[j]
                    
                    score_without_ij = current_score - scores[a_i] - scores[a_j]
                    used_primes_without_ij = set(used_primes)
                    used_primes_without_ij.difference_update(prime_factors[a_i])
                    used_primes_without_ij.difference_update(prime_factors[a_j])
                    
                    # Find best single replacement
                    best_gain = Fraction(0, 1)
                    best_k = -1
                    for a_k in range(1, N):
                        if a_k in selected:
                            continue
                        if not prime_factors[a_k].intersection(used_primes_without_ij):
                            gain = scores[a_k] - scores[a_i] - scores[a_j]
                            if gain > best_gain:
                                best_gain = gain
                                best_k = a_k
                    
                    if best_k != -1 and best_gain > 0:
                        # Apply swap
                        # Remove a_i, a_j
                        # Order matters for list removal
                        if i < j:
                            selected.pop(j)
                            selected.pop(i)
                        else:
                            selected.pop(i)
                            selected.pop(j)
                        selected.append(best_k)
                        
                        used_primes.difference_update(prime_factors[a_i])
                        used_primes.difference_update(prime_factors[a_j])
                        used_primes.update(prime_factors[best_k])
                        
                        current_score += best_gain
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = selected[:]
                        break # Restart
                if improved:
                    break

    return best_selected
