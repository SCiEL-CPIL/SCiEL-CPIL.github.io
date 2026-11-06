"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute weights: w[a] = 1/(512-a) for a in 1..511
    # We want to maximize sum of w[a] over pairwise coprime set
    
    # Step 1: Build conflict graph - two numbers conflict if gcd > 1
    # For branch and bound, we need efficient conflict checking
    
    # Step 2: Get initial solution via greedy (largest weights first)
    # Sort by weight descending (i.e., by a descending since w is decreasing in 512-a)
    
    # Greedy initialization
    greedy_set = []
    for a in range(511, 0, -1):
        if all(math.gcd(a, b) == 1 for b in greedy_set):
            greedy_set.append(a)
    
    best_sum = sum(1.0 / (512 - a) for a in greedy_set)
    best_solution = greedy_set.copy()
    
    # Step 3: Branch and bound with upper bound pruning
    # Upper bound: for remaining candidates, take sum of all weights (relaxation)
    # But we need a tighter bound. Use: sum of weights of candidates that are coprime to current partial solution
    
    # Precompute prime factors for each number to speed up gcd checks
    def prime_factors(n):
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
    
    # Precompute prime factor sets for 1..511
    pf = [set()] * 512
    for i in range(1, 512):
        pf[i] = prime_factors(i)
    
    def are_coprime(a, b):
        return len(pf[a] & pf[b]) == 0
    
    # Branch and bound
    # State: (current_set, available_candidates, current_weight)
    # We use a recursive approach with memoization-like pruning
    
    # For efficiency, represent candidate set as a bitset or sorted list
    # Since 511 is small, we can use a boolean array
    
    # Better approach: use a greedy upper bound based on maximum weight matching or just sum of top-k coprime candidates
    
    # Let's use a simpler but effective approach:
    # 1. Start from greedy solution
    # 2. Try to add elements that are coprime to all in current set
    # 3. Use local search with 2-swap moves
    
    # Enhanced local search: 1-swap and 2-swap
    
    current_set = set(greedy_set)
    current_weight = best_sum
    
    # Precompute for each a, the list of b in 1..511 such that gcd(a,b) > 1
    # This helps in quick conflict checking
    
    # Try 1-swap: remove one element, add one that improves
    improved = True
    max_iterations = 500
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        if time.time() > deadline:
            break
        
        improved = False
        iter_count += 1
        
        current_list = list(current_set)
        
        for i, a_out in enumerate(current_list):
            if time.time() > deadline:
                break
            
            # Remove a_out temporarily
            temp_set = current_set - {a_out}
            
            # Find best a_in to add
            best_gain = 0
            best_in = -1
            
            for a_in in range(511, 0, -1):
                if a_in in temp_set:
                    continue
                if not all(are_coprime(a_in, b) for b in temp_set):
                    continue
                
                gain = 1.0 / (512 - a_in) - 1.0 / (512 - a_out)
                if gain > best_gain:
                    best_gain = gain
                    best_in = a_in
            
            if best_in != -1 and best_gain > 1e-12:
                # Perform swap
                current_set.remove(a_out)
                current_set.add(best_in)
                current_weight += best_gain
                improved = True
                
                if current_weight > best_sum:
                    best_sum = current_weight
                    best_solution = list(current_set)
                
                break  # Restart with new configuration
        
        # Try 2-swap: remove two, add two
        if not improved:
            current_list = list(current_set)
            n = len(current_list)
            
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    
                    a_out1 = current_list[i]
                    a_out2 = current_list[j]
                    temp_set = current_set - {a_out1, a_out2}
                    
                    # Find best pair to add
                    best_gain = 0
                    best_pair = None
                    
                    # Try all pairs
                    for a_in1 in range(511, 0, -1):
                        if a_in1 in temp_set:
                            continue
                        if not all(are_coprime(a_in1, b) for b in temp_set):
                            continue
                        
                        for a_in2 in range(511, a_in1, -1):
                            if a_in2 in temp_set:
                                continue
                            if not all(are_coprime(a_in2, b) for b in temp_set):
                                continue
                            if not are_coprime(a_in1, a_in2):
                                continue
                            
                            gain = (1.0 / (512 - a_in1) + 1.0 / (512 - a_in2)) - (1.0 / (512 - a_out1) + 1.0 / (512 - a_out2))
                            if gain > best_gain:
                                best_gain = gain
                                best_pair = (a_in1, a_in2)
                    
                    if best_pair and best_gain > 1e-12:
                        current_set.remove(a_out1)
                        current_set.remove(a_out2)
                        current_set.add(best_pair[0])
                        current_set.add(best_pair[1])
                        current_weight += best_gain
                        improved = True
                        
                        if current_weight > best_sum:
                            best_sum = current_weight
                            best_solution = list(current_set)
                        
                        break  # Restart
                if improved:
                    break
        
    return best_solution
# EVOLVE-BLOCK-END
