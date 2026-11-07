<<<<<<< SEARCH
import math
import time
import sys
import random
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np
import scipy.sparse as sp

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    N = 256
    
    # Always include 1
    # Solve for 2..N
    candidates = list(range(2, N + 1))
    n = len(candidates)
    
    # 1. Try MILP for Exact Solution
    try:
        # Build conflict constraints
        # x_i + x_j <= 1 for gcd(i, j) > 1
        row_idx = []
        col_idx = []
        data = []
        
        constraint_id = 0
        # Optimization: Only check pairs where gcd > 1
        # Precompute prime factors for faster gcd checks
        def get_prime_factors(num):
            factors = set()
            d = 2
            while d * d <= num:
                while num % d == 0:
                    factors.add(d)
                    num //= d
                d += 1
            if num > 1:
                factors.add(num)
            return factors
        
        # Precompute prime factors for all candidates
        prime_factors = {}
        for i, ci in enumerate(candidates):
            prime_factors[i] = get_prime_factors(ci)
        
        for i in range(n):
            ci = candidates[i]
            pi = prime_factors[i]
            for j in range(i + 1, n):
                cj = candidates[j]
                pj = prime_factors[j]
                if pi & pj:  # If they share any prime factor, gcd > 1
                    row_idx.append(constraint_id)
                    col_idx.append(i)
                    data.append(1)
                    
                    row_idx.append(constraint_id)
                    col_idx.append(j)
                    data.append(1)
                    
                    constraint_id += 1
                    
                    # Time check during construction (should be negligible)
                    if time.time() > deadline - 5.0:
                        raise TimeoutError("Construction taking too long")
        
        if constraint_id == 0:
            # No conflicts, take all
            return [1] + candidates
        
        A_ub = sp.csr_matrix((data, (row_idx, col_idx)), shape=(constraint_id, n))
        
        # Objective: Maximize sum(i * x_i) -> Minimize -sum(i * x_i)
        c = np.array([-float(ci) for ci in candidates])
        
        # Constraints: A_ub @ x <= 1
        lincons = LinearConstraint(A_ub, lb=0, ub=1)
        
        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: x must be integer
        integrality = np.ones(n)
        
        # Solve with time limit
        remaining_time = deadline - time.time() - 2.0
        if remaining_time < 1.0:
            return _fallback_construct(N, max(0.5, deadline - time.time()))
            
        res = milp(c=c, constraints=lincons, integrality=integrality, bounds=bounds,
                   options={'time_limit': remaining_time})
        
        if time.time() > deadline:
            return _fallback_construct(N, max(0.5, deadline - time.time()))
            
        if not res.success or res.x is None:
            # If solver fails, fallback
            return _fallback_construct(N, max(0.5, deadline - time.time()))
            
        x = res.x
        # Round to nearest integer to handle floating point noise
        x = np.round(x).astype(int)
        
        # Construct result
        result = [1]  # Always include 1
        
        for i in range(n):
            if x[i] == 1:
                result.append(candidates[i])
        
        # Sanity check: Verify pairwise coprimality
        # If invalid, fallback
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if math.gcd(result[i], result[j]) != 1:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            return _fallback_construct(N, max(0.5, deadline - time.time()))
                    
        # Ensure types are Python int
        result = [int(r) for r in result]
        return result
        
    except Exception as e:
        # In case of any error, fallback
        return _fallback_construct(N, deadline - time.time())
=======
import math
import time
import sys
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    N = 256
    
    # Precompute prime factors for all numbers 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for p in range(2, N + 1):
        if not prime_factors[p]:  # p is prime
            for multiple in range(p, N + 1, p):
                prime_factors[multiple].add(p)
    
    # Precompute conflict graph: for each number, which other numbers conflict with it
    conflicts = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        for j in range(i + 1, N + 1):
            if prime_factors[i] & prime_factors[j]:  # share a prime factor
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Greedy initialization: select numbers in descending order
    selected = set()
    for n in range(N, 1, -1):
        if all(c not in selected for c in conflicts[n]):
            selected.add(n)
    selected.add(1)  # 1 is coprime to everything
    
    def get_conflicts_with_selected(n, selected_set):
        """Return set of selected elements that conflict with n."""
        return conflicts[n] & selected_set
    
    # Local search with 1-for-1 and 2-for-2 swaps
    current_sum = sum(selected)
    best_sum = current_sum
    best_result = list(selected)
    
    # Multi-start local search
    max_iterations = 50000
    iteration = 0
    
    while iteration < max_iterations and time.time() < deadline - 1.0:
        improved = False
        iteration += 1
        
        # Try 1-for-1 swaps: remove one selected, add one unselected
        selected_list = list(selected)
        random.shuffle(selected_list)
        
        for n_out in selected_list:
            if n_out == 1:
                continue
            
            # Find best replacement
            best_in = None
            best_gain = 0
            
            for n_in in range(N, 1, -1):
                if n_in in selected:
                    continue
                
                # Check if n_in conflicts with any selected element (except n_out)
                # n_in must not conflict with (selected - {n_out})
                rest = selected - {n_out}
                if get_conflicts_with_selected(n_in, rest):
                    continue
                
                gain = n_in - n_out
                if gain > best_gain:
                    best_gain = gain
                    best_in = n_in
                    break  # We're iterating descending, so first valid is best
            
            if best_in is not None:
                selected.remove(n_out)
                selected.add(best_in)
                current_sum += best_gain
                improved = True
                break
        
        if improved:
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = list(selected)
            continue
        
        # Try 2-for-2 swaps: remove two selected, add two unselected
        # Only attempt if time allows
        if time.time() < deadline - 2.0:
            selected_list = list(selected)
            random.shuffle(selected_list)
            
            found_2for2 = False
            for i in range(len(selected_list)):
                n_out1 = selected_list[i]
                if n_out1 == 1:
                    continue
                
                for j in range(i + 1, len(selected_list)):
                    n_out2 = selected_list[j]
                    if n_out2 == 1:
                        continue
                    
                    # Find best pair of replacements
                    best_in1 = None
                    best_in2 = None
                    best_gain = 0
                    
                    # Try all pairs of unselected numbers
                    unselected = [x for x in range(N, 1, -1) if x not in selected]
                    
                    for idx1 in range(len(unselected)):
                        n_in1 = unselected[idx1]
                        
                        # n_in1 must not conflict with (selected - {n_out1, n_out2})
                        rest = selected - {n_out1, n_out2}
                        if get_conflicts_with_selected(n_in1, rest):
                            continue
                        
                        for idx2 in range(idx1 + 1, len(unselected)):
                            n_in2 = unselected[idx2]
                            
                            # n_in2 must not conflict with (selected - {n_out1, n_out2})
                            if get_conflicts_with_selected(n_in2, rest):
                                continue
                            
                            # n_in1 and n_in2 must not conflict with each other
                            if n_in1 in conflicts[n_in2]:
                                continue
                            
                            gain = (n_in1 + n_in2) - (n_out1 + n_out2)
                            if gain > best_gain:
                                best_gain = gain
                                best_in1 = n_in1
                                best_in2 = n_in2
                                break  # n_in2 is descending, first valid is best
                        
                        if best_in2 is not None:
                            break  # n_in1 is descending, first valid pair is best
                    
                    if best_in1 is not None and best_in2 is not None:
                        selected.remove(n_out1)
                        selected.remove(n_out2)
                        selected.add(best_in1)
                        selected.add(best_in2)
                        current_sum += best_gain
                        improved = True
                        found_2for2 = True
                        break
                
                if found_2for2:
                    break
            
            if found_2for2:
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = list(selected)
                continue
        
        # No improvement found, try a random restart with different ordering
        # Occasionally perturb the current solution
        if iteration % 100 == 0:
            # Randomly remove 1-3 elements and re-greedy
            num_remove = random.randint(1, 3)
            removable = [x for x in selected if x != 1]
            if removable:
                to_remove = random.sample(removable, min(num_remove, len(removable)))
                for r in to_remove:
                    selected.remove(r)
                    current_sum -= r
                
                # Re-greedy to fill back in
                for n in range(N, 1, -1):
                    if n not in selected:
                        if all(c not in selected for c in conflicts[n]):
                            selected.add(n)
                            current_sum += n
        
        # If stuck for too long, do a fresh greedy with random tie-breaking
        if iteration % 500 == 0:
            # Fresh start with random ordering
            new_selected = {1}
            candidates = list(range(N, 1, -1))
            random.shuffle(candidates)
            for n in candidates:
                if all(c not in new_selected for c in conflicts[n]):
                    new_selected.add(n)
            
            new_sum = sum(new_selected)
            if new_sum > current_sum:
                selected = new_selected
                current_sum = new_sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = list(selected)
    
    # Final validation
    result = best_result
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Emergency fallback: just primes + 1
        result = [1] + [p for p in range(2, N + 1) if not prime_factors[p] or len(prime_factors[p]) == 1 and min(prime_factors[p]) == max(prime_factors[p]) and p in prime_factors[p]]
        # Simpler: just use numbers with no conflicts
        result = [1]
        for n in range(N, 1, -1):
            if all(c not in result for c in conflicts[n]):
                result.append(n)
    
    return [int(x) for x in result]
>>>>>>> REPLACE

