"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    # 1 has no prime factors, prime_factors[1] remains empty
    
    # Build conflict graph: conflict[i] is a bitset (Python int) of all j != i that share a prime factor with i
    # We'll use Python's arbitrary precision integers as bitsets for speed
    # Bit j is set if j conflicts with i
    conflict = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if prime_factors[i] & prime_factors[j]:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Precompute weight (the number itself)
    weight = list(range(N + 1))
    
    # Greedy initialization: start with 1, then add numbers in descending order
    solution = [1]
    solution_set = set(solution)
    
    def is_compatible(a, b):
        return not (prime_factors[a] & prime_factors[b])
    
    for a in range(N, 1, -1):
        if all(is_compatible(a, b) for b in solution):
            solution.append(a)
            solution_set.add(a)
    
    current_sum = sum(solution)
    
    # Branch and bound for Maximum Weight Independent Set
    # We use a recursive approach with memoization on the "available" set (represented as a frozenset or bitset)
    # Since N=256, we can't use full memoization easily, but we can use branch and bound with good pruning
    
    # Better approach: use a greedy coloring upper bound (like in maximum clique algorithms)
    # For independent set, we can use the fact that the upper bound is the sum of weights of all remaining vertices
    # but that's too loose. Instead, we use a greedy packing bound.
    
    # Let's implement a branch-and-bound that works on a list of candidate vertices
    # State: (candidates_tuple, current_sum, current_set)
    # But this is too slow for 256 vertices.
    
    # Alternative: use the fact that the problem structure is special.
    # Let's try a different approach: iterative improvement with better moves.
    
    # 1-for-1 swaps (already in original, but let's make it more thorough)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(solution)):
            a = solution[i]
            # Try to replace a with a larger number not in solution
            for b in range(N, a, -1):
                if b in solution_set:
                    continue
                # Check if b is compatible with all others
                compatible = True
                for c in solution:
                    if c == a:
                        continue
                    if not is_compatible(b, c):
                        compatible = False
                        break
                if compatible:
                    solution_set.remove(a)
                    solution_set.add(b)
                    solution[i] = b
                    current_sum += b - a
                    improved = True
                    break
    
    best_sum = current_sum
    best_solution = solution[:]
    
    # 2-for-1 and 1-for-2 moves
    # Try removing two elements and adding one
    for _ in range(50):  # Limited iterations
        if time.time() > deadline:
            break
        
        # Try 1-for-2: remove one, add two
        for i in range(len(solution)):
            a = solution[i]
            solution_set.remove(a)
            temp_solution = solution_set.copy()
            
            # Try to add two numbers
            added = []
            for b in range(N, 1, -1):
                if b in temp_solution:
                    continue
                if all(is_compatible(b, c) for c in temp_solution):
                    temp_solution.add(b)
                    added.append(b)
                    if len(added) == 2:
                        break
            
            if len(added) == 2:
                new_sum = current_sum - a + sum(added)
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_solution = list(temp_solution)
                    solution = list(temp_solution)
                    solution_set = temp_solution
                    current_sum = new_sum
            
            # Restore
            solution_set.add(a)
            for b in added:
                solution_set.discard(b)
        
        # Try 2-for-1: remove two, add one
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                a = solution[i]
                b = solution[j]
                solution_set.remove(a)
                solution_set.remove(b)
                temp_solution = solution_set.copy()
                
                # Try to add one number
                best_add = -1
                for c in range(N, 1, -1):
                    if c in temp_solution:
                        continue
                    if all(is_compatible(c, d) for d in temp_solution):
                        best_add = c
                        break
                
                if best_add > 0:
                    new_sum = current_sum - a - b + best_add
                    if new_sum > best_sum:
                        best_sum = new_sum
                        temp_solution.add(best_add)
                        best_solution = list(temp_solution)
                        solution = list(temp_solution)
                        solution_set = temp_solution
                        current_sum = new_sum
                
                # Restore
                solution_set.add(a)
                solution_set.add(b)
    
    # Final 1-for-1 polish
    solution = best_solution[:]
    solution_set = set(solution)
    current_sum = best_sum
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(solution)):
            a = solution[i]
            for b in range(N, a, -1):
                if b in solution_set:
                    continue
                compatible = True
                for c in solution:
                    if c == a:
                        continue
                    if not is_compatible(b, c):
                        compatible = False
                        break
                if compatible:
                    solution_set.remove(a)
                    solution_set.add(b)
                    solution[i] = b
                    current_sum += b - a
                    improved = True
                    break
    
    return [int(x) for x in solution]
# EVOLVE-BLOCK-END
