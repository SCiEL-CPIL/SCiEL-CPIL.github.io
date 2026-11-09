"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of denominators
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators, 1)
    
    # Values scaled by LCM
    values = [L // d for d in denominators]
    
    # Helper to calculate sum of a subset represented as a bitmask or list
    # We will use a set of indices for the current solution
    
    def get_sum(indices):
        s = 0
        for i in indices:
            s += values[i]
        return s
    
    # 1. Greedy Initialization
    # Sort indices by value descending (smallest denominator first) to pack tightly
    sorted_indices = sorted(range(n), key=lambda i: values[i], reverse=True)
    
    best_solution = None
    best_sum = 0
    
    # Try a few different greedy strategies or random restarts
    
    def greedy_init(order):
        curr_sum = 0
        curr_set = set()
        for i in order:
            if curr_sum + values[i] < L:
                curr_sum += values[i]
                curr_set.add(i)
        return curr_set, curr_sum

    def local_search(initial_set, initial_sum, time_limit):
        curr_set = set(initial_set)
        curr_sum = initial_sum
        best_local_set = set(curr_set)
        best_local_sum = curr_sum
        
        # Precompute lists for efficiency
        all_indices = list(range(n))
        
        while time.time() < time_limit:
            improved = False
            
            # Generate neighbors
            # 1. Add
            best_move = None
            best_move_sum = curr_sum
            
            # Try adding
            for i in all_indices:
                if i not in curr_set:
                    new_sum = curr_sum + values[i]
                    if new_sum < L and new_sum > best_move_sum:
                        best_move_sum = new_sum
                        best_move = ('add', i, None)
                        
            # Try removing
            for i in curr_set:
                new_sum = curr_sum - values[i]
                # Removing always keeps it < L, but we want to maximize sum.
                # So removal is only useful if it allows a better addition later?
                # In a single-step hill climb, we only take a move if it increases the sum.
                # So removals are never taken in a "maximize sum" hill climb unless we allow 
                # non-improving moves (Simulated Annealing).
                # However, a swap can be seen as remove + add.
                # So we just need to check swaps and adds.
                # Actually, if we remove a large item, we might be able to add two smaller ones.
                # But standard hill climb is 1-move.
                # Let's stick to:
                # - Add if improves
                # - Swap if improves
                
            # Try swapping
            for i in curr_set:
                for j in all_indices:
                    if j not in curr_set:
                        new_sum = curr_sum - values[i] + values[j]
                        if new_sum < L and new_sum > best_move_sum:
                            best_move_sum = new_sum
                            best_move = ('swap', i, j)
            
            if best_move and best_move_sum > curr_sum:
                op, i, j = best_move
                if op == 'add':
                    curr_set.add(i)
                elif op == 'swap':
                    curr_set.remove(i)
                    curr_set.add(j)
                
                curr_sum = best_move_sum
                if curr_sum > best_local_sum:
                    best_local_sum = curr_sum
                    best_local_set = set(curr_set)
                improved = True
            
            if not improved:
                # Stuck in local optimum. Do a random kick.
                # Change 1 or 2 bits randomly
                if curr_set:
                    # Remove a random element
                    i = random.choice(list(curr_set))
                    curr_set.remove(i)
                    curr_sum -= values[i]
                # Add a random element
                available = [i for i in all_indices if i not in curr_set]
                if available:
                    j = random.choice(available)
                    if curr_sum + values[j] < L:
                        curr_set.add(j)
                        curr_sum += values[j]
                    else:
                        # If it exceeds, maybe don't add, or add and accept violation?
                        # We must maintain strict < L.
                        # So if it exceeds, we just leave it out.
                        pass
                # If we violated constraint, revert? 
                # The kick is meant to escape, so we allow temporary "bad" states?
                # No, we must always return a valid solution.
                # So the kick should result in a valid state.
                # If the kick results in invalid, we just revert the kick or try another.
                # Simplest: The kick is just a random perturbation of the set, 
                # and we accept it if valid, else try another kick.
                # To keep it simple, we only kick if the new state is valid.
                # If not, we just retry the kick.
                
                # To avoid infinite loop on bad kicks, limit kick attempts?
                # Or just accept that we might stay in the same local optimum.
                # Let's try a few kicks.
                kicks = 0
                while kicks < 10:
                    new_set = set(curr_set)
                    # Randomly flip 2 bits
                    if len(new_set) >= 1:
                        i = random.choice(list(new_set))
                        new_set.remove(i)
                    if len(new_set) < n:
                        j = random.choice([x for x in all_indices if x not in new_set])
                        new_set.add(j)
                    
                    new_sum = get_sum(new_set)
                    if new_sum < L:
                        curr_set = new_set
                        curr_sum = new_sum
                        break
                    kicks += 1
                
                if kicks == 10:
                    break # Give up on this local search instance

        return best_local_set, best_local_sum

    # Run multiple restarts
    restarts = 20
    for r in range(restarts):
        if time.time() > deadline:
            break
        
        # Different orders for greedy init
        if r == 0:
            order = sorted_indices
        elif r == 1:
            order = list(reversed(sorted_indices))
        else:
            order = list(range(n))
            random.shuffle(order)
            
        init_set, init_sum = greedy_init(order)
        
        # Run local search with a time slice
        # Give each restart ~ 2 seconds
        local_deadline = time.time() + 2.0
        if time.time() + 2.0 > deadline:
            local_deadline = deadline - 0.5
            
        res_set, res_sum = local_search(init_set, init_sum, local_deadline)
        
        if res_sum > best_sum:
            best_sum = res_sum
            best_solution = res_set
            
    if best_solution is None:
        # Fallback to seed-like solution
        selected, total = [], 0
        for i, d in enumerate(denominators):
            if total + values[i] < L:
                selected.append(i)
                total += values[i]
        best_solution = set(selected)
        
    return list(best_solution)

# EVOLVE-BLOCK-END
