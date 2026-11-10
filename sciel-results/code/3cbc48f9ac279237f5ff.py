import random
import time
import math
from itertools import combinations

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 45.0
    
    n = 22
    elements = list(range(1, n + 1))
    
    # Calculate LCM of 1..22
    L = 1
    for i in elements:
        L = _lcm(L, i)
        
    weights = [L // i for i in elements]
    
    # Best known set (indices)
    best_set = [0, 1, 2, 4, 6, 10, 12, 16, 18] # Corresponds to [1,2,3,5,7,11,13,17,19]
    best_size = len(best_set)
    
    # Helper to check if a set of indices is dissociated
    def is_dissociated(indices):
        sums = {0}
        for idx in indices:
            w = weights[idx]
            new_sums = set()
            for s in sums:
                ns = s + w
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums = new_sums
        return True

    # Backtracking search for target size
    def search_target_size(target_size):
        # We want to find a set of size target_size
        # Use DFS
        found = [None]
        
        # Sort indices by weight? 
        # Larger weights might be "harder" to fit? 
        # Or smaller weights?
        # Let's try original order first.
        
        def dfs(start_idx, current_indices, current_sums, size):
            if size == target_size:
                found[0] = current_indices[:]
                return True
                
            if size + (n - start_idx) < target_size:
                return False
                
            for i in range(start_idx, n):
                w = weights[i]
                # Check collision
                added = []
                collision = False
                for s in current_sums:
                    ns = s + w
                    if ns in current_sums:
                        collision = True
                        break
                    added.append(ns)
                    
                if not collision:
                    new_sums = current_sums.union(added)
                    current_indices.append(i)
                    if dfs(i + 1, current_indices, new_sums, size + 1):
                        return True
                    current_indices.pop()
                    
            return False
            
        dfs(0, [], {0}, 0)
        return found[0]

    # Try to find larger sets
    # Start from best_size + 1
    # We know 16 is possible. Try 17.
    
    # First, quickly verify best_size is valid and maybe update with random greedy using ints
    # Random greedy with ints is very fast.
    
    def try_greedy_random():
        indices = list(range(n))
        random.shuffle(indices)
        current_set = []
        current_sums = {0}
        for i in indices:
            w = weights[i]
            added = []
            collision = False
            for s in current_sums:
                ns = s + w
                if ns in current_sums:
                    collision = True
                    break
                added.append(ns)
            if not collision:
                current_set.append(i)
                current_sums = current_sums.union(added)
        return current_set

    # Run many random greeds
    attempts = 0
    while time.time() < deadline - 10.0: # Leave 10s for backtracking
        attempts += 1
        s = try_greedy_random()
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # Now try backtracking for best_size + 1
    # If we found 16, try 17.
    # If we found 17, try 18.
    
    target = best_size + 1
    while target <= n and time.time() < deadline:
        result = search_target_size(target)
        if result:
            best_set = result
            best_size = target
            target += 1
        else:
            break
            
    # Convert indices back to values
    return [elements[i] for i in best_set]

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 45.0
    
    n = 22
    elements = list(range(1, n + 1))
    
    # Calculate LCM of 1..22
    L = 1
    for i in elements:
        L = _lcm(L, i)
        
    weights = [L // i for i in elements]
    
    # Best known set (indices)
    # [1, 2, 3, 5, 7, 11, 13, 17, 19] -> indices 0, 1, 2, 4, 6, 10, 12, 16, 18
    best_set = [0, 1, 2, 4, 6, 10, 12, 16, 18]
    best_size = len(best_set)
    
    # Helper to check if a set of indices is dissociated
    def is_dissociated(indices):
        sums = {0}
        for idx in indices:
            w = weights[idx]
            new_sums = set()
            for s in sums:
                ns = s + w
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums = new_sums
        return True

    # Backtracking search for target size
    def search_target_size(target_size):
        found = [None]
        
        def dfs(start_idx, current_indices, current_sums, size):
            if size == target_size:
                found[0] = current_indices[:]
                return True
                
            if size + (n - start_idx) < target_size:
                return False
                
            for i in range(start_idx, n):
                w = weights[i]
                added = []
                collision = False
                for s in current_sums:
                    ns = s + w
                    if ns in current_sums:
                        collision = True
                        break
                    added.append(ns)
                    
                if not collision:
                    new_sums = current_sums.union(added)
                    current_indices.append(i)
                    if dfs(i + 1, current_indices, new_sums, size + 1):
                        return True
                    current_indices.pop()
                    
            return False
            
        dfs(0, [], {0}, 0)
        return found[0]

    def try_greedy_random():
        indices = list(range(n))
        random.shuffle(indices)
        current_set = []
        current_sums = {0}
        for i in indices:
            w = weights[i]
            added = []
            collision = False
            for s in current_sums:
                ns = s + w
                if ns in current_sums:
                    collision = True
                    break
                added.append(ns)
            if not collision:
                current_set.append(i)
                current_sums = current_sums.union(added)
        return current_set

    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        s = try_greedy_random()
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    target = best_size + 1
    while target <= n and time.time() < deadline:
        result = search_target_size(target)
        if result:
            best_set = result
            best_size = target
            target += 1
        else:
            break
            
    return [elements[i] for i in best_set]
# EVOLVE-BLOCK-END
