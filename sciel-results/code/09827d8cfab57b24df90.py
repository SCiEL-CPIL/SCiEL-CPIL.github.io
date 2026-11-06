import time
import math
import bisect
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    N = len(denominators)
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    L = 1
    for d in denominators:
        L = (L * d) // gcd(L, d)
        
    weights = [L // d for d in denominators]
    
    start_time = time.time()
    time_limit = 80.0  # Safety margin before 90s kill
    
    # Greedy baseline: sort by weight descending (smaller denominator = larger weight)
    order = sorted(range(N), key=lambda i: -weights[i])
    
    greedy_set = set()
    greedy_sum = 0
    for i in order:
        if greedy_sum + weights[i] < L:
            greedy_set.add(i)
            greedy_sum += weights[i]
    
    best_sum = greedy_sum
    best_set = set(greedy_set)
    
    def evaluate_set(s):
        """Compute sum of weights for a set of indices."""
        return sum(weights[i] for i in s)
    
    def local_search(start_set, start_sum, max_iters=200000):
        """Local search: try adding, removing, swapping elements."""
        current_set = set(start_set)
        current_sum = start_sum
        best_local_sum = current_sum
        best_local_set = set(current_set)
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            improved = False
            
            # Try adding an element
            for i in range(N):
                if i not in current_set:
                    new_sum = current_sum + weights[i]
                    if new_sum < L and new_sum > best_local_sum:
                        current_set.add(i)
                        current_sum = new_sum
                        best_local_sum = new_sum
                        best_local_set = set(current_set)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try removing an element
            for i in list(current_set):
                new_sum = current_sum - weights[i]
                if new_sum < L:  # always true since we remove
                    # Check if removing allows a better add
                    # Just track if we can improve
                    if new_sum > best_local_sum:
                        current_set.discard(i)
                        current_sum = new_sum
                        best_local_sum = new_sum
                        best_local_set = set(current_set)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swapping: remove one, add one
            for i in list(current_set):
                if time.time() - start_time > time_limit:
                    break
                for j in range(N):
                    if j not in current_set:
                        new_sum = current_sum - weights[i] + weights[j]
                        if new_sum < L and new_sum > best_local_sum:
                            current_set.discard(i)
                            current_set.add(j)
                            current_sum = new_sum
                            best_local_sum = new_sum
                            best_local_set = set(current_set)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_local_sum, best_local_set
    
    # Run local search from greedy
    best_sum, best_set = local_search(best_set, best_sum)
    
    # Try exhaustive over top-k items by weight
    # Top 12 items: 2^12 = 4096 subsets, for each do greedy fill on remaining
    top_k = 12
    top_items = order[:top_k]
    remaining_items = order[top_k:]
    
    # Pre-sort remaining by weight descending for greedy fill
    remaining_sorted = sorted(remaining_items, key=lambda i: -weights[i])
    
    def greedy_fill(available, target):
        """Greedy fill from available items to get sum as close to target as possible, < L."""
        s = 0
        chosen = set()
        for i in available:
            if s + weights[i] < L and s + weights[i] <= target:
                s += weights[i]
                chosen.add(i)
        return s, chosen
    
    # Enumerate all subsets of top_k
    for mask in range(1 << top_k):
        if time.time() - start_time > time_limit:
            break
        
        top_set = set()
        top_sum = 0
        for bit in range(top_k):
            if mask & (1 << bit):
                i = top_items[bit]
                top_set.add(i)
                top_sum += weights[i]
        
        if top_sum >= L:
            continue
        
        # Greedy fill from remaining
        fill_sum, fill_set = greedy_fill(remaining_sorted, L - 1 - top_sum)
        
        total_sum = top_sum + fill_sum
        if total_sum < L and total_sum > best_sum:
            best_sum = total_sum
            best_set = top_set | fill_set
    
    # Another local search from the new best
    best_sum, best_set = local_search(best_set, best_sum)
    
    # Try a second pass with different top_k
    top_k2 = 10
    top_items2 = order[:top_k2]
    remaining_items2 = order[top_k2:]
    remaining_sorted2 = sorted(remaining_items2, key=lambda i: -weights[i])
    
    for mask in range(1 << top_k2):
        if time.time() - start_time > time_limit:
            break
        
        top_set = set()
        top_sum = 0
        for bit in range(top_k2):
            if mask & (1 << bit):
                i = top_items2[bit]
                top_set.add(i)
                top_sum += weights[i]
        
        if top_sum >= L:
            continue
        
        fill_sum, fill_set = greedy_fill(remaining_sorted2, L - 1 - top_sum)
        
        total_sum = top_sum + fill_sum
        if total_sum < L and total_sum > best_sum:
            best_sum = total_sum
            best_set = top_set | fill_set
    
    # Final local search
    best_sum, best_set = local_search(best_set, best_sum)
    
    return list(best_set)
