"""Simulated annealing with pair-swaps for reciprocal weight of 3-progression-free sets."""
import time
import random
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def _can_add(x, S_set, S_list):
    """Check if x can be added to S without creating a 3-AP. O(K) time."""
    for a in S_list:
        if a == x:
            continue
        # Case 1: x is the middle term. a, x, c. c = 2x - a.
        c = 2 * x - a
        if 1 <= c <= 256 and c != x and c in S_set:
            return False
        # Case 2: x is an endpoint. a, b, x or x, b, a. b = (a+x)/2.
        s_sum = a + x
        if s_sum % 2 == 0:
            b = s_sum // 2
            if b != x and b in S_set:
                return False
    return True

def _recip_sum(selected):
    """Compute exact sum of reciprocals."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def _recip_sum_float(selected):
    """Compute float sum of reciprocals for quick comparison."""
    return sum(1.0 / x for x in selected)

def _greedy_by_reciprocal(n=256):
    """Greedy: add elements in order of decreasing 1/x (smallest x first)."""
    selected = []
    selected_set = set()
    for x in range(1, n + 1):
        if _can_add(x, selected_set, selected):
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_search_single_swap(start_set, n=256, max_iters=5000, seed=0):
    """Quick local search with single swaps to polish the solution."""
    rng = random.Random(seed)
    current = list(start_set)
    current_set = set(current)
    current_val = _recip_sum_float(current)
    
    improved = True
    while improved and max_iters > 0:
        improved = False
        max_iters -= 1
        # Try removing each element and adding the best candidate
        for i in range(len(current)):
            removed = current[i]
            temp_set = current_set - {removed}
            temp_list = [x for x in current if x != removed]
            
            best_add = None
            best_gain = 0.0
            for x in range(1, n + 1):
                if x in temp_set:
                    continue
                if _can_add(x, temp_set, temp_list):
                    gain = 1.0 / x - 1.0 / removed
                    if gain > best_gain:
                        best_gain = gain
                        best_add = x
            
            if best_add is not None:
                new_list = temp_list + [best_add]
                new_val = current_val + best_gain
                if new_val > current_val + 1e-12:
                    current = new_list
                    current_set = temp_set | {best_add}
                    current_val = new_val
                    improved = True
                    break
    
    return current

def _simulated_annealing(start_set, n=256, t_start=1.0, t_end=0.01, max_iters=200000, seed=0):
    """Simulated annealing with pair-swap neighborhood."""
    rng = random.Random(seed)
    current = list(start_set)
    current_set = set(current)
    current_val = _recip_sum_float(current)
    
    best = list(current)
    best_val = current_val
    
    # Precompute all possible pairs for efficiency? No, just sample.
    
    for it in range(max_iters):
        T = t_start * (t_end / t_start) ** (it / max_iters)
        
        # Generate a move: remove two random elements, add two random candidates
        if len(current) < 2:
            # If set is too small, do single add
            candidates = [x for x in range(1, n + 1) if x not in current_set]
            if not candidates:
                continue
            add = rng.choice(candidates)
            if _can_add(add, current_set, current):
                new_list = current + [add]
                new_val = current_val + 1.0 / add
                if new_val > best_val:
                    best_val = new_val
                    best = new_list
                current = new_list
                current_set.add(add)
                current_val = new_val
            continue
        
        # Remove two distinct elements
        idx1 = rng.randrange(len(current))
        idx2 = rng.randrange(len(current))
        while idx2 == idx1:
            idx2 = rng.randrange(len(current))
        
        removed1 = current[idx1]
        removed2 = current[idx2]
        
        temp_set = current_set - {removed1, removed2}
        temp_list = [x for x in current if x != removed1 and x != removed2]
        
        # Find two candidates to add
        candidates = [x for x in range(1, n + 1) if x not in temp_set]
        if len(candidates) < 2:
            continue
        
        # Weighted random choice for two candidates (prefer small x)
        # Simple approach: pick two random from candidates, but bias towards small
        # Let's just pick two random, but with a bias by sorting and picking from top half + random
        # Or simply: pick two random distinct
        c1_idx = rng.randrange(len(candidates))
        c2_idx = rng.randrange(len(candidates))
        while c2_idx == c1_idx:
            c2_idx = rng.randrange(len(candidates))
        
        add1 = candidates[c1_idx]
        add2 = candidates[c2_idx]
        
        # Check validity
        if _can_add(add1, temp_set, temp_list) and _can_add(add2, temp_set | {add1}, temp_list + [add1]):
            new_list = temp_list + [add1, add2]
            new_val = current_val - 1.0/removed1 - 1.0/removed2 + 1.0/add1 + 1.0/add2
            delta = new_val - current_val
            
            if delta > 0 or rng.random() < math.exp(delta / T):
                current = new_list
                current_set = temp_set | {add1, add2}
                current_val = new_val
                if new_val > best_val:
                    best_val = new_val
                    best = list(current)
        
        # Occasionally do a single swap to allow size changes
        if rng.random() < 0.1:
            remove_idx = rng.randrange(len(current))
            removed = current[remove_idx]
            temp_set = current_set - {removed}
            temp_list = [x for x in current if x != removed]
            
            candidates = [x for x in range(1, n + 1) if x not in temp_set]
            if candidates:
                add = rng.choice(candidates)
                if _can_add(add, temp_set, temp_list):
                    new_list = temp_list + [add]
                    new_val = current_val - 1.0/removed + 1.0/add
                    delta = new_val - current_val
                    if delta > 0 or rng.random() < math.exp(delta / T):
                        current = new_list
                        current_set = temp_set | {add}
                        current_val = new_val
                        if new_val > best_val:
                            best_val = new_val
                            best = list(current)
        
        # Occasionally try adding just one element (grow set)
        if rng.random() < 0.05:
            candidates = [x for x in range(1, n + 1) if x not in current_set]
            if candidates:
                add = rng.choice(candidates)
                if _can_add(add, current_set, current):
                    new_list = current + [add]
                    new_val = current_val + 1.0/add
                    delta = new_val - current_val
                    if delta > 0 or rng.random() < math.exp(delta / T):
                        current = new_list
                        current_set.add(add)
                        current_val = new_val
                        if new_val > best_val:
                            best_val = new_val
                            best = list(current)
        
        # Occasionally try removing just one element (shrink set)
        if rng.random() < 0.05 and len(current) > 1:
            remove_idx = rng.randrange(len(current))
            removed = current[remove_idx]
            new_list = [x for x in current if x != removed]
            new_val = current_val - 1.0/removed
            delta = new_val - current_val
            if delta > 0 or rng.random() < math.exp(delta / T):
                current = new_list
                current_set.remove(removed)
                current_val = new_val
                if new_val > best_val:
                    best_val = new_val
                    best = list(current)
    
    return best

def construct():
    """Main entry point: find a 3-free subset of [1,256] maximizing sum of reciprocals."""
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit
    
    # Start with greedy by reciprocal
    best_set = _greedy_by_reciprocal(256)
    best_val = _recip_sum_float(best_set)
    
    # Polish with local search
    polished = _local_search_single_swap(best_set, 256, max_iters=1000, seed=0)
    polished_val = _recip_sum_float(polished)
    if polished_val > best_val:
        best_set = polished
        best_val = polished_val
    
    # Run simulated annealing from multiple starts
    seed = 0
    while time.time() - start_time < deadline - 5:
        # Start SA from current best
        candidate = _simulated_annealing(best_set, 256, t_start=1.0, t_end=0.01, max_iters=50000, seed=seed)
        if candidate:
            val = _recip_sum_float(candidate)
            if val > best_val:
                best_val = val
                best_set = candidate
        seed += 1
        
        # Also try SA from a fresh greedy start occasionally
        if seed % 5 == 0:
            fresh_greedy = _greedy_by_reciprocal(256)
            fresh_polished = _local_search_single_swap(fresh_greedy, 256, max_iters=500, seed=seed)
            candidate = _simulated_annealing(fresh_polished, 256, t_start=1.0, t_end=0.01, max_iters=30000, seed=seed + 1000)
            if candidate:
                val = _recip_sum_float(candidate)
                if val > best_val:
                    best_val = val
                    best_set = candidate
    
    # Final polish with local search
    final_polished = _local_search_single_swap(best_set, 256, max_iters=2000, seed=9999)
    final_val = _recip_sum_float(final_polished)
    if final_val > best_val:
        best_set = final_polished
        best_val = final_val
    
    return best_set

# EVOLVE-BLOCK-END
