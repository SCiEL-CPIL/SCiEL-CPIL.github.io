"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def construct():
    deadline = time.time() + 85  # 90s limit, keep 5s margin
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    weights = [L // (i + 1) for i in range(30)]
    
    def compute_sum(coeffs):
        s = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * weights[i]
        return s
    
    def local_search(coeffs, current_sum):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(30):
                if time.time() > deadline:
                    break
                old_val = coeffs[i]
                for new_val in (-1, 0, 1):
                    if new_val == old_val:
                        continue
                    coeffs[i] = new_val
                    new_sum = current_sum - old_val * weights[i] + new_val * weights[i]
                    if new_sum == 0:
                        coeffs[i] = old_val
                        continue
                    if abs(new_sum) < abs(current_sum):
                        current_sum = new_sum
                        improved = True
                        break
                    else:
                        coeffs[i] = old_val
                if improved:
                    break
        return coeffs, current_sum
    
    # Meet-in-the-middle approach
    # Split indices into two halves of 15
    half = 15
    idx1 = list(range(half))       # 0..14
    idx2 = list(range(half, 30))   # 15..29
    
    # Generate all 3^15 = 14348901 combinations for each half
    # For each combination, store (sum, coeffs_tuple)
    # We want to find pairs (s1, s2) such that s1 + s2 is nonzero and |s1+s2| is minimized
    
    def gen_half_sums(indices):
        """Generate all (sum, coeffs_dict) for a set of indices."""
        n = len(indices)
        results = []
        # Use iterative approach
        # For each of the 3^n combinations, compute the sum
        # We can do this recursively or iteratively
        
        # Iterative: build up the list
        # Start with [(0, {})]
        # For each index, triple the list
        current = [(0, [])]  # (sum, list of (index, value))
        for idx in indices:
            new_current = []
            for s, clist in current:
                for val in (-1, 0, 1):
                    if val == 0:
                        new_current.append((s, clist))
                    else:
                        new_current.append((s + val * weights[idx], clist + [(idx, val)]))
            current = new_current
        return current
    
    # This would use too much memory for 3^15. Let's use a smarter approach.
    # Instead, generate all sums for half 1, sort them, then for each sum in half 2,
    # binary search for the closest sum in half 1 that makes the total nonzero and minimal.
    
    def gen_half_sums_sorted(indices):
        """Generate all sums for a set of indices, return sorted list of (sum, coeffs_list)."""
        n = len(indices)
        # Use a dictionary to avoid storing duplicate (sum, coeffs) pairs? No, we need coeffs.
        # Actually, for meet-in-the-middle, we just need the sums and then reconstruct coeffs.
        # But we need to know which coeffs give which sum.
        
        # Let's just generate all (sum, coeffs_tuple) pairs
        # 3^15 = 14,348,901 which is large but manageable if we're careful
        # Actually, storing 14M tuples might be too much memory. Let's think differently.
        
        # Alternative: generate all sums for half 1, sort them.
        # For each sum in half 2, find the best matching sum in half 1.
        # We don't need to store all coeffs, just the sums, and then reconstruct.
        
        # But we need to reconstruct the coefficients. So we need to store them.
        # Let's try a different approach: generate sums for half 1, store in a dict
        # mapping sum -> list of coeffs (but there could be many).
        
        # Actually, let's just generate all (sum, coeffs) for half 1, sort by sum.
        # Then for each (sum2, coeffs2) in half 2, binary search in the sorted list of half 1.
        
        pass
    
    # Let's use a more memory-efficient approach
    # Generate all sums for half 1, store as sorted list of (sum, coeffs_tuple)
    # coeffs_tuple is a tuple of 15 values
    
    def gen_half(indices):
        """Generate all (sum, coeffs_tuple) for given indices."""
        n = len(indices)
        results = []
        # Build up iteratively
        current = [(0, ())]
        for idx in indices:
            w = weights[idx]
            new_current = []
            for s, ct in current:
                new_current.append((s, ct + (0,)))
                new_current.append((s + w, ct + (1,)))
                new_current.append((s - w, ct + (-1,)))
            current = new_current
        return current
    
    # 3^15 = 14,348,901. This is large but let's see if we can do it.
    # Actually, this might be too slow and memory-intensive. Let's try a smaller split.
    # Split into 10 and 20? 3^10 = 59049, 3^20 is too big.
    # Split into 12 and 18? 3^12 = 531441, 3^18 is too big.
    # Let's stick with 15 and 15, but be efficient.
    
    # Actually, let's try a different approach: use the meet-in-the-middle but with
    # a hash table for one half.
    
    # Generate all (sum, coeffs) for half 1, store in a dict: sum -> list of coeffs
    # Then for each (sum2, coeffs2) in half 2, look up sums close to -sum2 in the dict.
    
    # But the dict could have many entries with the same sum.
    
    # Let's just try the straightforward approach and see if it fits in time.
    
    # Actually, 3^15 = 14,348,901. Generating this list might take a few seconds.
    # Let's try it.
    
    def gen_half_list(indices):
        """Generate all (sum, coeffs_tuple) for given indices."""
        n = len(indices)
        results = [(0, ())]
        for idx in indices:
            w = weights[idx]
            new_results = []
            for s, ct in results:
                new_results.append((s, ct + (0,)))
                new_results.append((s + w, ct + (1,)))
                new_results.append((s - w, ct + (-1,)))
            results = new_results
        return results
    
    # This is O(3^n) in both time and space. For n=15, that's ~14M entries.
    # Each entry is a tuple (int, tuple of 15 ints). This might use a lot of memory.
    # Let's estimate: 14M * (8 + 15*8) bytes = 14M * 128 bytes = 1.8 GB. Too much.
    
    # We need a more efficient approach. Let's use a different strategy.
    
    # Strategy: Generate all sums for half 1, sort them. For each sum in half 2,
    # binary search for the best match. But we need to store the coeffs.
    
    # Alternative: Use a hash table mapping sum -> coeffs for half 1.
    # But multiple coeffs can give the same sum.
    
    # Let's try a different split: 12 and 18.
    # 3^12 = 531,441. This is manageable.
    # 3^18 is too big.
    
    # How about 10 and 20? 3^10 = 59,049. 3^20 is too big.
    
    # Let's use a different approach: iterate over all combinations of half 1 (3^15),
    # and for each, compute the best match in half 2 using a precomputed table.
    
    # Actually, let's just use the local search + simulated annealing approach but
    # with better initialization and more restarts, as in the current program.
    # The meet-in-the-middle might be too expensive.
    
    # Let's go back to the current approach but with improvements.
    
    def greedy_init():
        coeffs = [0] * 30
        current_sum = 0
        indices = sorted(range(30), key=lambda i: weights[i], reverse=True)
        for i in indices:
            best_val = 0
            best_abs = abs(current_sum) if current_sum != 0 else float('inf')
            for val in (-1, 0, 1):
                new_sum = current_sum + val * weights[i]
                if new_sum == 0:
                    continue
                new_abs = abs(new_sum)
                if new_abs < best_abs:
                    best_abs = new_abs
                    best_val = val
            coeffs[i] = best_val
            current_sum += best_val * weights[i]
        if current_sum == 0:
            for i in range(30):
                old = coeffs[i]
                for new in (-1, 0, 1):
                    if new == old:
                        continue
                    coeffs[i] = new
                    new_sum = current_sum - old * weights[i] + new * weights[i]
                    if new_sum != 0:
                        current_sum = new_sum
                        return coeffs, current_sum
                coeffs[i] = old
            coeffs[0] = 1
            current_sum = weights[0]
        return coeffs, current_sum
    
    best_sum = None
    best_coeffs = None
    
    def update_best(coeffs, int_sum):
        nonlocal best_sum, best_coeffs
        if int_sum == 0:
            return
        if best_sum is None or abs(int_sum) < abs(best_sum):
            best_sum = int_sum
            best_coeffs = list(coeffs)
    
    random.seed(42)
    
    # Try meet-in-the-middle with a smaller split
    # Split into 12 and 18: 3^12 = 531441, 3^18 is too big
    # Let's split into 10 and 20: 3^10 = 59049
    # For each combination of the first 10, compute the sum
    # For each combination of the last 20, compute the sum
    # But 3^20 is too big.
    
    # Let's try a different approach: use the fact that we can enumerate 3^15 = 14M
    # but we don't need to store all of them. We can generate them one at a time
    # and use a hash table for the other half.
    
    # Actually, let's just use the simulated annealing approach with more iterations
    # and better cooling schedule.
    
    coeffs, current_sum = greedy_init()
    update_best(coeffs, current_sum)
    coeffs, current_sum = local_search(coeffs, current_sum)
    update_best(coeffs, current_sum)
    
    num_restarts = 100
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initial state
        coeffs = [random.choice([-1, 0, 1]) for _ in range(30)]
        current_sum = compute_sum(coeffs)
        if current_sum == 0:
            idx = random.randint(0, 29)
            coeffs[idx] = random.choice([-1, 1])
            current_sum = compute_sum(coeffs)
        
        update_best(coeffs, current_sum)
        
        # Simulated annealing
        T_start = max(abs(current_sum), 10000)
        T_end = 1
        num_iterations = 5000
        cooling_rate = (T_end / T_start) ** (1.0 / num_iterations) if T_start > 0 else 1.0
        
        T = T_start
        for it in range(num_iterations):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            if move_type < 0.7:
                i = random.randint(0, 29)
                old_val = coeffs[i]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    continue
                coeffs[i] = new_val
                new_sum = current_sum - old_val * weights[i] + new_val * weights[i]
                
                if new_sum == 0:
                    coeffs[i] = old_val
                    T *= cooling_rate
                    continue
                
                old_abs = abs(current_sum)
                new_abs = abs(new_sum)
                delta = new_abs - old_abs
                
                if delta < 0 or (T > 0 and random.random() < (2.718281828 ** (-delta / T))):
                    current_sum = new_sum
                    update_best(coeffs, current_sum)
                else:
                    coeffs[i] = old_val
            else:
                i, j = random.sample(range(30), 2)
                old_i, old_j = coeffs[i], coeffs[j]
                new_i = random.choice([-1, 0, 1])
                new_j = random.choice([-1, 0, 1])
                if new_i == old_i and new_j == old_j:
                    continue
                coeffs[i] = new_i
                coeffs[j] = new_j
                new_sum = current_sum - old_i * weights[i] - old_j * weights[j] + new_i * weights[i] + new_j * weights[j]
                
                if new_sum == 0:
                    coeffs[i] = old_i
                    coeffs[j] = old_j
                    T *= cooling_rate
                    continue
                
                old_abs = abs(current_sum)
                new_abs = abs(new_sum)
                delta = new_abs - old_abs
                
                if delta < 0 or (T > 0 and random.random() < (2.718281828 ** (-delta / T))):
                    current_sum = new_sum
                    update_best(coeffs, current_sum)
                else:
                    coeffs[i] = old_i
                    coeffs[j] = old_j
            
            T *= cooling_rate
        
        coeffs, current_sum = local_search(coeffs, current_sum)
        update_best(coeffs, current_sum)
    
    # Try meet-in-the-middle with a smaller split
    # Split into 10 and 20
    half1 = list(range(10))
    half2 = list(range(10, 30))
    
    # Generate all sums for half1
    def gen_sums(indices):
        """Generate all (sum, coeffs_list) for given indices."""
        results = [(0, [])]
        for idx in indices:
            w = weights[idx]
            new_results = []
            for s, clist in results:
                new_results.append((s, clist + [0]))
                new_results.append((s + w, clist + [1]))
                new_results.append((s - w, clist + [-1]))
            results = new_results
        return results
    
    # 3^10 = 59049, manageable
    sums1 = gen_sums(half1)
    # Sort by sum
    sums1.sort(key=lambda x: x[0])
    
    # For each combination in half2, find the best match in half1
    # But 3^20 is too big. So we can't enumerate all of half2.
    
    # Instead, let's enumerate half1 and for each, try to find a good half2.
    # Or, enumerate half2 with a smaller subset.
    
    # Let's try a different approach: use the meet-in-the-middle with 15 and 15,
    # but generate one half at a time and use a hash table.
    
    # Actually, let's just rely on the simulated annealing approach.
    
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
