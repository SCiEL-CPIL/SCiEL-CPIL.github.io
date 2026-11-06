import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
    
    w = [L // i for i in range(1, n + 1)]
    
    # Meet-in-the-middle with coefficient tracking
    mid = n // 2
    w1 = w[:mid]
    w2 = w[mid:]
    
    def generate_sums_with_coeffs(weights):
        """Generate all possible sums and track one coefficient vector for each sum."""
        sums_dict = {0: [0] * len(weights)}
        for i, wi in enumerate(weights):
            new_dict = {}
            for s, coeffs in sums_dict.items():
                for c in [-1, 0, 1]:
                    ns = s + c * wi
                    if ns not in new_dict:
                        new_coeffs = coeffs.copy()
                        new_coeffs[i] = c
                        new_dict[ns] = new_coeffs
            sums_dict = new_dict
        return sums_dict
    
    def local_search(coeffs, w, max_iters=200000):
        """Greedy local search to minimize |sum| while keeping it nonzero."""
        current_sum = sum(c * wi for c, wi in zip(coeffs, w))
        if current_sum == 0:
            # Try flipping a random coefficient to make it nonzero
            for _ in range(1000):
                idx = random.randint(0, len(w) - 1)
                old_c = coeffs[idx]
                coeffs[idx] = (old_c + 1) % 3 - 1
                new_sum = current_sum + (coeffs[idx] - old_c) * w[idx]
                if new_sum != 0:
                    current_sum = new_sum
                    break
                else:
                    coeffs[idx] = old_c
            if current_sum == 0:
                return None
        
        best_sum = current_sum
        best_coeffs = coeffs.copy()
        
        for _ in range(max_iters):
            improved = False
            for i in range(len(w)):
                old_c = coeffs[i]
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    delta = (new_c - old_c) * w[i]
                    new_sum = current_sum + delta
                    if new_sum == 0:
                        continue
                    if abs(new_sum) < abs(best_sum):
                        best_sum = new_sum
                        best_coeffs = coeffs.copy()
                        coeffs[i] = new_c
                        current_sum = new_sum
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                # Try random restarts from current state
                for _ in range(50):
                    idx = random.randint(0, len(w) - 1)
                    old_c = coeffs[idx]
                    new_c = random.choice([-1, 0, 1])
                    if new_c == old_c:
                        continue
                    delta = (new_c - old_c) * w[idx]
                    new_sum = current_sum + delta
                    if new_sum == 0:
                        continue
                    if abs(new_sum) < abs(best_sum):
                        best_sum = new_sum
                        best_coeffs = coeffs.copy()
                        coeffs[i] = new_c
                        current_sum = new_sum
                        improved = True
                        break
                    else:
                        coeffs[idx] = old_c
                if not improved:
                    break
        
        return best_coeffs if abs(best_sum) < abs(current_sum) else best_coeffs
    
    # Generate sums for both halves
    sums1_dict = generate_sums_with_coeffs(w1)
    sums2_dict = generate_sums_with_coeffs(w2)
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # Find best combination
    sums1_sorted = sorted(sums1_dict.keys())
    import bisect
    
    best_val = float('inf')
    best_s1 = 0
    best_s2 = 0
    best_c1 = None
    best_c2 = None
    
    for s2, c2 in sums2_dict.items():
        target = -s2
        idx = bisect.bisect_left(sums1_sorted, target)
        
        candidates = []
        if idx < len(sums1_sorted):
            candidates.append(sums1_sorted[idx])
        if idx > 0:
            candidates.append(sums1_sorted[idx - 1])
        
        for s1 in candidates:
            total = s1 + s2
            if total != 0:
                val = abs(total)
                if val < best_val:
                    best_val = val
                    best_s1 = s1
                    best_s2 = s2
                    best_c1 = sums1_dict[s1]
                    best_c2 = c2
    
    if best_c1 is None:
        return [0] * 28 + [1, -1]
    
    # Combine and do local search
    combined_coeffs = best_c1 + best_c2
    combined_sum = sum(c * wi for c, wi in zip(combined_coeffs, w))
    
    if combined_sum == 0:
        # Try to fix
        for i in range(n):
            old_c = combined_coeffs[i]
            for new_c in [-1, 0, 1]:
                if new_c == old_c:
                    continue
                new_sum = combined_sum + (new_c - old_c) * w[i]
                if new_sum != 0:
                    combined_coeffs[i] = new_c
                    combined_sum = new_sum
                    break
            if combined_sum != 0:
                break
    
    if combined_sum == 0:
        return [0] * 28 + [1, -1]
    
    # Local search to improve
    improved_coeffs = local_search(combined_coeffs.copy(), w, max_iters=100000)
    
    if improved_coeffs is not None:
        improved_sum = sum(c * wi for c, wi in zip(improved_coeffs, w))
        if improved_sum != 0 and abs(improved_sum) < abs(combined_sum):
            combined_coeffs = improved_coeffs
            combined_sum = improved_sum
    
    # Verify
    final_sum = sum(c * wi for c, wi in zip(combined_coeffs, w))
    if final_sum == 0:
        return [0] * 28 + [1, -1]
    
    return combined_coeffs
