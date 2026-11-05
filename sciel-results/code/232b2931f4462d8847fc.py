import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = range(min_sum, max_sum + 1)
    
    # Fast count computation using indicator vector and convolution
    # For a set A, the number of unordered pairs (a,b) with a<=b in A such that a+b=s
    # is: for s even, floor(count of a in A with a=s/2) contributes 1 if s/2 in A
    # Actually, let's use the standard approach:
    # Let f(x) = 1 if x in A else 0
    # The number of ordered pairs (a,b) in A×A with a+b=s is (f*f)(s)
    # For unordered pairs with a<=b: 
    #   if s is even: (f*f)(s) - f(s/2) + 1 if f(s/2)=1... 
    #   Actually simpler: unordered count = ((f*f)(s) + f(s//2 if s even else 0)) / 2 ... no
    # Let me think more carefully.
    # Ordered pairs: sum_{a in A} sum_{b in A} [a+b=s] = (f*f)(s)
    # Unordered pairs with a<=b: for each a<=b in A with a+b=s, count 1
    # This equals: (number of ordered pairs + number of pairs with a=b) / 2
    # Number of ordered pairs = (f*f)(s)
    # Number of pairs with a=b: 1 if s is even and s/2 in A, else 0
    # So unordered count = ((f*f)(s) + [s even and s/2 in A]) / 2
    
    def get_counts_fast(A):
        """Fast count using FFT convolution"""
        f = np.zeros(max_sum + 1, dtype=float)
        for x in A:
            f[x] = 1.0
        # Convolve f with itself
        # Use FFT for speed
        n_fft = 1
        while n_fft < 2 * (max_sum + 1):
            n_fft *= 2
        F = np.fft.rfft(f, n=n_fft)
        conv = np.fft.irfft(F * F, n=n_fft)
        # conv[s] = number of ordered pairs (a,b) in A×A with a+b=s
        counts = np.zeros(max_sum + 1, dtype=int)
        for s in range(max_sum + 1):
            ordered = int(round(conv[s]))
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= N and half in A:
                    unordered = (ordered + 1) // 2
                else:
                    unordered = ordered // 2
            else:
                unordered = ordered // 2
            counts[s] = unordered
        return counts

    def get_cost(A):
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum+1]
        if np.any(target_counts == 0):
            return 1000 + int(np.sum(target_counts == 0))
        return int(np.max(target_counts))

    def local_search(A, time_limit):
        """Targeted local search to minimize peak multiplicity"""
        current_cost = get_cost(A)
        best_cost = current_cost
        best_A = set(A)
        A_list = sorted(A)
        A_set = set(A)
        
        # Precompute all candidate elements outside A
        all_elements = set(range(N + 1))
        
        while time.time() < time_limit:
            counts = get_counts_fast(A_set)
            target_counts = counts[min_sum:max_sum+1]
            
            if np.any(target_counts == 0):
                # Need to fix coverage first
                uncovered = set(s for s in target_sums if target_counts[s - min_sum] == 0)
                # Try adding elements that cover uncovered sums
                best_x = None
                best_gain = -1
                for x in all_elements - A_set:
                    gain = 0
                    for y in A_set:
                        s = x + y
                        if min_sum <= s <= max_sum and s in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is not None and best_gain > 0:
                    A_set.add(best_x)
                    current_cost = get_cost(A_set)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A_set)
                continue
            
            # Try targeted swaps: find elements in A that contribute to high-count sums
            # and swap them with elements that would reduce the peak
            peak = int(np.max(target_counts))
            if peak <= 2:
                break
            
            # Find sums with peak count
            peak_sums = [min_sum + i for i in range(len(target_counts)) if target_counts[i] == peak]
            
            # For each element in A, compute its "contribution" to peak sums
            # Try swapping each element with a candidate outside
            improved = False
            A_sorted = sorted(A_set)
            
            for idx, x_in in enumerate(A_sorted):
                if time.time() >= time_limit:
                    break
                # Try removing x_in and adding various x_out
                # Focus on x_out that might reduce peak
                # Heuristic: try x_out that are "far" from elements causing peak
                candidates = list(all_elements - A_set)
                random.shuffle(candidates)
                for x_out in candidates[:30]:  # Limit candidates
                    A_test = A_set.copy()
                    A_test.discard(x_in)
                    A_test.add(x_out)
                    new_cost = get_cost(A_test)
                    if new_cost < current_cost:
                        A_set = A_test
                        current_cost = new_cost
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_A = set(A_set)
                        break
                if improved:
                    break
            
            if not improved:
                # Try random swap
                if len(A_set) > 0:
                    x_in = random.choice(list(A_set))
                    candidates = list(all_elements - A_set)
                    if candidates:
                        x_out = random.choice(candidates)
                        A_test = A_set.copy()
                        A_test.discard(x_in)
                        A_test.add(x_out)
                        new_cost = get_cost(A_test)
                        if new_cost <= current_cost:
                            A_set = A_test
                            current_cost = new_cost
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_A = set(A_set)
        
        return best_A, best_cost

    # Multi-start approach
    random.seed(42)
    best_overall_A = None
    best_overall_cost = float('inf')
    
    # Try different initial set sizes
    for k in [16, 18, 20, 22, 24]:
        if time.time() >= deadline - 10:
            break
        # Random initial set
        A_init = set(random.sample(range(0, N + 1), k))
        
        # Quick local search
        A_refined, cost = local_search(A_init, min(deadline - 5, start_time + 10))
        
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_A = A_refined
        
        if best_overall_cost <= 2:
            break
    
    # Final polish with remaining time
    if best_overall_A is not None and time.time() < deadline - 2:
        best_overall_A, best_overall_cost = local_search(best_overall_A, deadline - 2)
    
    if best_overall_A is None:
        # Fallback
        best_overall_A = set(random.sample(range(0, N + 1), 20))
    
    return list(best_overall_A)
