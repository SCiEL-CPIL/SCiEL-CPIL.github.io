import time
import random
import numpy as np
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = range(min_sum, max_sum + 1)

    def compute_counts(A_set):
        """Compute unordered pair-sum multiplicities for all sums in [0, 256]."""
        f = np.zeros(max_sum + 1, dtype=float)
        for x in A_set:
            f[x] = 1.0
        n_fft = 1
        while n_fft < 2 * (max_sum + 1):
            n_fft *= 2
        F = np.fft.rfft(f, n=n_fft)
        conv = np.fft.irfft(F * F, n=n_fft)
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        for s in range(min_sum, max_sum + 1):
            ordered = int(round(conv[s]))
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= N and half in A_set:
                    counts[s] = (ordered + 1) // 2
                else:
                    counts[s] = ordered // 2
            else:
                counts[s] = ordered // 2
        return counts

    def get_cost(A_set):
        """Return peak multiplicity, or large value if coverage incomplete."""
        counts = compute_counts(A_set)
        target_counts = counts[min_sum:max_sum + 1]
        if np.any(target_counts == 0):
            return 10000 + int(np.sum(target_counts == 0))
        return int(np.max(target_counts))

    def local_search(A_set, time_limit):
        """Fast local search to minimize peak multiplicity."""
        current_cost = get_cost(A_set)
        best_cost = current_cost
        best_A = set(A_set)
        all_elements = set(range(N + 1))
        
        while time.time() < time_limit:
            counts = compute_counts(A_set)
            target_counts = counts[min_sum:max_sum + 1]
            
            if np.any(target_counts == 0):
                # Fix coverage first
                uncovered = set(s for s in target_sums if target_counts[s - min_sum] == 0)
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
            
            peak = int(np.max(target_counts))
            if peak <= 2:
                break
            
            # Try swaps to reduce peak
            improved = False
            A_sorted = sorted(A_set)
            
            for x_in in A_sorted:
                if time.time() >= time_limit:
                    break
                candidates = list(all_elements - A_set)
                random.shuffle(candidates)
                for x_out in candidates[:20]:
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
                # Random swap
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

    random.seed(42)
    best_overall_A = None
    best_overall_cost = float('inf')

    # Phase 1: Differential Evolution for global exploration
    # Represent A as binary vector of length N+1=129
    # DE works on continuous, so we'll use rounding
    
    def de_objective(x_continuous):
        """Objective for differential evolution: minimize peak multiplicity."""
        # Round to get binary vector
        x_bin = (x_continuous > 0.5).astype(int)
        A_set = set(i for i in range(N + 1) if x_bin[i] == 1)
        if len(A_set) == 0:
            return 100000
        return get_cost(A_set)

    # Run DE with limited time
    de_deadline = start_time + 25.0
    try:
        # Bounds: each variable in [0, 1], threshold at 0.5
        bounds = [(0, 1)] * (N + 1)
        
        # DE callback to check time
        def de_callback(xk, convergence):
            return time.time() > de_deadline
        
        result = differential_evolution(
            de_objective,
            bounds,
            maxiter=100,
            popsize=15,
            tol=0.01,
            mutation=(0.4, 1.0),
            recombination=0.8,
            seed=42,
            callback=de_callback,
            polish=False
        )
        
        if result.x is not None:
            x_bin = (result.x > 0.5).astype(int)
            A_de = set(i for i in range(N + 1) if x_bin[i] == 1)
            if len(A_de) > 0:
                cost_de = get_cost(A_de)
                if cost_de < best_overall_cost:
                    best_overall_cost = cost_de
                    best_overall_A = A_de
    except Exception:
        pass

    # Phase 2: Local search refinement from DE result and random starts
    if best_overall_A is not None and time.time() < deadline - 10:
        best_overall_A, best_overall_cost = local_search(best_overall_A, min(deadline - 5, time.time() + 15))

    # Phase 3: Multiple random restarts with local search
    for k in [16, 18, 20, 22, 24, 26]:
        if time.time() >= deadline - 8:
            break
        A_init = set(random.sample(range(0, N + 1), k))
        A_refined, cost = local_search(A_init, min(deadline - 5, time.time() + 8))
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_A = A_refined
        if best_overall_cost <= 2:
            break

    # Final polish
    if best_overall_A is not None and time.time() < deadline - 2:
        best_overall_A, best_overall_cost = local_search(best_overall_A, deadline - 2)

    if best_overall_A is None:
        best_overall_A = set(random.sample(range(0, N + 1), 20))

    return list(best_overall_A)
