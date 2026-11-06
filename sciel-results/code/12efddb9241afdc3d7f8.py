import math
import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    limit = 140.0  # Safety margin
    
    N = 512
    # Precompute prime factors for each number 1..511
    # factors[i] is a set of prime factors of i
    factors = [set() for _ in range(N)]
    for i in range(2, N):
        if not factors[i]:
            for j in range(i, N, i):
                factors[j].add(i)
    
    def is_coprime(a, b):
        # Check if a and b share any prime factor
        if not factors[a] or not factors[b]:
            return True
        # Small optimization: if min size is small, iterate
        if len(factors[a]) < len(factors[b]):
            small, large = factors[a], factors[b]
        else:
            small, large = factors[b], factors[a]
        for p in small:
            if p in large:
                return False
        return True

    def get_score(chosen):
        s = Fraction(0, 1)
        for a in chosen:
            s += Fraction(1, N - a)
        return s

    # Initial solution: Greedy
    chosen = []
    for a in range(N - 1, 0, -1):
        if all(is_coprime(a, b) for b in chosen):
            chosen.append(a)
    
    current_score = get_score(chosen)
    best_score = current_score
    best_set = set(chosen)
    
    # Local Search with Simulated Annealing / Hill Climbing
    # We will use a hybrid: try to find best 1-opt swap, if none, try 2-opt, if none, random move.
    
    # To speed up, maintain a set of chosen for O(1) lookup
    chosen_set = set(chosen)
    
    # Precompute weights
    weights = {a: Fraction(1, N - a) for a in range(1, N)}
    
    def try_swap_1():
        nonlocal chosen, chosen_set, current_score
        best_delta = Fraction(0, 1)
        best_a = -1
        best_b = -1
        
        # Iterate over elements in chosen
        for a in chosen:
            # Iterate over candidates not in chosen
            # We can iterate 1..N-1 and skip if in chosen
            for b in range(1, N):
                if b in chosen_set:
                    continue
                # Check if b is coprime with all in chosen except a
                # Optimization: if b shares factor with a, it's fine, we are removing a.
                # But b must be coprime with (chosen - {a})
                valid = True
                for c in chosen:
                    if c == a:
                        continue
                    if not is_coprime(b, c):
                        valid = False
                        break
                if not valid:
                    continue
                
                delta = weights[b] - weights[a]
                if delta > best_delta:
                    best_delta = delta
                    best_a = a
                    best_b = b
        
        if best_a != -1 and best_delta > 0:
            chosen.remove(best_a)
            chosen_set.remove(best_a)
            chosen.append(best_b)
            chosen_set.add(best_b)
            current_score += best_delta
            return True
        return False

    def try_swap_2():
        # Try swapping 2 elements out and 2 elements in
        # This is O(|S|^2 * (N-|S|)^2) which is too slow for full search.
        # We can restrict to "promising" candidates.
        # Or skip 2-opt if 1-opt is fast enough.
        # Given the finding mentioned 2-opt, let's implement a limited 2-opt.
        # Pick a random pair from chosen, and random pair from candidates.
        # This is a stochastic 2-opt.
        pass

    # Main Loop
    # We will alternate between Best-Improvement 1-opt and Randomized moves to escape local optima.
    # However, Best-Improvement 1-opt is deterministic and might get stuck.
    # Let's use a "First Improvement" with random order, or Simulated Annealing.
    
    # Simulated Annealing
    T = 0.05  # Initial temperature (in terms of score delta)
    cooling = 0.999
    min_T = 0.0001
    
    # Convert fractions to float for SA decision?
    # No, we want exactness. But SA probability needs float.
    # We can use float for the *decision* of acceptance of worse moves, 
    # but keep exact fraction for tracking the best score.
    
    # Actually, the "1.041" result likely came from a simple hill climb that found a better local optimum 
    # than the greedy one. The greedy one is very specific.
    # Let's try multiple random restarts of the greedy algorithm with different orderings,
    # then hill climb each.
    
    best_overall_score = best_score
    best_overall_set = best_set
    
    # Helper to hill climb a specific set
    def hill_climb(start_set):
        nonlocal best_overall_score, best_overall_set
        s = list(start_set)
        s_set = set(s)
        score = get_score(s)
        
        improved = True
        while improved:
            improved = False
            # Find best 1-opt swap
            best_delta = Fraction(0, 1)
            best_a = -1
            best_b = -1
            
            for a in s:
                for b in range(1, N):
                    if b in s_set:
                        continue
                    valid = True
                    for c in s:
                        if c == a:
                            continue
                        if not is_coprime(b, c):
                            valid = False
                            break
                    if not valid:
                        continue
                    delta = weights[b] - weights[a]
                    if delta > best_delta:
                        best_delta = delta
                        best_a = a
                        best_b = b
            
            if best_a != -1 and best_delta > 0:
                s.remove(best_a)
                s_set.remove(best_a)
                s.append(best_b)
                s_set.add(best_b)
                score += best_delta
                improved = True
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_set = set(s)
                    
        return s, score

    # Run hill climb on the initial greedy solution
    hill_climb(chosen)
    
    # Time check
    if time.time() - start_time > limit - 10:
        return list(best_overall_set)

    # Try randomized restarts
    # Randomize the order of greedy selection
    for _ in range(20):
        if time.time() - start_time > limit - 5:
            break
            
        # Random Greedy
        r_chosen = []
        candidates = list(range(1, N))
        random.shuffle(candidates)
        # Bias towards larger a (smaller denominator)
        # We can sort by weight descending, but with some randomness
        # Let's just shuffle and pick
        # Or, sort by -weight + noise
        weights_list = [weights[a] for a in candidates]
        # Add small noise to break ties and allow different orders
        noise = [random.uniform(0, 0.01) for _ in range(len(candidates))]
        # Combine: we want to pick high weight first.
        # Let's just shuffle the sorted list slightly?
        # Simpler: Randomly permute, then greedily pick.
        # This might yield worse initial solutions.
        
        # Better: Use the standard greedy but with a random tie-breaker or slight randomization of the "desirability".
        # Desirability = 1/(512-a) + noise
        desirabilities = []
        for a in range(1, N):
            desirabilities.append((weights[a] + random.uniform(0, 0.05), a))
        desirabilities.sort(reverse=True)
        
        r_chosen = []
        for _, a in desirabilities:
            if all(is_coprime(a, b) for b in r_chosen):
                r_chosen.append(a)
        
        # Hill climb this solution
        hill_climb(r_chosen)

    return list(best_overall_set)
