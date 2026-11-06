import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin under 150s

    N = 511
    # Weights: w[a] = 1/(512-a)
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (512.0 - a)

    # Precompute coprime sets
    # coprime_sets[a] contains all b in 1..511 such that gcd(a, b) == 1
    coprime_sets = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        s = set()
        for b in range(1, N + 1):
            if math.gcd(a, b) == 1:
                s.add(b)
        coprime_sets[a] = s

    def can_add(a, chosen_set):
        """Check if a is coprime to all elements in chosen_set."""
        # chosen_set is a set of integers
        # Check if chosen_set is a subset of coprime_sets[a]
        return chosen_set.issubset(coprime_sets[a])

    def fill_greedy(chosen_set, score):
        """Try to add all compatible elements greedily."""
        improved = True
        # Sort candidates by weight descending, but we can just iterate and check
        # To be deterministic and fast, we iterate from high weight to low
        # But adding any valid element increases score, so order doesn't matter for *if* we can add,
        # but it matters for *which* one if we have a choice? No, we add ALL compatible ones.
        # So we just need to find all a not in chosen_set such that can_add(a, chosen_set).
        # Since adding one might block another? No, if a and b are both coprime to chosen_set,
        # are they coprime to each other? Not necessarily.
        # So we must add them one by one.
        # Greedy: pick the highest weight one first?
        # Let's iterate from N down to 1.
        for a in range(N, 0, -1):
            if a in chosen_set:
                continue
            if can_add(a, chosen_set):
                chosen_set.add(a)
                score += w[a]
        return chosen_set, score

    # Greedy initialization
    # Start with empty set? Or start with the best element?
    # The current code starts by picking from largest weight to smallest if compatible.
    # Let's do that.
    chosen = set()
    for a in range(N, 0, -1):
        if can_add(a, chosen):
            chosen.add(a)
    
    chosen, score = fill_greedy(chosen, 0.0)
    best_score = score
    best_chosen = set(chosen)

    random.seed(42)

    def local_search(chosen_set, score):
        improved = True
        while improved and (time.time() < deadline - 10):
            improved = False

            # 1. Try adding any compatible element
            # We can just call fill_greedy, but it might be slow if we do it repeatedly.
            # However, fill_greedy adds ALL possible. If we just added one, maybe we can add more.
            # So just calling fill_greedy is a good "add" move.
            new_chosen, new_score = fill_greedy(set(chosen_set), score)
            if new_score > score + 1e-12:
                chosen_set = new_chosen
                score = new_score
                improved = True
                continue # Restart loop to check other moves

            if time.time() > deadline - 10:
                break

            # 2. Try 1-for-1 swaps
            # For each s in chosen, try replacing with u not in chosen
            for s in list(chosen_set):
                if time.time() > deadline - 10:
                    break
                temp_set = chosen_set - {s}
                # Find best u
                best_u = None
                best_gain = 0
                for u in range(N, 0, -1):
                    if u in chosen_set:
                        continue
                    if can_add(u, temp_set):
                        gain = w[u] - w[s]
                        if gain > best_gain:
                            best_gain = gain
                            best_u = u
                if best_u is not None and best_gain > 1e-12:
                    chosen_set.remove(s)
                    chosen_set.add(best_u)
                    score += best_gain
                    improved = True
                    break # Restart loop

            if time.time() > deadline - 10:
                break

            # 3. Try 1-for-2 swaps: remove s, add u1 and u2
            # This is expensive. We limit to a few s.
            for s in list(chosen_set):
                if time.time() > deadline - 10:
                    break
                temp_set = chosen_set - {s}
                # Find candidates coprime to temp_set
                # candidates = [u for u in range(N, 0, -1) if u not in chosen_set and can_add(u, temp_set)]
                # This list could be large.
                # We want to pick two u1, u2 from candidates such that gcd(u1, u2)==1.
                # And w[u1] + w[u2] > w[s].
                
                # Optimization: Only consider top K candidates by weight?
                # Let's get candidates and sort by weight desc.
                candidates = []
                for u in range(N, 0, -1):
                    if u in chosen_set:
                        continue
                    if can_add(u, temp_set):
                        candidates.append(u)
                        if len(candidates) >= 20: # Limit search space
                            break
                
                # Try pairs from top candidates
                found = False
                for i in range(len(candidates)):
                    if time.time() > deadline - 10:
                        break
                    u1 = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        u2 = candidates[j]
                        if math.gcd(u1, u2) == 1:
                            gain = w[u1] + w[u2] - w[s]
                            if gain > 1e-12:
                                chosen_set.remove(s)
                                chosen_set.add(u1)
                                chosen_set.add(u2)
                                score += gain
                                improved = True
                                found = True
                                break
                    if found:
                        break
                if found:
                    break

            if time.time() > deadline - 10:
                break

            # 4. Try 2-for-1 swaps: remove s1 and s2, add u
            # Less likely to improve if weights are similar, but possible if s1, s2 are small.
            # We skip this for now to save time, or do it rarely.
            
        # Final fill pass
        chosen_set, score = fill_greedy(chosen_set, score)
        return chosen_set, score

    # Run local search
    best_chosen, best_score = local_search(set(best_chosen), best_score)

    # Random restarts with perturbation
    for iteration in range(300):
        if time.time() > deadline - 15:
            break

        # Perturb best_chosen
        # Strategy: Remove k random elements
        k = random.randint(1, 3)
        if len(best_chosen) <= k:
            continue
        remove_set = set(random.sample(list(best_chosen), k))
        perturbed = best_chosen - remove_set
        
        # Optionally add some random compatible elements to the perturbed set
        # to create a different landscape
        for _ in range(random.randint(0, 2)):
            candidates = [a for a in range(1, N + 1) if a not in perturbed and can_add(a, perturbed)]
            if candidates:
                u = random.choice(candidates)
                perturbed.add(u)

        # Greedy fill
        perturbed, p_score = fill_greedy(perturbed, sum(w[a] for a in perturbed))
        
        # Local search
        perturbed, p_score = local_search(perturbed, p_score)

        if p_score > best_score:
            best_score = p_score
            best_chosen = set(perturbed)

    # Final high-precision polish
    if time.time() < deadline - 5:
        for _ in range(5):
            improved = False
            # 1-for-1 swaps with tight tolerance
            for s in list(best_chosen):
                if time.time() > deadline - 5:
                    break
                temp_set = best_chosen - {s}
                for u in range(N, 0, -1):
                    if u in best_chosen:
                        continue
                    if can_add(u, temp_set):
                        new_score = best_score - w[s] + w[u]
                        if new_score > best_score + 1e-15:
                            best_chosen.remove(s)
                            best_chosen.add(u)
                            best_score = new_score
                            improved = True
                            break
            if not improved:
                break
        # Final fill
        best_chosen, best_score = fill_greedy(best_chosen, best_score)

    return list(best_chosen)
