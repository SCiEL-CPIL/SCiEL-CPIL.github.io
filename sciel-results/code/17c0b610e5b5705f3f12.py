"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd

def construct():
    _t0 = time.time()
    _DEADLINE = 125.0  # safety margin before 150s wall-clock kill

    def _remaining():
        return _DEADLINE - time.time()

    # Precompute weights: score = sum 1/(512 - a)
    # Larger a => larger weight
    candidates = list(range(1, 512))
    weights = {a: 1.0 / (512.0 - a) for a in candidates}

    # Precompute prime factorization for fast gcd check via shared primes
    # Build a conflict graph: a and b conflict if gcd(a,b) > 1
    # We'll use a greedy + local search approach

    # Greedy: pick highest weight candidates that don't conflict
    def _greedy(order):
        chosen = []
        chosen_set = set()
        for a in order:
            if a in chosen_set:
                continue
            # Check coprime with all chosen
            ok = True
            for b in chosen:
                if gcd(a, b) != 1:
                    ok = False
                    break
            if ok:
                chosen.append(a)
                chosen_set.add(a)
        return chosen

    # Initial greedy: sort by weight descending
    order = sorted(candidates, key=lambda x: -weights[x])
    best = _greedy(order)
    best_score = sum(weights[a] for a in best)

    # Local search: try to swap out a low-value element for a higher-value one
    def _local_search(initial):
        chosen = list(initial)
        chosen_set = set(chosen)
        score = sum(weights[a] for a in chosen)
        improved = True
        while improved and _remaining() > 5.0:
            improved = False
            # Try removing each element and adding the best compatible replacement
            for i, a in enumerate(chosen):
                trial = chosen[:i] + chosen[i+1:]
                trial_set = set(trial)
                # Find best a' not in trial_set that is coprime to all in trial
                best_add = None
                best_add_w = 0.0
                for cand in candidates:
                    if cand in trial_set:
                        continue
                    if weights[cand] < weights[a]:  # only consider improvement
                        break  # candidates sorted? No, just iterate all
                    # Check coprime
                    ok = True
                    for b in trial:
                        if gcd(cand, b) != 1:
                            ok = False
                            break
                    if ok and weights[cand] > best_add_w:
                        best_add = cand
                        best_add_w = weights[cand]
                if best_add is not None and best_add_w > weights[a]:
                    chosen = trial + [best_add]
                    chosen_set = set(cholen)
                    score = sum(weights[x] for x in chosen)
                    improved = True
                    break
        return chosen, score

    # Better local search: iterate over positions
    def _local_search2(initial):
        chosen = list(initial)
        score = sum(weights[a] for a in chosen)
        n_iter = 0
        max_iter = 200
        improved = True
        while improved and n_iter < max_iter and _remaining() > 5.0:
            improved = False
            n_iter += 1
            for i in range(len(chosen)):
                a = chosen[i]
                without = chosen[:i] + chosen[i+1:]
                without_set = set(without)
                best_add = None
                best_w = weights[a] - 1e-12  # must strictly improve
                for cand in candidates:
                    if cand in without_set:
                        continue
                    if weights[cand] <= best_w:
                        continue
                    ok = True
                    for b in without:
                        if gcd(cand, b) != 1:
                            ok = False
                            break
                    if ok:
                        best_add = cand
                        best_w = weights[cand]
                if best_add is not None:
                    chosen = without + [best_add]
                    score = sum(weights[x] for x in chosen)
                    improved = True
                    break
        return chosen, score

    # Multiple greedy orders with randomization
    import random
    rng = random.Random(42)
    
    # Try different orderings
    attempts = 0
    while _remaining() > 10.0 and attempts < 200:
        attempts += 1
        # Random perturbation of order
        perturbed = order[:]
        rng.shuffle(perturbed)
        # Keep top few in weight order
        top_k = 30
        top = sorted(perturbed, key=lambda x: -weights[x])[:top_k]
        rest = [x for x in perturbed if x not in set(top)]
        order_try = top + rest
        cand = _greedy(order_try)
        s = sum(weights[a] for a in cand)
        if s > best_score:
            best = cand
            best_score = s
    
    # Polish with local search
    if _remaining() > 15.0:
        best, best_score = _local_search2(best)
    
    return best
# EVOLVE-BLOCK-END
