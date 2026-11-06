"""Improved program: portfolio of randomized greedy constructions for max-cardinality subset of [1,100] with no subset summing to 200."""

import time
import random
import math

TARGET = 200

# Precompute subset-sum reachability for validation
def _validate(S):
    """Check that no subset of S sums to TARGET. Returns True if valid."""
    reachable = {0}
    for x in S:
        new = set()
        for s in reachable:
            ns = s + x
            if ns <= TARGET:
                new.add(ns)
        reachable |= new
    return TARGET not in reachable

def _greedy_construction(order, rng=None):
    """
    Greedily add elements in the given order if doing so doesn't create a subset summing to TARGET.
    Returns (selected_set, reachable_sum_set).
    """
    selected = []
    reachable = {0}
    for x in order:
        new = set()
        for s in reachable:
            ns = s + x
            if ns <= TARGET:
                new.add(ns)
        if TARGET not in new:
            selected.append(x)
            reachable |= new
    return selected, reachable

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # Strategy 1: Descending order (the seed approach)
    best_selected = []
    best_size = 0
    
    # Try descending order first
    order_desc = list(range(100, 0, -1))
    selected, reachable = _greedy_construction(order_desc)
    if len(selected) > best_size:
        best_size = len(selected)
        best_selected = list(selected)
    
    # Strategy 2: Random shuffles with various heuristics
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Generate a random permutation with some structure
        if attempts % 5 == 0:
            # Pure random shuffle
            order = list(range(1, 101))
            random.shuffle(order)
        elif attempts % 5 == 1:
            # Descending with some random swaps
            order = list(range(100, 0, -1))
            for _ in range(random.randint(5, 20)):
                i, j = random.sample(range(100), 2)
                order[i], order[j] = order[j], order[i]
        elif attempts % 5 == 2:
            # Ascending
            order = list(range(1, 101))
            random.shuffle(order)
            # Bias toward larger elements
            order.sort(key=lambda x: x + random.gauss(0, 30), reverse=True)
        elif attempts % 5 == 3:
            # Weighted random: larger elements more likely to come first
            order = list(range(1, 101))
            weights = [x ** 0.5 for x in order]  # sqrt weighting
            shuffled = []
            remaining = list(order)
            remaining_weights = list(weights)
            while remaining:
                total = sum(remaining_weights)
                r = random.random() * total
                cumulative = 0
                for i, w in enumerate(remaining_weights):
                    cumulative += w
                    if cumulative >= r:
                        shuffled.append(remaining[i])
                        remaining.pop(i)
                        remaining_weights.pop(i)
                        break
            order = shuffled
        else:
            # Reverse random shuffle
            order = list(range(1, 101))
            random.shuffle(order)
            order.reverse()
        
        selected, reachable = _greedy_construction(order)
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = list(selected)
    
    # Final validation
    if best_selected and _validate(best_selected):
        return best_selected
    
    # Fallback: return the seed result
    selected, reachable = _greedy_construction(list(range(100, 0, -1)))
    return selected
