import numpy as np
from math import lcm
import random

def construct():
    N = 256
    numbers = list(range(2, N + 1))
    weights = {x: 1.0 / x for x in numbers}
    
    def conflicts(a, b):
        return lcm(a, b) <= N

    def is_valid_addition(selected, x):
        for y in selected:
            if conflicts(x, y):
                return False
        return True

    def greedy_fill(selected, forced=None):
        # selected is a set
        # forced is a number that MUST be included (if not already)
        # We sort remaining numbers by weight desc
        remaining = [x for x in numbers if x not in selected]
        if forced is not None and forced not in selected:
            if is_valid_addition(selected, forced):
                selected.add(forced)
                remaining.remove(forced)
            else:
                # If forced is invalid, we can't use this anchor
                return None
        
        # Sort remaining by weight descending (i.e., x ascending)
        remaining.sort(key=lambda x: -weights[x])
        
        for x in remaining:
            if is_valid_addition(selected, x):
                selected.add(x)
                
        return selected

    def score(selected):
        return sum(weights[x] for x in selected)

    best_set = set()
    best_score = 0.0

    # Strategy 1: Standard Greedy (no force)
    s = greedy_fill(set())
    if s:
        if score(s) > best_score:
            best_score = score(s)
            best_set = s

    # Strategy 2: Force small numbers as anchors
    # It's likely that optimal solutions contain small numbers, but maybe not 2.
    # Try forcing 2, 3, 4, 5, 6, 7, 8, 9, 10
    for anchor in range(2, 11):
        s = greedy_fill(set(), forced=anchor)
        if s:
            sc = score(s)
            if sc > best_score:
                best_score = sc
                best_set = s

    # Strategy 3: Local Search (Hill Climbing)
    # We will try to improve best_set
    
    improved = True
    while improved:
        improved = False
        current_set = set(best_set)
        current_score = best_score
        
        # 1. Try adding any single unselected node
        for x in numbers:
            if x not in current_set:
                if is_valid_addition(current_set, x):
                    new_set = current_set | {x}
                    new_score = current_score + weights[x]
                    if new_score > best_score:
                        best_score = new_score
                        best_set = new_set
                        improved = True
                        break # Restart loop with new best
        
        # 2. Try swapping one selected node with one unselected node
        if not improved:
            selected_list = list(best_set)
            unselected_list = [x for x in numbers if x not in best_set]
            for u in selected_list:
                for v in unselected_list:
                    if weights[v] > weights[u]:
                        # Try swap
                        new_set = best_set - {u}
                        if is_valid_addition(new_set, v):
                            new_score = best_score - weights[u] + weights[v]
                            if new_score > best_score:
                                best_score = new_score
                                best_set = new_set
                                improved = True
                                break
                if improved:
                    break
                    
        # 3. Try removing one node and adding two nodes (if time permits)
        # This is O(N^3) roughly, might be slow. 
        # N=255. 255^3 is ~16M. In Python, this is too slow for 60s.
        # We'll stick to 1-node additions and 1-for-1 swaps.
        
    return list(best_set)
