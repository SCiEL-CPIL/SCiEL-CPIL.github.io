"""Maximum Weight Independent Set on 3-AP hypergraph via specialized branch-and-bound."""

from fractions import Fraction
import time
import sys

N = 256

def construct():
    start_time = time.time()
    
    # 1. Precompute conflict pairs
    # conflict_pairs[v] is a list of tuples (a, b) such that {a, b, v} is a 3-AP.
    conflict_pairs = [[] for _ in range(N + 1)]
    
    for v in range(1, N + 1):
        # v as smallest: (v, v+d, v+2d) -> pair (v+d, v+2d)
        d = 1
        while v + 2 * d <= N:
            conflict_pairs[v].append((v + d, v + 2 * d))
            d += 1
        # v as middle: (v-d, v, v+d) -> pair (v-d, v+d)
        d = 1
        while v - d >= 1:
            conflict_pairs[v].append((v - d, v + d))
            d += 1
        # v as largest: (v-2d, v-d, v) -> pair (v-2d, v-d)
        d = 1
        while v - 2 * d >= 1:
            conflict_pairs[v].append((v - 2 * d, v - d))
            d += 1

    # 2. Precompute weights using Fraction
    weights = [Fraction(0)] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = Fraction(1, i)
        
    # 3. Greedy initialization to get a good initial lower bound
    selected_greedy = set()
    available_greedy = set(range(1, N + 1))
    current_weight_greedy = Fraction(0)
    
    while available_greedy:
        v = min(available_greedy)
        selected_greedy.add(v)
        current_weight_greedy += weights[v]
        available_greedy.discard(v)
        
        for a, b in conflict_pairs[v]:
            if a in selected_greedy:
                available_greedy.discard(b)
            if b in selected_greedy:
                available_greedy.discard(a)
                
    best_set = list(selected_greedy)
    best_weight = current_weight_greedy
    
    sys.setrecursionlimit(10000)
    
    def bnb(selected, available, current_weight):
        nonlocal best_set, best_weight
        
        # Time check
        if time.time() - start_time > 58:
            return
        
        # Upper bound: sum of weights of all available vertices
        # Any valid extension must be a subset of 'available'.
        ub = current_weight
        for v in available:
            ub += weights[v]
            
        if ub <= best_weight:
            return
            
        if not available:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = list(selected)
            return
        
        # Pick vertex with highest weight (min index) to branch on
        v = min(available)
        
        # Branch 1: Include v
        new_selected = set(selected)
        new_selected.add(v)
        
        new_available = set(available)
        new_available.discard(v)
        
        for a, b in conflict_pairs[v]:
            if a in selected:
                new_available.discard(b)
            if b in selected:
                new_available.discard(a)
                
        bnb(new_selected, new_available, current_weight + weights[v])
        
        # Branch 2: Exclude v
        new_available_excl = set(available)
        new_available_excl.discard(v)
        
        bnb(selected, new_available_excl, current_weight)
        
    # Start B&B from empty set
    bnb(set(), set(range(1, N + 1)), Fraction(0))
    
    return best_set
