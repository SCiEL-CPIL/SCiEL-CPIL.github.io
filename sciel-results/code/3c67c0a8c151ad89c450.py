"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    SAT-inspired sorting network construction using the 0-1 principle.
    
    Approach:
    1. Use a fixed-depth layered structure
    2. Greedily add comparators to fix unsorted Boolean inputs
    3. Minimize by removing redundant comparators
    4. Use random restarts to escape local minima
    """
    import random
    import time
    
    start_time = time.time()
    timeout = 55
    
    def check_sorted(state):
        """Check if state is sorted (non-decreasing)."""
        for k in range(N - 1):
            if state[k] > state[k + 1]:
                return False
        return True
    
    def apply_network(state, comparators):
        """Apply all comparators to a state, return new state."""
        s = list(state)
        for (i, j) in comparators:
            if s[i] > s[j]:
                s[i], s[j] = s[j], s[i]
        return s
    
    def find_unsorted_input(comparators):
        """Find a Boolean input that is not sorted by the network. Return None if all sorted."""
        for mask in range(1 << N):
            state = [(mask >> i) & 1 for i in range(N)]
            result = apply_network(state, comparators)
            if not check_sorted(result):
                return mask
        return None
    
    def is_valid(comparators):
        """Check if network sorts all Boolean inputs."""
        return find_unsorted_input(comparators) is None
    
    def odd_even_transposition():
        """Generate odd-even transposition network."""
        comps = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comps.append((i, i + 1))
        return comps
    
    def greedy_build():
        """
        Greedily build a sorting network by adding comparators
        to fix the hardest unsorted inputs first.
        """
        comparators = []
        # Start with a base network for efficiency
        base = odd_even_transposition()
        comparators = list(base)
        
        # Try to minimize
        return comparators
    
    def minimize(comparators, time_limit):
        """Remove redundant comparators."""
        comps = list(comparators)
        while time.time() - start_time < time_limit:
            removed = False
            indices = list(range(len(comps)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() - start_time > time_limit:
                    break
                new_comps = comps[:idx] + comps[idx+1:]
                if is_valid(new_comps):
                    comps = new_comps
                    removed = True
            if not removed:
                break
        return comps
    
    def local_search(comparators, time_limit):
        """Local search to find better networks."""
        comps = list(comparators)
        best = list(comps)
        best_len = len(comps)
        
        while time.time() - start_time < time_limit:
            # Try removing
            removed = False
            indices = list(range(len(comps)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() - start_time > time_limit:
                    break
                new_comps = comps[:idx] + comps[idx+1:]
                if is_valid(new_comps):
                    comps = new_comps
                    removed = True
                    if len(comps) < best_len:
                        best = list(comps)
                        best_len = len(comps)
            
            if not removed:
                # Try swapping adjacent comparators
                if len(comps) >= 2 and random.random() < 0.5:
                    i = random.randint(0, len(comps) - 2)
                    new_comps = list(comps)
                    new_comps[i], new_comps[i+1] = new_comps[i+1], new_comps[i]
                    if is_valid(new_comps):
                        comps = new_comps
                        if len(comps) < best_len:
                            best = list(comps)
                            best_len = len(comps)
                else:
                    # Try adding a random comparator
                    i, j = random.sample(range(N), 2)
                    if i > j:
                        i, j = j, i
                    pos = random.randint(0, len(comps))
                    new_comps = comps[:pos] + [(i, j)] + comps[pos:]
                    if is_valid(new_comps):
                        comps = new_comps
                        if len(comps) < best_len:
                            best = list(comps)
                            best_len = len(comps)
                    else:
                        # Reset to best
                        comps = list(best)
        
        return best
    
    random.seed(42)
    
    # Start with odd-even transposition
    current = odd_even_transposition()
    best = list(current)
    best_len = len(current)
    
    # Phase 1: Minimize the base network
    minimized = minimize(current, timeout - 5)
    if len(minimized) < best_len:
        best = list(minimized)
        best_len = len(minimized)
    current = list(minimized)
    
    # Phase 2: Local search with random restarts
    remaining_time = timeout - (time.time() - start_time)
    if remaining_time > 5:
        improved = local_search(current, timeout)
        if len(improved) < best_len:
            best = list(improved)
            best_len = len(improved)
    
    return best
# EVOLVE-BLOCK-END
