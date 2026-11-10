"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    
    def is_valid_set(S):
        """Check if set S satisfies all constraints."""
        S_list = list(S)
        n = len(S_list)
        for i, a in enumerate(S_list):
            others = [S_list[j] for j in range(n) if j != i]
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    return False
        return True
    
    def can_add(S, c):
        """Check if c can be added to set S."""
        S_list = list(S)
        n = len(S_list)
        
        # Check constraint for c: no triple from S sums to multiple of c
        if n >= 3:
            for t in combinations(S_list, 3):
                if sum(t) % c == 0:
                    return False
        
        # Check constraint for each a in S: no triple from S∪{c} excluding a sums to multiple of a
        for i, a in enumerate(S_list):
            # Triples from S excluding a (already valid, skip)
            # Triples containing c: {c, s1, s2} where s1,s2 from S\{a}
            others = [S_list[j] for j in range(n) if j != i]
            if len(others) >= 2:
                for s1_idx in range(len(others)):
                    s1 = others[s1_idx]
                    for s2_idx in range(s1_idx + 1, len(others)):
                        s2 = others[s2_idx]
                        if (c + s1 + s2) % a == 0:
                            return False
        return True
    
    def greedy_build(order):
        """Build a valid set by processing candidates in given order."""
        S = []
        for c in order:
            if len(S) >= 48:
                break
            if time.time() > deadline:
                break
            if can_add(S, c):
                S.append(c)
        return S
    
    def local_search(S):
        """Try to improve set S by swaps and additions."""
        S = list(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding any unused element
            used = set(S)
            for c in range(192, 0, -1):
                if c not in used and len(S) < 48:
                    if can_add(S, c):
                        S.append(c)
                        improved = True
                        break
            # Try swaps: remove one, add one
            if not improved:
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    S_without_i = [S[j] for j in range(len(S)) if j != i]
                    used = set(S_without_i)
                    for c in range(192, 0, -1):
                        if c not in used:
                            if can_add(S_without_i, c):
                                S = S_without_i + [c]
                                improved = True
                                break
                    if improved:
                        break
        return S
    
    # Multiple random orderings
    best = []
    base_order = list(range(192, 0, -1))
    
    # First try descending order
    S = greedy_build(base_order)
    S = local_search(S)
    if len(S) > len(best):
        best = S
    
    # Try several random perturbations
    for attempt in range(200):
        if time.time() > deadline - 5.0:
            break
        # Random shuffle with bias toward larger numbers
        order = list(range(1, 193))
        random.shuffle(order)
        # Bias: sort by a random key that favors larger numbers sometimes
        if random.random() < 0.5:
            order.sort(key=lambda x: random.random() * 0.5 + x / 192.0, reverse=True)
        else:
            order.sort(reverse=True)
        
        S = greedy_build(order)
        S = local_search(S)
        if len(S) > len(best):
            best = S
    
    # Final local search from best
    if time.time() < deadline - 2.0:
        best = local_search(best)
    
    return best
# EVOLVE-BLOCK-END
