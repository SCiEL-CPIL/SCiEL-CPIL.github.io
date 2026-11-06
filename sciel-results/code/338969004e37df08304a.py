"""Rewritten program: maximize exact reciprocal sum of a subset of [2,256]
with lcm(a,b) > 256 for all distinct pairs, using local search with swaps."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm, gcd
from fractions import Fraction
from itertools import combinations

DEADLINE = 82.0  # safety margin before 90s kill

def construct():
    start_time = time.time()
    
    N = 255  # values 2..256
    vals = list(range(2, 257))
    recip = [1.0 / vals[i] for i in range(N)]
    
    # Precompute incompatible pairs as bitsets for fast checking
    # incompat[i] is a bitmask of indices j where lcm(vals[i], vals[j]) <= 256
    incompat = [0] * N
    for i in range(N):
        vi = vals[i]
        for j in range(i + 1, N):
            vj = vals[j]
            if lcm(vi, vj) <= 256:
                incompat[i] |= (1 << j)
                incompat[j] |= (1 << i)
    
    def is_valid_with_set(i, selected_set):
        """Check if index i is compatible with all indices in selected_set using bitset."""
        # Check if any bit in incompat[i] overlaps with selected_set bits
        # We need a bitset for selected_set
        # For speed, we'll pass a bitset
        pass
    
    def is_valid_bitset(i, selected_bits):
        """Check if index i is compatible with all indices in selected_bits (int bitmask)."""
        return (incompat[i] & selected_bits) == 0
    
    def reciprocal_sum_float(selected_indices):
        s = 0.0
        for i in selected_indices:
            s += recip[i]
        return s
    
    def reciprocal_sum_exact(selected_indices):
        s = Fraction(0, 1)
        for i in selected_indices:
            s += Fraction(1, vals[i])
        return s
    
    def greedy_init(order):
        selected = []
        selected_bits = 0
        for i in order:
            if is_valid_bitset(i, selected_bits):
                selected.append(i)
                selected_bits |= (1 << i)
        return selected
    
    # Generate multiple greedy inits
    best_selected = None
    best_score = 0.0
    
    # Strategy 1: greedy by smallest value first (largest reciprocal)
    order1 = list(range(N))
    sel1 = greedy_init(order1)
    score1 = reciprocal_sum_float(sel1)
    best_selected = sel1
    best_score = score1
    
    # Strategy 2: greedy by largest value first
    order2 = list(range(N - 1, -1, -1))
    sel2 = greedy_init(order2)
    score2 = reciprocal_sum_float(sel2)
    if score2 > best_score:
        best_selected = sel2
        best_score = score2
    
    # Strategy 3: random orderings
    random.seed(42)
    for trial in range(50):
        if time.time() - start_time > DEADLINE - 25:
            break
        order = list(range(N))
        random.shuffle(order)
        sel = greedy_init(order)
        score = reciprocal_sum_float(sel)
        if score > best_score:
            best_selected = sel
            best_score = score
    
    def local_search(selected):
        """1-swap and 2-swap local search using float scores."""
        selected_set = set(selected)
        selected_bits = 0
        for i in selected:
            selected_bits |= (1 << i)
        score = reciprocal_sum_float(selected)
        
        improved = True
        while improved:
            improved = False
            
            # Try adding each unselected element (1-add)
            for i in range(N):
                if not (selected_bits & (1 << i)):
                    if is_valid_bitset(i, selected_bits):
                        new_score = score + recip[i]
                        if new_score > score:
                            selected.append(i)
                            selected_set.add(i)
                            selected_bits |= (1 << i)
                            score = new_score
                            improved = True
            
            # Try 1-swap: remove j from selected, add i not in selected
            for j in list(selected):
                vj = vals[j]
                # Bits of selected without j
                sel_without_j = selected_bits & ~(1 << j)
                # Find best i to swap in
                best_i = -1
                best_delta = 0.0
                for i in range(N):
                    if (selected_bits & (1 << i)) or i == j:
                        continue
                    if is_valid_bitset(i, sel_without_j):
                        delta = recip[i] - recip[j]
                        if delta > best_delta:
                            best_delta = delta
                            best_i = i
                if best_i >= 0 and best_delta > 0:
                    selected.remove(j)
                    selected_set.discard(j)
                    selected_bits &= ~(1 << j)
                    selected.append(best_i)
                    selected_set.add(best_i)
                    selected_bits |= (1 << best_i)
                    score += best_delta
                    improved = True
                    break  # restart scan after a swap
            
            # Try 2-swap: remove two {j1, j2}, add two {i1, i2}
            # Only try if 1-swap didn't improve (to save time)
            if not improved and len(selected) >= 2:
                found_2swap = False
                sel_list = list(selected)
                for idx_j1 in range(len(sel_list)):
                    j1 = sel_list[idx_j1]
                    for idx_j2 in range(idx_j1 + 1, len(sel_list)):
                        j2 = sel_list[idx_j2]
                        sel_without = selected_bits & ~(1 << j1) & ~(1 << j2)
                        # Find best pair (i1, i2) to add
                        best_pair = None
                        best_2delta = 0.0
                        # Candidate i's must be compatible with sel_without and with each other
                        candidates = []
                        for i in range(N):
                            if (selected_bits & (1 << i)) or i == j1 or i == j2:
                                continue
                            if is_valid_bitset(i, sel_without):
                                candidates.append(i)
                        # Try all pairs from candidates (limit for speed)
                        if len(candidates) <= 200:
                            for a in range(len(candidates)):
                                i1 = candidates[a]
                                for b in range(a + 1, len(candidates)):
                                    i2 = candidates[b]
                                    # Check i1 and i2 are compatible with each other
                                    if (incompat[i1] & (1 << i2)) == 0:
                                        delta = recip[i1] + recip[i2] - recip[j1] - recip[j2]
                                        if delta > best_2delta:
                                            best_2delta = delta
                                            best_pair = (i1, i2)
                        if best_pair and best_2delta > 0:
                            i1, i2 = best_pair
                            selected.remove(j1)
                            selected.remove(j2)
                            selected_set.discard(j1)
                            selected_set.discard(j2)
                            selected_bits &= ~(1 << j1)
                            selected_bits &= ~(1 << j2)
                            selected.append(i1)
                            selected.append(i2)
                            selected_set.add(i1)
                            selected_set.add(i2)
                            selected_bits |= (1 << i1)
                            selected_bits |= (1 << i2)
                            score += best_2delta
                            improved = True
                            found_2swap = True
                            break
                    if found_2swap:
                        break
            
            if time.time() - start_time > DEADLINE - 3:
                break
        
        return selected, score
    
    # Run local search on best solution
    best_selected, best_score = local_search(best_selected)
    
    # Try local search on other initializations
    for trial in range(15):
        if time.time() - start_time > DEADLINE - 5:
            break
        order = list(range(N))
        random.shuffle(order)
        sel = greedy_init(order)
        sel, score = local_search(sel)
        if score > best_score:
            best_selected = sel
            best_score = score
    
    # Final polish: try 1-swap and 1-add one more time
    for _ in range(3):
        sel, score = local_search(best_selected)
        if score > best_score:
            best_selected = sel
            best_score = score
        else:
            break
    
    # Convert back to values
    result = sorted([vals[i] for i in best_selected])
    return result
# EVOLVE-BLOCK-END
