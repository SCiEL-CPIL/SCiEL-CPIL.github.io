"""Rewritten program for closest strict underpacking of unit fractions."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    inv = [Fraction(1, d) for d in denominators]
    # Also store as float for fast comparisons where possible, but use Fraction for exactness
    inv_float = [1.0 / d for d in denominators]
    
    # Precompute all pair sums as Fractions for speed in double-swap
    # Actually, let's just use Fraction arithmetic but optimize the search.
    
    # 1. Greedy baseline: pick in order while sum < 1
    selected = set()
    total = Fraction(0, 1)
    for i in range(n):
        if total + inv[i] < 1:
            selected.add(i)
            total += inv[i]
    
    best_selected = set(selected)
    best_total = total
    best_deficit = 1 - total
    
    # Helper to check if a set is valid and update best
    def try_update(sel, tot):
        nonlocal best_selected, best_total, best_deficit
        if tot < 1 and tot > best_total:
            best_total = tot
            best_selected = set(sel)
            best_deficit = 1 - tot
            return True
        return False
    
    # Local Search with time budget
    # We will use a multi-start local search with random restarts if time permits.
    # But first, exhaust local search from the greedy start.
    
    def local_search(start_sel, start_tot, time_limit):
        nonlocal best_selected, best_total
        selected = set(start_sel)
        total = start_tot
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Single insertion
            for i in range(n):
                if i not in selected:
                    new_total = total + inv[i]
                    if new_total < 1:
                        if try_update(selected | {i}, new_total):
                            selected = best_selected
                            total = best_total
                            improved = True
            
            # 2. Single swap
            for i in list(selected):
                for j in range(n):
                    if j not in selected:
                        new_total = total - inv[i] + inv[j]
                        if new_total < 1:
                            if try_update((selected - {i}) | {j}, new_total):
                                selected = best_selected
                                total = best_total
                                improved = True
            
            # 3. Double swap (remove 2, add 2)
            if time.time() < time_limit:
                sel_list = list(selected)
                unselected = [j for j in range(n) if j not in selected]
                if len(sel_list) >= 2 and len(unselected) >= 2:
                    for a_idx in range(len(sel_list)):
                        if time.time() >= time_limit:
                            break
                        for b_idx in range(a_idx + 1, len(sel_list)):
                            if time.time() >= time_limit:
                                break
                            i1, i2 = sel_list[a_idx], sel_list[b_idx]
                            for j1_idx in range(len(unselected)):
                                if time.time() >= time_limit:
                                    break
                                for j2_idx in range(j1_idx + 1, len(unselected)):
                                    j1, j2 = unselected[j1_idx], unselected[j2_idx]
                                    new_total = total - inv[i1] - inv[i2] + inv[j1] + inv[j2]
                                    if new_total < 1:
                                        if try_update((selected - {i1, i2}) | {j1, j2}, new_total):
                                            selected = best_selected
                                            total = best_total
                                            improved = True
        return selected, total
    
    # Run local search from greedy
    local_search(best_selected, best_total, deadline - 5)
    
    # If we still have time, try random restarts
    if time.time() < deadline - 5:
        # Random starts: pick random subsets
        # To make it effective, we can start with a random subset of size similar to best
        # or just random subsets and let local search fix them.
        # A good heuristic is to start with a random permutation greedy.
        
        # Let's try a few random permutations greedy + local search
        remaining_time = deadline - time.time()
        # Estimate cost of one local search. If it's too high, we can't do many.
        # Let's just do a few if there's ample time.
        
        # A better approach: Simulated Annealing or just more aggressive local search?
        # Given the small N=47, the state space is huge.
        
        # Let's try random subset of size ~15-20 and local search.
        # But local search might be slow.
        
        # Alternative: Improve the double-swap to be faster?
        # The bottleneck is usually the O(N^4) double swap.
        
        # Let's add a "triple swap" if time permits? No, O(N^6) is too slow.
        
        # Let's just do a few random restarts with limited local search time.
        # Each restart gets 1 second.
        
        while time.time() < deadline - 2:
            # Generate a random starting set
            # Pick a random subset
            # To be close to 1, we want a mix of small and large denominators.
            # Random subset of size ~ 20
            size = random.randint(10, 25)
            start_sel = set(random.sample(range(n), size))
            start_tot = sum(inv[i] for i in start_sel)
            
            if start_tot >= 1:
                # Remove random elements until < 1
                while start_tot >= 1 and start_sel:
                    to_remove = random.choice(list(start_sel))
                    start_sel.remove(to_remove)
                    start_tot -= inv[to_remove]
            
            # Run a quick local search (limited time)
            # We need a version of local search that respects a short time limit
            # and updates the global best.
            
            # Since local_search above updates global best, we can just call it
            # but we need to be careful it doesn't take too long.
            # Let's limit it to 1 second per restart.
            local_search(start_sel, start_tot, time.time() + 1.0)
            
    return list(best_selected)
# EVOLVE-BLOCK-END
