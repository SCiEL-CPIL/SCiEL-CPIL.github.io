"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S_set):
        """Check if adding x to S_set maintains sum-free property.
        
        Violations when adding x:
        1. 2*x in S_set (x+x=2x)
        2. exists c in S_set such that c-x in S_set (i.e., (c-x)+x=c)
        3. exists a,b in S_set such that a+b=x
        """
        # Check 1: 2x in S_set
        if 2 * x in S_set:
            return False
        # Check 2: exists c in S_set with c-x in S_set
        for c in S_set:
            if c - x in S_set:
                return False
        # Check 3: exists a in S_set with x-a in S_set
        for a in S_set:
            if x - a in S_set:
                return False
        return True

    def greedy_build(S, order=None):
        """Greedy construction: add elements in given order if they don't create conflicts."""
        if order is None:
            order = list(S)
        S_set = set()
        result = []
        for x in order:
            if can_add(x, S_set):
                result.append(x)
                S_set.add(x)
        return result

    def local_search_improve(S, initial_subset, max_rounds=50):
        """Local search: iteratively try to add elements that fit."""
        current = set(initial_subset)
        remaining = [x for x in S if x not in current]
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            added = False
            for x in remaining[:]:
                if can_add(x, current):
                    current.add(x)
                    remaining.remove(x)
                    added = True
            if not added:
                break
        return list(current)

    def swap_improve(S, current, max_rounds=30):
        """Try to improve by removing one element and adding one or two others."""
        current_set = set(current)
        remaining = [x for x in S if x not in current_set]
        
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            
            # Try 1-swap: remove 1, add 1
            for rem in list(current_set):
                if time.time() > deadline:
                    break
                temp_set = current_set - {rem}
                for x in remaining:
                    if can_add(x, temp_set):
                        current_set = temp_set | {x}
                        remaining = [y for y in S if y not in current_set]
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try 2-swap: remove 1, add 2
            for rem in list(current_set):
                if time.time() > deadline:
                    break
                temp_set = current_set - {rem}
                new_elements = []
                for x in remaining:
                    if len(new_elements) >= 2:
                        break
                    if can_add(x, temp_set):
                        temp_set.add(x)
                        new_elements.append(x)
                if len(new_elements) >= 2:
                    current_set = temp_set
                    remaining = [y for y in S if y not in current_set]
                    improved = True
                    break
        
            if not improved:
                break
        return list(current_set)

    def solve_one(S, seed):
        """Find a large sum-free subset of S using multiple strategies."""
        n = len(S)
        if n == 0:
            return []
        
        best = []
        
        # Strategy 1: All odd numbers (always sum-free)
        odds = [x for x in S if x % 2 != 0]
        if len(odds) > len(best):
            best = odds
        
        # Strategy 2: Greedy by absolute value
        greedy_abs = greedy_build(S, sorted(S, key=abs))
        if len(greedy_abs) > len(best):
            best = greedy_abs
        
        # Strategy 3: Greedy by value (ascending)
        greedy_asc = greedy_build(S, sorted(S))
        if len(greedy_asc) > len(best):
            best = greedy_asc
        
        # Strategy 4: Greedy by descending absolute value
        greedy_desc_abs = greedy_build(S, sorted(S, key=abs, reverse=True))
        if len(greedy_desc_abs) > len(best):
            best = greedy_desc_abs
        
        # Strategy 5: Local search from odds
        if time.time() < deadline - 5:
            ls_from_odds = local_search_improve(S, odds, max_rounds=50)
            if len(ls_from_odds) > len(best):
                best = ls_from_odds
        
        # Strategy 6: Local search from greedy_abs
        if time.time() < deadline - 5:
            ls_from_greedy = local_search_improve(S, greedy_abs, max_rounds=50)
            if len(ls_from_greedy) > len(best):
                best = ls_from_greedy
        
        # Strategy 7: Random restarts with local search - more diverse
        rng = Random(seed)
        num_restarts = 20 if n <= 100 else 12
        for i in range(num_restarts):
            if time.time() > deadline - 3:
                break
            # Vary the probability based on iteration
            base_p = 0.25 + 0.35 * (i / max(num_restarts - 1, 1))
            p = base_p + rng.random() * 0.15
            random_subset = [x for x in S if rng.random() < p]
            # Ensure it's sum-free first by removing conflicts
            cleaned = greedy_build(random_subset, random_subset)
            # Then local search
            improved = local_search_improve(S, cleaned, max_rounds=30)
            if len(improved) > len(best):
                best = improved
        
        # Strategy 8: Swap improvement on best (now includes 1-swap and 2-swap)
        if time.time() < deadline - 3:
            swapped = swap_improve(S, best, max_rounds=25)
            if len(swapped) > len(best):
                best = swapped
            # Local search after swap
            if time.time() < deadline - 2:
                final_ls = local_search_improve(S, best, max_rounds=30)
                if len(final_ls) > len(best):
                    best = final_ls
        
        # Strategy 9: Final polish - try adding any remaining elements
        if time.time() < deadline - 1:
            final_polish = local_search_improve(S, best, max_rounds=20)
            if len(final_polish) > len(best):
                best = final_polish
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline - 5:
            # Fallback: just return odds
            S = instance(size, seed)
            results.append([x for x in S if x % 2 != 0])
        else:
            S = instance(size, seed)
            result = solve_one(S, seed)
            # Verify sum-free
            if not is_sum_free(result):
                # Fallback to odds
                result = [x for x in S if x % 2 != 0]
            results.append(result)
    
    return results
# EVOLVE-BLOCK-END
