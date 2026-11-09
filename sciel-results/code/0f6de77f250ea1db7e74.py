"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from random import Random

def construct():
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def can_add(x, S, sums_S, diffs_S):
        """
        Check if x can be added to S to maintain sum-free property.
        S is a list, sums_S is set of a+b for a,b in S, diffs_S is set of a-b for a,b in S.
        Conditions for violation if x added:
        1. 2x in S
        2. x in sums_S (exists a,b in S such that a+b=x)
        3. x in diffs_S (exists a,b in S such that a-b=x)
        """
        if 2 * x in S:
            return False
        if x in sums_S:
            return False
        if x in diffs_S:
            return False
        return True

    def update_structures_on_add(x, S, sums_S, diffs_S):
        """
        Update S, sums_S, diffs_S after adding x.
        Returns new structures.
        """
        new_S = S + [x]
        new_sums = set(sums_S)
        new_diffs = set(diffs_S)
        
        # Add sums involving x
        for a in S:
            new_sums.add(x + a)
        new_sums.add(2 * x)
        
        # Add diffs involving x
        for a in S:
            new_diffs.add(x - a)
            new_diffs.add(a - x)
            
        return new_S, new_sums, new_diffs

    def local_search(elements, seed, max_iters=5000):
        """
        Simulated Annealing / Iterated Local Search.
        """
        n = len(elements)
        if n == 0:
            return []
        
        # Pre-compute all possible sums and diffs for the full set? No, dynamic is better for SA.
        # But for initialization, we can use a greedy approach.
        
        best_size = -1
        best_set = []
        
        # Multiple restarts
        num_restarts = 5
        for r in range(num_restarts):
            # Randomized Greedy Initialization
            # Shuffle elements
            elems = elements[:]
            Random(seed + r).shuffle(elems)
            
            S = []
            S_set = set()
            sums_S = set()
            diffs_S = set()
            
            for x in elems:
                if can_add(x, S_set, sums_S, diffs_S):
                    S.append(x)
                    S_set.add(x)
                    S, sums_S, diffs_S = update_structures_on_add(x, S, sums_S, diffs_S)
                    # Note: update_structures_on_add returns new_S which is a list. 
                    # We need to keep S_set in sync. 
                    # Actually, update_structures_on_add uses S (list) to compute new sums/diffs.
                    # It returns new_S (list), new_sums, new_diffs.
                    # We also need to update S_set.
                    pass # The loop above is a bit messy with state. Let's refactor.

        # Refactored Local Search
        def run_sa(elements, seed):
            n = len(elements)
            elems = elements[:]
            
            # Initialization: Greedy
            Random(seed).shuffle(elems)
            S = []
            S_set = set()
            sums_S = set()
            diffs_S = set()
            
            for x in elems:
                if can_add(x, S_set, sums_S, diffs_S):
                    S.append(x)
                    S_set.add(x)
                    # Update sums and diffs
                    for a in S: # S does not include x yet
                        sums_S.add(x + a)
                        diffs_S.add(x - a)
                        diffs_S.add(a - x)
                    sums_S.add(2 * x)
            
            best_S = S[:]
            best_size = len(S)
            
            # Local Search: Try to add remaining elements
            # Then try swaps?
            # For simplicity, we will do a "fill" phase and then a "swap" phase.
            
            # Fill phase: Try to add any element from the pool not in S
            pool = [x for x in elements if x not in S_set]
            # Sort pool by some heuristic? Random is fine for SA.
            
            # Simulated Annealing steps
            # We will try to add an element from pool, or remove one from S to allow adding others?
            # Adding is always good if possible.
            # Removing is only good if it allows adding > 1 elements.
            
            # Let's do a simple iterative improvement:
            # 1. Try to add any element from pool.
            # 2. If no element can be added, try a swap: remove x from S, add y from pool.
            
            for step in range(2000):
                added = False
                # Try to add a random element from pool
                if pool:
                    idx = random.Random(seed + step).randint(0, len(pool) - 1)
                    y = pool[idx]
                    if can_add(y, S_set, sums_S, diffs_S):
                        # Add y
                        S.append(y)
                        S_set.add(y)
                        pool.remove(y)
                        
                        # Update structures
                        for a in S[:-1]:
                            sums_S.add(y + a)
                            diffs_S.add(y - a)
                            diffs_S.add(a - y)
                        sums_S.add(2 * y)
                        
                        added = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = S[:]
                
                if not added:
                    # Try a swap
                    if pool and S:
                        # Pick a random element from S to remove
                        rem_idx = random.Random(seed + step + 1).randint(0, len(S) - 1)
                        x = S[rem_idx]
                        
                        # Temporarily remove x
                        S.remove(x)
                        S_set.remove(x)
                        
                        # Rebuild sums and diffs? Expensive.
                        # For n=240, S~100. Rebuild O(100^2) = 10000.
                        # 2000 steps * 10000 = 20M ops. Acceptable in Python? Maybe slow.
                        # Let's just rebuild for simplicity in this thought process, 
                        # but in code we might optimize.
                        
                        new_sums = set()
                        new_diffs = set()
                        for a in S:
                            for b in S:
                                new_sums.add(a + b)
                                new_diffs.add(a - b)
                        
                        sums_S = new_sums
                        diffs_S = new_diffs
                        
                        # Try to add a random element from pool
                        idx = random.Random(seed + step + 2).randint(0, len(pool) - 1)
                        y = pool[idx]
                        
                        if can_add(y, S_set, sums_S, diffs_S):
                            # Add y
                            S.append(y)
                            S_set.add(y)
                            pool.remove(y)
                            
                            # Update structures
                            for a in S[:-1]:
                                sums_S.add(y + a)
                                diffs_S.add(y - a)
                                diffs_S.add(a - y)
                            sums_S.add(2 * y)
                            
                            if len(S) > best_size:
                                best_size = len(S)
                                best_S = S[:]
                        else:
                            # Put x back
                            S.append(x)
                            S_set.add(x)
                            # Rebuild again? Or just revert?
                            # Reverting is hard. Rebuild.
                            new_sums = set()
                            new_diffs = set()
                            for a in S:
                                for b in S:
                                    new_sums.add(a + b)
                                    new_diffs.add(a - b)
                            sums_S = new_sums
                            diffs_S = new_diffs

            return best_S

        return run_sa(elements, seed)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        subset = local_search(elems, seed)
        results.append(subset)

    return results
# EVOLVE-BLOCK-END
