"""Stronger solver for sum-free subset maximization on three fixed instances."""
import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = 115.0  # seconds, safety margin before 150s kill

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve_instance(values, time_limit=35.0):
        """
        Solve one instance using Multi-start Greedy with Local Search.
        """
        n = len(values)
        if n == 0:
            return []
        
        val_set = set(values)
        
        def can_add(v, S, Sums_S):
            """
            Check if v can be added to S (which is sum-free) to keep it sum-free.
            """
            if 2 * v in S:
                return False
            if v in Sums_S:
                return False
            for s in S:
                if v + s in S:
                    return False
            return True

        def get_initial_greedy(order):
            """Build a sum-free set greedily based on given order of indices."""
            S = set()
            Sums_S = set()
            for idx in order:
                v = values[idx]
                if can_add(v, S, Sums_S):
                    S.add(v)
                    Sums_S.add(2 * v)
                    for s in S:
                        Sums_S.add(v + s)
            return S, Sums_S

        def local_search(S, Sums_S, max_iters=100):
            """
            Improve S by adding elements or swapping.
            """
            improved = True
            iters = 0
            while improved and iters < max_iters and time.time() < time_limit + start_time:
                improved = False
                iters += 1
                
                # Try to add any element
                added_any = False
                for v in values:
                    if v in S:
                        continue
                    if can_add(v, S, Sums_S):
                        S.add(v)
                        Sums_S.add(2 * v)
                        for s in S:
                            Sums_S.add(v + s)
                        improved = True
                        added_any = True
                        break 
                if added_any:
                    continue

                # If no single addition works, try swaps
                found_swap = False
                # Try swapping out a few random elements
                candidates_for_removal = random.sample(list(S), min(5, len(S)))
                for u in candidates_for_removal:
                    S_sim = S - {u}
                    Sums_S_sim = set()
                    for a in S_sim:
                        Sums_S_sim.add(2*a)
                        for b in S_sim:
                            Sums_S_sim.add(a+b)
                    
                    added_count = 0
                    for v in values:
                        if v in S_sim:
                            continue
                        if can_add(v, S_sim, Sums_S_sim):
                            S_sim.add(v)
                            Sums_S_sim.add(2*v)
                            for s in S_sim:
                                Sums_S_sim.add(v+s)
                            added_count += 1
                            
                    if added_count > 1:
                        S = S_sim
                        Sums_S = Sums_S_sim
                        improved = True
                        found_swap = True
                        break
                    elif added_count == 1 and random.random() < 0.2:
                        S = S_sim
                        Sums_S = Sums_S_sim
                        improved = True
                        found_swap = True
                        break
                        
                if found_swap:
                    continue
                    
            return S, Sums_S

        best_S = set()
        best_size = 0
        
        # Multi-start strategy
        orders = [
            list(range(n)),
            list(range(n-1, -1, -1)),
            sorted(range(n), key=lambda i: values[i]),
            sorted(range(n), key=lambda i: -values[i]),
            sorted(range(n), key=lambda i: abs(values[i])),
            sorted(range(n), key=lambda i: -abs(values[i])),
            sorted(range(n), key=lambda i: (values[i] % 2 == 0, abs(values[i]))),
            sorted(range(n), key=lambda i: (values[i] < 0, abs(values[i]))),
        ]
        
        num_random_starts = 20
        for _ in range(num_random_starts):
            random_order = list(range(n))
            random.shuffle(random_order)
            orders.append(random_order)
            
        for order in orders:
            if time.time() > start_time + time_limit:
                break
                
            S, Sums_S = get_initial_greedy(order)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
                
            S, Sums_S = local_search(S, Sums_S)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
                
        S_final = list(best_S)
        
        # Final validation
        S_set = set(S_final)
        valid = True
        for a in S_final:
            for b in S_final:
                if a + b in S_set:
                    valid = False
                    break
            if not valid:
                break
                
        if not valid:
            # Fallback: Simple greedy
            S_fb, _ = get_initial_greedy(sorted(range(n), key=lambda i: abs(values[i])))
            return list(S_fb)
            
        return S_final

    def simple_greedy_fb(values):
        n = len(values)
        S = set()
        Sums_S = set()
        for v in sorted(values, key=abs):
            if 2*v in S: continue
            if v in Sums_S: continue
            conflict = False
            for s in S:
                if v+s in S:
                    conflict = True
                    break
            if not conflict:
                S.add(v)
                Sums_S.add(2*v)
                for s in S:
                    Sums_S.add(v+s)
        return list(S)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > deadline:
            while len(results) < 3:
                vals = instance(size, seed)
                results.append(simple_greedy_fb(vals))
            break
        
        vals = instance(size, seed)
        subset = solve_instance(vals)
        results.append(subset)
    
    return results
