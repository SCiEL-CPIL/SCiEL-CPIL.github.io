import time
import random

def construct():
    start = time.time()
    DEADLINE = 140.0  # Safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Sset = set(S)
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i, n):
                b = S_list[j]
                if a + b in Sset:
                    return False
        return True

    def greedy_add(V, order, S_initial=None):
        """
        Greedily add elements from 'order' to 'S_initial' (or empty set)
        if they maintain the sum-free property.
        """
        if S_initial is None:
            S = []
            Sset = set()
        else:
            S = list(S_initial)
            Sset = set(S)
        
        for x in order:
            if x in Sset:
                continue
            # Check if x can be added
            # Violation if x+a in Sset (x+a=c) or x-a in Sset (x-a=b => a+b=x)
            valid = True
            for a in S:
                if x + a in Sset or x - a in Sset:
                    valid = False
                    break
            if valid:
                S.append(x)
                Sset.add(x)
        return S

    def local_search(S, V, time_limit, start_time, rng):
        """
        Local search: 
        1. Try to add any available element.
        2. Try to remove a random element and re-greedy the rest.
        """
        S = list(S)
        Sset = set(S)
        best_S = list(S)
        best_size = len(S)
        
        # Pre-compute available elements
        available = [x for x in V if x not in Sset]
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # Phase 1: Add any valid element
            if available:
                rng.shuffle(available)
                new_available = []
                for x in available:
                    valid = True
                    for a in S:
                        if x + a in Sset or x - a in Sset:
                            valid = False
                            break
                    if valid:
                        S.append(x)
                        Sset.add(x)
                        improved = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = list(S)
                    else:
                        new_available.append(x)
                available = new_available
            
            # Phase 2: Perturb and Re-greedy
            if not improved and len(S) > 0:
                # Remove 1 or 2 random elements
                k = 1 if rng.random() < 0.5 else 2
                if len(S) >= k:
                    indices = rng.sample(range(len(S)), k)
                    indices.sort(reverse=True)
                    removed = []
                    for idx in indices:
                        removed.append(S[idx])
                        Sset.remove(S[idx])
                        S.pop(idx)
                    
                    # Re-greedy from scratch using remaining S as initial
                    # Order: shuffled V
                    perm = list(V)
                    rng.shuffle(perm)
                    S_new = greedy_add(V, perm, S_initial=S)
                    
                    if len(S_new) > best_size:
                        S = S_new
                        Sset = set(S)
                        best_size = len(S)
                        best_S = list(S)
                        available = [x for x in V if x not in Sset]
                        improved = True
                    else:
                        # Revert S to best_S? Or just continue with current S?
                        # If we revert to best_S, we might loop.
                        # Better to accept the perturbation even if not better, to escape local minima?
                        # But we want to maximize size.
                        # Let's just keep the best_S found so far, and use a fresh random perturbation from best_S next time.
                        S = list(best_S)
                        Sset = set(S)
                        available = [x for x in V if x not in Sset]
                        
            if not improved:
                # Stagnation: try a larger perturbation
                if len(S) > 5:
                    k = 3
                    indices = rng.sample(range(len(S)), k)
                    indices.sort(reverse=True)
                    for idx in indices:
                        Sset.remove(S[idx])
                        S.pop(idx)
                    perm = list(V)
                    rng.shuffle(perm)
                    S_new = greedy_add(V, perm, S_initial=S)
                    if len(S_new) > best_size:
                        S = S_new
                        Sset = set(S)
                        best_size = len(S)
                        best_S = list(S)
                        available = [x for x in V if x not in Sset]
                    else:
                        S = list(best_S)
                        Sset = set(S)
                        available = [x for x in V if x not in Sset]
                else:
                    break

        return best_S

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        elapsed = time.time() - start
        remaining_total = DEADLINE - elapsed
        
        if remaining_total < 5:
            # Fallback
            S_fallback = greedy_add(V, sorted(V, reverse=True))
            results.append(sorted(S_fallback))
            continue
        
        # Allocate time
        instances_left = 3 - len(results)
        time_per_instance = remaining_total / instances_left - 2
        
        best_S = []
        best_size = 0
        rng = random.Random(seed + 42)
        
        # Strategy 1: Multiple Greedy Restarts
        t_greedy = time_per_instance * 0.4
        t_start = time.time()
        n_trials = max(20, int(t_greedy / 0.02))
        
        for i in range(n_trials):
            if time.time() - t_start > t_greedy:
                break
            perm = list(V)
            rng.shuffle(perm)
            S_trial = greedy_add(V, perm)
            if len(S_trial) > best_size:
                best_size = len(S_trial)
                best_S = S_trial
        
        # Also try specific orders
        S_desc = greedy_add(V, sorted(V, reverse=True))
        if len(S_desc) > best_size:
            best_size = len(S_desc)
            best_S = S_desc
            
        S_asc = greedy_add(V, sorted(V))
        if len(S_asc) > best_size:
            best_size = len(S_asc)
            best_S = S_asc
            
        # Strategy 2: Local Search
        t_ls = time_per_instance * 0.6
        t_start_ls = time.time()
        S_final = local_search(best_S, V, t_ls, time.time(), rng)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)
            
        # Final Safety Check
        if not is_sum_free(best_S):
            # Reconstruct from scratch to ensure validity
            # Use a very conservative greedy
            S_safe = []
            Sset = set()
            for x in sorted(V, reverse=True):
                valid = True
                for a in S_safe:
                    if x + a in Sset or x - a in Sset:
                        valid = False
                        break
                if valid:
                    S_safe.append(x)
                    Sset.add(x)
            best_S = S_safe
            
        results.append(sorted(best_S))
        
    return results
