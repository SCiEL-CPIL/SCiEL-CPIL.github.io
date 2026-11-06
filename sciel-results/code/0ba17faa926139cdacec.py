"""Sidon set construction via simulated annealing with kick moves."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s limit
    
    # Prescribed elements
    base = [1, 2, 5, 11]
    base_set = set(base)
    
    def compute_sum_conflicts(elements):
        """Return the set of all pairwise sums (a+b where a<=b) and count duplicates."""
        sums = {}
        for i in range(len(elements)):
            for j in range(i, len(elements)):
                s = elements[i] + elements[j]
                sums[s] = sums.get(s, 0) + 1
        # Count how many sums have multiplicity > 1
        conflicts = 0
        for s, count in sums.items():
            if count > 1:
                conflicts += count - 1
        return conflicts, sums
    
    def get_conflict_count_fast(element, elements, sums_dict):
        """Count conflicts if we add element to elements with given sums_dict."""
        conflicts = 0
        for x in elements:
            s = element + x
            if s in sums_dict:
                conflicts += 1
        ds = 2 * element
        if ds in sums_dict:
            conflicts += 1
        return conflicts
    
    # Initial greedy construction
    def greedy_init():
        elems = list(base)
        sums_dict = {}
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s = elems[i] + elems[j]
                sums_dict[s] = sums_dict.get(s, 0) + 1
        
        for x in range(12, 285):
            if time.time() > deadline:
                break
            if x in elems:
                continue
            if get_conflict_count_fast(x, elems, sums_dict) == 0:
                elems.append(x)
                for y in elems:
                    s = x + y
                    sums_dict[s] = sums_dict.get(s, 0) + 1
                s = 2 * x
                sums_dict[s] = sums_dict.get(s, 0) + 1
        
        return elems, sums_dict
    
    # Simulated annealing
    def simulated_annealing(initial_elems, initial_sums_dict):
        elems = list(initial_elems)
        sums_dict = dict(initial_sums_dict)
        conflicts, _ = compute_sum_conflicts(elems)
        
        best_elems = list(elems)
        best_conflicts = conflicts
        best_size = len(elems)
        
        T = 50.0
        T_min = 0.1
        cooling_rate = 0.9995
        max_steps = 500000
        
        rng = random.Random(42)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Choose move type
            r = rng.random()
            
            if r < 0.3:
                # Add a random element not in set
                available = [x for x in range(1, 285) if x not in set(elems)]
                if not available:
                    continue
                new_elem = rng.choice(available)
                new_elems = elems + [new_elem]
                
                # Compute new conflicts
                new_sums_dict = dict(sums_dict)
                added_conflicts = 0
                for x in elems:
                    s = new_elem + x
                    if s in new_sums_dict:
                        added_conflicts += 1
                    new_sums_dict[s] = new_sums_dict.get(s, 0) + 1
                ds = 2 * new_elem
                if ds in new_sums_dict:
                    added_conflicts += 1
                new_sums_dict[ds] = new_sums_dict.get(ds, 0) + 1
                
                new_conflicts = conflicts + added_conflicts
                delta = new_conflicts - conflicts
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    elems = new_elems
                    sums_dict = new_sums_dict
                    conflicts = new_conflicts
                    if conflicts < best_conflicts or (conflicts == best_conflicts and len(elems) > best_size):
                        best_conflicts = conflicts
                        best_size = len(elems)
                        best_elems = list(elems)
                        
            elif r < 0.6:
                # Remove a random non-base element
                removable = [e for e in elems if e not in base_set]
                if not removable:
                    continue
                remove_elem = rng.choice(removable)
                new_elems = [e for e in elems if e != remove_elem]
                
                # Recompute conflicts
                new_conflicts, new_sums_dict = compute_sum_conflicts(new_elems)
                delta = new_conflicts - conflicts
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    elems = new_elems
                    sums_dict = new_sums_dict
                    conflicts = new_conflicts
                    if conflicts < best_conflicts or (conflicts == best_conflicts and len(elems) > best_size):
                        best_conflicts = conflicts
                        best_size = len(elems)
                        best_elems = list(elems)
                        
            elif r < 0.8:
                # Swap: remove one non-base, add one new
                removable = [e for e in elems if e not in base_set]
                if not removable:
                    continue
                remove_elem = rng.choice(removable)
                available = [x for x in range(1, 285) if x not in set(elems)]
                if not available:
                    continue
                add_elem = rng.choice(available)
                
                new_elems = [e for e in elems if e != remove_elem] + [add_elem]
                new_conflicts, new_sums_dict = compute_sum_conflicts(new_elems)
                delta = new_conflicts - conflicts
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    elems = new_elems
                    sums_dict = new_sums_dict
                    conflicts = new_conflicts
                    if conflicts < best_conflicts or (conflicts == best_conflicts and len(elems) > best_size):
                        best_conflicts = conflicts
                        best_size = len(elems)
                        best_elems = list(elems)
                        
            else:
                # Kick move: remove two non-base elements, insert two new ones
                removable = [e for e in elems if e not in base_set]
                if len(removable) < 2:
                    continue
                remove_elems = rng.sample(removable, 2)
                remove_set = set(remove_elems)
                new_elems_no_removal = [e for e in elems if e not in remove_set]
                
                available = [x for x in range(1, 285) if x not in set(new_elems_no_removal)]
                if len(available) < 2:
                    continue
                add_elems = rng.sample(available, 2)
                
                new_elems = new_elems_no_removal + add_elems
                new_conflicts, new_sums_dict = compute_sum_conflicts(new_elems)
                delta = new_conflicts - conflicts
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    elems = new_elems
                    sums_dict = new_sums_dict
                    conflicts = new_conflicts
                    if conflicts < best_conflicts or (conflicts == best_conflicts and len(elems) > best_size):
                        best_conflicts = conflicts
                        best_size = len(elems)
                        best_elems = list(elems)
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
            
            # If we found a valid Sidon set of size >= 20, we can stop early
            if best_conflicts == 0 and best_size >= 20:
                break
            
            # If we found a valid Sidon set, try to improve
            if best_conflicts == 0 and best_size >= 18:
                # Try a few more steps at low temperature
                if T < 1.0:
                    break
        
        return best_elems
    
    # Run greedy initialization
    greedy_elems, greedy_sums = greedy_init()
    
    # Run simulated annealing
    sa_elems = simulated_annealing(greedy_elems, greedy_sums)
    
    # Verify the result
    def is_valid_sidon(elems):
        if not all(1 <= e <= 284 for e in elems):
            return False
        if not all(b in elems for b in base):
            return False
        if len(elems) != len(set(elems)):
            return False
        conflicts, _ = compute_sum_conflicts(elems)
        return conflicts == 0
    
    if is_valid_sidon(sa_elems):
        return sa_elems
    
    # Fallback: return greedy solution if valid, else base
    if is_valid_sidon(greedy_elems):
        return greedy_elems
    
    return list(base)
# EVOLVE-BLOCK-END
