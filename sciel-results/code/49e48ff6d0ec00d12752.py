"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    best_n = 9
    best_colors = None
    
    # Precompute triples for various n values
    def get_triples(n):
        return list(combinations(range(n), 3))
    
    # Precompute 4-sets and 5-sets
    def get_k_sets(n, k):
        return list(combinations(range(n), k))
    
    # For a given n, precompute which triples are in which 4-sets and 5-sets
    def precompute_sets(n):
        triples = get_triples(n)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # For each 4-set, list the 4 triples in it
        four_sets = get_k_sets(n, 4)
        triples_in_4set = []
        for s in four_sets:
            tlist = []
            for i in range(4):
                for j in range(i+1, 4):
                    for k in range(j+1, 4):
                        tlist.append(triple_index[(s[i], s[j], s[k])])
            triples_in_4set.append(tlist)
        
        # For each 5-set, list the 10 triples in it
        five_sets = get_k_sets(n, 5)
        triples_in_5set = []
        for s in five_sets:
            tlist = []
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        tlist.append(triple_index[(s[i], s[j], s[k])])
            triples_in_5set.append(tlist)
        
        return triples, triple_index, triples_in_4set, triples_in_5set
    
    def is_valid(colors, triples_in_4set, triples_in_5set):
        """Check if coloring is valid (no red K4^3, no blue K5^3)."""
        # Check no 4-set has all 4 triples red
        for tlist in triples_in_4set:
            if all(colors[i] == 0 for i in tlist):
                return False
        # Check no 5-set has all 10 triples blue
        for tlist in triples_in_5set:
            if all(colors[i] == 1 for i in tlist):
                return False
        return True
    
    def count_violations(colors, triples_in_4set, triples_in_5set):
        """Count number of violations."""
        v = 0
        for tlist in triples_in_4set:
            if all(colors[i] == 0 for i in tlist):
                v += 1
        for tlist in triples_in_5set:
            if all(colors[i] == 1 for i in tlist):
                v += 1
        return v
    
    def random_coloring(n, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        m = len(get_triples(n))
        return [rng.randint(0, 1) for _ in range(m)]
    
    def local_search(n, initial_colors, max_iters=10000, deadline=None):
        """Perform local search to find a valid coloring."""
        triples, triple_index, triples_in_4set, triples_in_5set = precompute_sets(n)
        m = len(triples)
        colors = list(initial_colors)
        
        # Build adjacency: for each triple, which 4-sets and 5-sets contain it
        triple_to_4sets = defaultdict(list)
        for si, tlist in enumerate(triples_in_4set):
            for ti in tlist:
                triple_to_4sets[ti].append(si)
        
        triple_to_5sets = defaultdict(list)
        for si, tlist in enumerate(triples_in_5set):
            for ti in tlist:
                triple_to_5sets[ti].append(si)
        
        # Current violation count
        cur_viol = count_violations(colors, triples_in_4set, triples_in_5set)
        best_colors = list(colors)
        best_viol = cur_viol
        
        if best_viol == 0:
            return best_colors, 0
        
        no_improve = 0
        temperature = 1.0
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            if cur_viol == 0:
                return list(colors), 0
            
            # Pick a random triple to flip
            ti = random.randint(0, m - 1)
            old_color = colors[ti]
            new_color = 1 - old_color
            
            # Calculate change in violations
            delta_viol = 0
            
            # Check 4-sets containing this triple
            for si in triple_to_4sets[ti]:
                tlist = triples_in_4set[si]
                # Count reds before and after
                reds_before = sum(1 for i in tlist if colors[i] == 0)
                if old_color == 0:
                    reds_after = reds_before - 1
                else:
                    reds_after = reds_before + 1
                
                # Violation if all 4 are red
                viol_before = 1 if reds_before == 4 else 0
                viol_after = 1 if reds_after == 4 else 0
                delta_viol += viol_after - viol_before
            
            # Check 5-sets containing this triple
            for si in triple_to_5sets[ti]:
                tlist = triples_in_5set[si]
                # Count blues before and after
                blues_before = sum(1 for i in tlist if colors[i] == 1)
                if old_color == 1:
                    blues_after = blues_before - 1
                else:
                    blues_after = blues_before + 1
                
                # Violation if all 10 are blue
                viol_before = 1 if blues_before == 10 else 0
                viol_after = 1 if blues_after == 10 else 0
                delta_viol += viol_after - viol_before
            
            new_viol = cur_viol + delta_viol
            
            # Accept if better or equal, or with some probability if worse
            if new_viol <= cur_viol:
                colors[ti] = new_color
                cur_viol = new_viol
                if cur_viol < best_viol:
                    best_viol = cur_viol
                    best_colors = list(colors)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Simulated annealing: accept worse with some probability
                if random.random() < min(1.0, 1.0 / (1 + delta_viol * 0.5)):
                    colors[ti] = new_color
                    cur_viol = new_viol
                    no_improve += 1
                else:
                    no_improve += 1
            
            if no_improve > 2000:
                # Restart with a perturbation
                no_improve = 0
                # Flip a few random triples
                for _ in range(min(10, m // 10)):
                    ti2 = random.randint(0, m - 1)
                    colors[ti2] = 1 - colors[ti2]
                cur_viol = count_violations(colors, triples_in_4set, triples_in_5set)
        
        return best_colors, best_viol
    
    def solve_for_n(n, time_budget):
        """Try to find a valid coloring for n vertices."""
        end_time = time.time() + time_budget
        triples, triple_index, triples_in_4set, triples_in_5set = precompute_sets(n)
        m = len(triples)
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > end_time:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # Random
                init = random_coloring(n)
            elif attempt == 1:
                # Based on sum mod 3
                init = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
            elif attempt == 2:
                # Based on sum mod 4
                init = [0 if (a + b + c) % 4 < 2 else 1 for a, b, c in triples]
            elif attempt == 3:
                # Based on product mod 3
                init = [0 if ((a + 1) * (b + 1) * (c + 1)) % 3 == 0 else 1 for a, b, c in triples]
            elif attempt == 4:
                # Based on max mod 2
                init = [0 if max(a, b, c) % 2 == 0 else 1 for a, b, c in triples]
            elif attempt == 5:
                # Based on (a+b) mod 2
                init = [0 if (a + b) % 2 == 0 else 1 for a, b, c in triples]
            elif attempt == 6:
                # Based on (a+c) mod 3
                init = [0 if (a + c) % 3 == 0 else 1 for a, b, c in triples]
            else:
                # Fully random
                init = random_coloring(n, seed=attempt * 12345)
            
            result_colors, viol = local_search(n, init, max_iters=50000, deadline=end_time)
            
            if viol == 0:
                return result_colors
        
        return None
    
    # Try n values from large to small
    # Known: published construction has 34 points
    # Try from 34 down
    n_values_to_try = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9]
    
    # Allocate more time to larger n values
    total_time = 130
    # We'll try each n with a decreasing time budget
    time_per_n = {}
    remaining = total_time
    for n in n_values_to_try:
        if n >= 30:
            time_per_n[n] = remaining * 0.15
        elif n >= 20:
            time_per_n[n] = remaining * 0.10
        else:
            time_per_n[n] = remaining * 0.05
        remaining -= time_per_n[n]
        if remaining < 5:
            # Give remaining to the next few
            time_per_n[n] += remaining
            remaining = 0
    
    for n in n_values_to_try:
        if time.time() > deadline - 5:
            break
        
        if n <= best_n:
            break
        
        t_budget = time_per_n.get(n, 5)
        if t_budget < 1:
            t_budget = 1
        
        print(f"Trying n={n} with {t_budget:.1f}s budget...")
        result = solve_for_n(n, t_budget)
        
        if result is not None:
            # Verify
            triples, triple_index, triples_in_4set, triples_in_5set = precompute_sets(n)
            if is_valid(result, triples_in_4set, triples_in_5set):
                best_n = n
                best_colors = result
                print(f"Found valid coloring for n={n}!")
                # Continue trying larger? No, we go from large to small
                # But maybe we can try to improve? No, n is fixed per call
                break
        else:
            print(f"Failed to find valid coloring for n={n}")
    
    # If we didn't find anything better than 9, use the simple construction
    if best_colors is None:
        n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
