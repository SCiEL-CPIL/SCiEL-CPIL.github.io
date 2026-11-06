import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    # Precompute all triple sums from 1..192
    # For fast checking: triple_sum_set = set of all sums of 3 distinct numbers from 1..192
    # Actually, we only need triple sums from the selected set, which changes.
    # But we can precompute pair sums for faster lookup.

    # Precompute pair sums for all pairs in 1..192
    # pair_sum[a][b] = a + b
    # This is just addition, no need to precompute.

    # Better: precompute for each a in 1..192, the set of residues (b+c) % a
    # for all b, c in 1..192, b != c, b != a, c != a.
    # This is too large. Let's just do direct checks with early termination.

    def can_add(selected, candidate):
        """Check if candidate can be added to selected (list of ints)."""
        n = len(selected)
        
        # Check 1: no triple from selected has sum divisible by candidate
        for i in range(n):
            a = selected[i]
            for j in range(i+1, n):
                b = selected[j]
                for k in range(j+1, n):
                    c = selected[k]
                    if (a + b + c) % candidate == 0:
                        return False
        
        # Check 2: for each a in selected, no pair (b,c) from selected\{a} 
        #           has (b+c+candidate) % a == 0
        for i in range(n):
            a = selected[i]
            for j in range(n):
                if j == i:
                    continue
                b = selected[j]
                for k in range(j+1, n):
                    if k == i:
                        continue
                    c = selected[k]
                    if (b + c + candidate) % a == 0:
                        return False
        
        return True

    def can_add_fast(selected, candidate, triple_sums, pair_sums_mod):
        """Check if candidate can be added using precomputed state."""
        # Check 1: No triple sum divisible by candidate
        for s in triple_sums:
            if s % candidate == 0:
                return False
        
        # Check 2: For each a in selected, no pair sum (b+c) from selected\{a}
        #          has (b+c+candidate) % a == 0
        #          => (b+c) % a == (-candidate) % a
        for a in selected:
            target = (-candidate) % a
            if target in pair_sums_mod[a]:
                return False
        
        return True

    def add_element(selected, candidate, triple_sums, pair_sums_mod):
        """Add candidate to state. O(N^2)."""
        n = len(selected)
        
        # Update triple_sums: add triples (candidate, b, c)
        for i in range(n):
            b = selected[i]
            for j in range(i+1, n):
                c = selected[j]
                triple_sums.add(candidate + b + c)
                
        # Update pair_sums_mod for existing a
        for a in selected:
            for b in selected:
                if a == b:
                    continue
                pair_sums_mod[a].add((candidate + b) % a)
                
        # Initialize pair_sums_mod for candidate
        new_pairs = set()
        for i in range(n):
            b = selected[i]
            for j in range(i+1, n):
                c = selected[j]
                new_pairs.add((b+c) % candidate)
        pair_sums_mod[candidate] = new_pairs
        
        selected.append(candidate)

    def build_state(selected):
        """Rebuild state from scratch. O(N^3)."""
        n = len(selected)
        triple_sums = set()
        pair_sums_mod = {a: set() for a in selected}
        
        for i in range(n):
            a = selected[i]
            for j in range(i+1, n):
                b = selected[j]
                for k in range(j+1, n):
                    c = selected[k]
                    triple_sums.add(a+b+c)
        
        for i in range(n):
            a = selected[i]
            for j in range(n):
                if j == i:
                    continue
                b = selected[j]
                for k in range(j+1, n):
                    if k == i:
                        continue
                    c = selected[k]
                    pair_sums_mod[a].add((b+c) % a)
                    
        return triple_sums, pair_sums_mod

    # Main Search
    best_set = []
    best_len = 0
    
    # Try multiple random orderings
    for trial in range(80):
        if time.time() > deadline - 10:
            break
            
        random.seed(12345 + trial)
        order = list(range(192, 0, -1))
        # Perturb order to escape local minima
        if trial > 0:
            random.shuffle(order)
            # Sort with noise to keep large numbers somewhat prioritized
            order.sort(key=lambda x: -x + random.uniform(-20, 20))
            
        selected = []
        triple_sums = set()
        pair_sums_mod = {}
        
        for x in order:
            if len(selected) >= 48:
                break
            if can_add_fast(selected, x, triple_sums, pair_sums_mod):
                add_element(selected, x, triple_sums, pair_sums_mod)
                
        if len(selected) > best_len:
            best_set = selected[:]
            best_len = len(selected)
            
        if best_len >= 48:
            break
            
    # Local Search: 1-out-2-in swaps
    if best_len < 48 and time.time() < deadline - 5:
        current = best_set[:]
        current_len = best_len
        improved = True
        while improved and current_len < 48 and time.time() < deadline - 5:
            improved = False
            # Rebuild state for current
            triple_sums, pair_sums_mod = build_state(current)
            
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    break
                a = current[i]
                # Remove a
                current_without_a = [v for j, v in enumerate(current) if j != i]
                # Rebuild state for current_without_a
                ts_wo_a, psm_wo_a = build_state(current_without_a)
                
                # Try to add b
                for b in range(192, 0, -1):
                    if b in current_without_a:
                        continue
                    if not can_add_fast(current_without_a, b, ts_wo_a, psm_wo_a):
                        continue
                    
                    # Tentatively add b
                    cur_b = current_without_a + [b]
                    ts_b, psm_b = build_state(cur_b)
                    
                    # Try to add c
                    for c in range(192, 0, -1):
                        if c in cur_b:
                            continue
                        if can_add_fast(cur_b, c, ts_b, psm_b):
                            new_set = cur_b + [c]
                            if len(new_set) > current_len:
                                current = new_set
                                current_len = len(current)
                                best_set = current[:]
                                best_len = current_len
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if current_len >= 48:
                break

    # Final polish: try to add more elements if we have room
    if best_len < 48 and time.time() < deadline - 2:
        current = best_set[:]
        triple_sums, pair_sums_mod = build_state(current)
        for x in range(192, 0, -1):
            if time.time() > deadline - 2:
                break
            if len(current) >= 48:
                break
            if x in current:
                continue
            if can_add_fast(current, x, triple_sums, pair_sums_mod):
                add_element(current, x, triple_sums, pair_sums_mod)
        if len(current) > best_len:
            best_set = current[:]
            best_len = len(current)

    return best_set[:48]

# EVOLVE-BLOCK-END
