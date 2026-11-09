"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0
    N = 4096
    MAX_SUM = 3 * N  # max possible triple sum
    
    def check_add_fast(x, selected, triple_sums_set):
        """Check if x can be added. Returns (ok, new_sums_list) or (False, None)."""
        n = len(selected)
        seen_new = set()
        
        # Triples with one x: x + a_i + a_j for i <= j
        for i in range(n):
            for j in range(i, n):
                s = x + selected[i] + selected[j]
                if s in seen_new or s in triple_sums_set:
                    return False, None
                seen_new.add(s)
        
        # Triples with two x's: 2*x + a_i
        for i in range(n):
            s = 2 * x + selected[i]
            if s in seen_new or s in triple_sums_set:
                return False, None
            seen_new.add(s)
        
        # Triple with three x's
        s = 3 * x
        if s in seen_new or s in triple_sums_set:
            return False, None
        seen_new.add(s)
        
        return True, list(seen_new)
    
    def build_triple_sums(selected):
        """Build the set of all triple sums for a given list."""
        sums = set()
        n = len(selected)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums.add(selected[i] + selected[j] + selected[k])
        return sums
    
    def greedy_fill(selected, triple_sums_set, start_from=1, end_to=N, deadline_check=None):
        """Greedy fill: try to add numbers from start_from to end_to."""
        improved = list(selected)
        improved_set = set(improved)
        ts = set(triple_sums_set)
        
        for x in range(start_from, end_to + 1):
            if deadline_check and time.time() > deadline_check:
                break
            if x in improved_set:
                continue
            ok, new_sums = check_add_fast(x, improved, ts)
            if ok:
                improved.append(x)
                improved_set.add(x)
                ts.update(new_sums)
        
        return improved, ts
    
    # Phase 1: Standard greedy from 1
    result = []
    triple_sums = set()
    for x in range(1, N + 1):
        if time.time() > DEADLINE - 30:
            break
        ok, new_sums = check_add_fast(x, result, triple_sums)
        if ok:
            result.append(x)
            triple_sums.update(new_sums)
    
    current_size = len(result)
    
    # Phase 2: Try greedy from various offsets
    if time.time() < DEADLINE - 25:
        for offset in [2, 3, 5, 8, 13]:
            if time.time() > DEADLINE - 25:
                break
            alt = []
            alt_ts = set()
            for x in range(offset, N + 1):
                if time.time() > DEADLINE - 25:
                    break
                ok, new_sums = check_add_fast(x, alt, alt_ts)
                if ok:
                    alt.append(x)
                    alt_ts.update(new_sums)
            if len(alt) > current_size:
                result = alt
                triple_sums = alt_ts
                current_size = len(result)
    
    # Phase 3: Local improvement - remove one element, try to add better ones
    if time.time() < DEADLINE - 15:
        for round_num in range(5):
            if time.time() > DEADLINE - 15:
                break
            improved = False
            for i in range(len(result)):
                if time.time() > DEADLINE - 15:
                    break
                # Remove result[i]
                removed = result[i]
                temp = [x for x in result if x != removed]
                temp_ts = build_triple_sums(temp)
                
                # Try to add numbers from high to low (to maximize spread)
                improved_list, improved_ts = greedy_fill(
                    temp, temp_ts, start_from=1, end_to=N,
                    deadline_check=DEADLINE - 15
                )
                
                if len(improved_list) > current_size:
                    result = improved_list
                    triple_sums = improved_ts
                    current_size = len(result)
                    improved = True
                    break
            
            if not improved:
                break
    
    # Phase 4: Random restarts with partial seeds
    if time.time() < DEADLINE - 10:
        rng = random.Random(12345)
        best_size = current_size
        best_result = result
        
        for restart in range(30):
            if time.time() > DEADLINE - 10:
                break
            
            # Create a random subset seed from current result
            if len(result) > 8:
                seed_size = min(len(result), rng.randint(4, max(4, len(result) - 3)))
                seed = rng.sample(result, seed_size)
                seed.sort()
            else:
                seed = list(result)
            
            seed_set = set(seed)
            seed_ts = build_triple_sums(seed)
            
            # Greedy fill from seed
            improved_list, improved_ts = greedy_fill(
                seed, seed_ts, start_from=1, end_to=N,
                deadline_check=DEADLINE - 10
            )
            
            if len(improved_list) > best_size:
                best_size = len(improved_list)
                best_result = improved_list
    
    if len(best_result) > current_size:
        result = best_result
        current_size = len(result)
    
    # Phase 5: Final targeted improvement - try to add one more by removing one
    if time.time() < DEADLINE - 5:
        for i in range(len(result)):
            if time.time() > DEADLINE - 5:
                break
            removed = result[i]
            temp = [x for x in result if x != removed]
            temp_ts = build_triple_sums(temp)
            
            # Try to find ANY number that fits
            found = False
            for x in range(N, 0, -1):
                if time.time() > DEADLINE - 5:
                    break
                if x in temp:
                    continue
                ok, new_sums = check_add_fast(x, temp, temp_ts)
                if ok:
                    temp.append(x)
                    temp_ts.update(new_sums)
                    found = True
                    break
            
            if found and len(temp) > current_size:
                result = temp
                triple_sums = temp_ts
                current_size = len(result)
    
    return result
# EVOLVE-BLOCK-END
