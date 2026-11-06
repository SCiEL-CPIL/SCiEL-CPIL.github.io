import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    best_set = []
    best_size = 0
    
    def is_valid(S):
        """Check if S is valid: no element divides sum of any 3 distinct others."""
        n = len(S)
        if n < 4:
            return True
        
        S_set = set(S)
        
        # Build map: sum -> list of triples (as index tuples for efficiency)
        sums_map = {}
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    s = S[i] + S[j] + S[k]
                    if s not in sums_map:
                        sums_map[s] = []
                    sums_map[s].append((i, j, k))
        
        # For each a in S, check if any multiple of a is a triple sum not containing a
        for idx_a, a in enumerate(S):
            # Multiples of a that could be sums of 3 positive ints from S
            # Min possible sum: 3 * min(S\{a}), Max: 3 * max(S)
            min_other = min(x for x in S if x != a) if n > 1 else 1
            max_possible = 3 * max(S)
            
            k_start = max(1, (3 * min_other + a - 1) // a)  # ceil(3*min_other / a)
            k_end = max_possible // a
            
            for k in range(k_start, k_end + 1):
                target = k * a
                if target in sums_map:
                    for (i, j, k_idx) in sums_map[target]:
                        if idx_a not in (i, j, k_idx):
                            return False
        return True
    
    def find_conflict(S):
        """Find a conflict. Returns (a_idx, triple_indices) or None."""
        n = len(S)
        if n < 4:
            return None
        
        sums_map = {}
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    s = S[i] + S[j] + S[k]
                    if s not in sums_map:
                        sums_map[s] = []
                    sums_map[s].append((i, j, k))
        
        for idx_a, a in enumerate(S):
            min_other = min(x for x in S if x != a) if n > 1 else 1
            max_possible = 3 * max(S)
            k_start = max(1, (3 * min_other + a - 1) // a)
            k_end = max_possible // a
            
            for k in range(k_start, k_end + 1):
                target = k * a
                if target in sums_map:
                    for (i, j, k_idx) in sums_map[target]:
                        if idx_a not in (i, j, k_idx):
                            return (idx_a, (i, j, k_idx))
        return None
    
    def greedy_build(pool, order='desc'):
        """Greedily build a valid set from pool."""
        if order == 'desc':
            candidates = sorted(pool, reverse=True)
        elif order == 'asc':
            candidates = sorted(pool)
        else:
            candidates = pool[:]
            random.shuffle(candidates)
        
        S = []
        for x in candidates:
            if time.time() > deadline:
                break
            new_S = S + [x]
            if is_valid(new_S):
                S.append(x)
            if len(S) >= 48:
                break
        return S
    
    def local_search(S, pool, max_swaps=200):
        """Try to improve S by swapping elements."""
        S = S[:]
        S_set = set(S)
        pool_set = set(pool)
        
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            
            # Try random swaps
            for _ in range(50):
                if time.time() > deadline:
                    break
                
                # Pick a random element from S to remove
                idx = random.randint(0, len(S)-1)
                remove_elem = S[idx]
                
                # Pick a random element from pool not in S to add
                unused = [x for x in pool if x not in S_set]
                if not unused:
                    break
                add_elem = random.choice(unused)
                
                if add_elem == remove_elem:
                    continue
                
                new_S = [x for i, x in enumerate(S) if i != idx] + [add_elem]
                if is_valid(new_S):
                    S = new_S
                    S_set = set(S)
                    break
        
        return S
    
    # Strategy 1: Greedy with different orderings
    pools = [
        list(range(128, 193)),  # 65 numbers - largest, safest
        list(range(112, 193)),  # 81 numbers
        list(range(96, 193)),   # 97 numbers
        list(range(80, 193)),   # 113 numbers
        list(range(64, 193)),   # 129 numbers
        list(range(1, 193)),    # All 192
    ]
    
    # Try greedy with descending order on largest pool first
    for pool in pools:
        if time.time() > deadline:
            break
        
        # Try multiple orderings
        for order in ['desc', 'asc', 'shuffled1', 'shuffled2']:
            if time.time() > deadline:
                break
            
            if order == 'shuffled1':
                random.seed(42)
            elif order == 'shuffled2':
                random.seed(123)
            
            S = greedy_build(pool, order)
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            
            if len(S) >= 48:
                return S
        
        # Try local search to improve
        if len(best_set) >= 40 and len(best_set) < 48:
            if time.time() < deadline - 10:
                improved = local_search(best_set, pool, max_swaps=100)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_set = improved
                if len(best_set) >= 48:
                    return best_set
    
    # Strategy 2: Random restarts with repair
    if time.time() < deadline - 5:
        for attempt in range(50):
            if time.time() > deadline - 2:
                break
            
            # Random sample from a large pool
            pool = list(range(96, 193))
            S = random.sample(pool, 48)
            
            if is_valid(S):
                return S
            
            # Try to repair
            for _ in range(20):
                if time.time() > deadline - 2:
                    break
                
                conflict = find_conflict(S)
                if conflict is None:
                    return S
                
                a_idx, (i, j, k_idx) = conflict
                a = S[a_idx]
                
                # Remove one of the summands and replace with a random unused
                unused = [x for x in pool if x not in set(S)]
                if not unused:
                    break
                
                # Try removing each summand
                for rem_idx in [i, j, k_idx]:
                    new_S = [x for idx, x in enumerate(S) if idx != rem_idx]
                    new_S.append(random.choice(unused))
                    if is_valid(new_S):
                        S = new_S
                        break
                else:
                    # Try removing a itself
                    new_S = [x for idx, x in enumerate(S) if idx != a_idx]
                    new_S.append(random.choice(unused))
                    if is_valid(new_S):
                        S = new_S
    
    return best_set
