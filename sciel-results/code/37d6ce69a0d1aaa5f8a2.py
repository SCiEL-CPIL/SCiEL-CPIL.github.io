"""Improved program using randomized greedy with constraint checking."""
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    total_triples = len(all_triples)  # 560
    
    best_result = None
    best_count = 0
    
    def check_valid(triple_set):
        """Check if a set of triples contains no K(2,2,2)."""
        triple_set = set(triple_set)
        # For each triple, check if it can be part of a K(2,2,2)
        for t in triple_set:
            a, b, c = t
            # Find 3 other vertices d, e, f
            remaining = [v for v in range(n) if v not in (a, b, c)]
            for d, e, f in itertools.combinations(remaining, 3):
                # Pair d,e,f with a,b,c in all 6 ways
                # Each way: one of d,e,f pairs with a, one with b, one with c
                for perm in itertools.permutations([d, e, f]):
                    dd, ee, ff = perm
                    # Pairs: {a,dd}, {b,ee}, {c,ff}
                    # Check all 8 transversals
                    valid = True
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                v1 = a if i == 0 else dd
                                v2 = b if j == 0 else ee
                                v3 = c if k == 0 else ff
                                t_tuple = tuple(sorted((v1, v2, v3)))
                                if t_tuple not in triple_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        return False
        return True
    
    def greedy_randomized(seed):
        """Greedy with random ordering."""
        rng = random.Random(seed)
        order = all_triples[:]
        rng.shuffle(order)
        
        current = set()
        triple_set = set()
        
        for t in order:
            # Check if adding t creates a K(2,2,2)
            # Only need to check K(2,2,2)s that include t
            a, b, c = t
            remaining = [v for v in range(n) if v not in (a, b, c)]
            conflict = False
            for d, e, f in itertools.combinations(remaining, 3):
                for perm in itertools.permutations([d, e, f]):
                    dd, ee, ff = perm
                    # Check all 8 transversals of pairs {a,dd}, {b,ee}, {c,ff}
                    all_present = True
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                v1 = a if i == 0 else dd
                                v2 = b if j == 0 else ee
                                v3 = c if k == 0 else ff
                                t_tuple = tuple(sorted((v1, v2, v3)))
                                if t == t_tuple:
                                    continue  # This is the new triple, assume present
                                if t_tuple not in triple_set:
                                    all_present = False
                                    break
                            if not all_present:
                                break
                        if not all_present:
                            break
                    if all_present:
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                triple_set.add(t)
                current.add(t)
        
        return [list(t) for t in current]
    
    # Try multiple random restarts
    seed = 42
    while time.time() < deadline:
        result = greedy_randomized(seed)
        count = len(result)
        if count > best_count:
            best_count = count
            best_result = result
        seed += 1
    
    if best_result is None:
        # Fallback: return seed program's result
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return best_result
