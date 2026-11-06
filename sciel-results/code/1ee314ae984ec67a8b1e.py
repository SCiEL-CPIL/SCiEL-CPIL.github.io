"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True
    
    def count_violations(S):
        S_set = set(S)
        violations = 0
        for a in S:
            for b in S:
                if a + b in S_set:
                    violations += 1
        return violations
    
    def greedy_remove(S):
        S = list(S)
        S_set = set(S)
        changed = True
        while changed:
            changed = False
            # Count how many violations each element is involved in
            violation_count = {x: 0 for x in S}
            for a in S:
                for b in S:
                    c = a + b
                    if c in S_set:
                        violation_count[a] = violation_count.get(a, 0) + 1
                        violation_count[b] = violation_count.get(b, 0) + 1
                        violation_count[c] = violation_count.get(c, 0) + 1
            # Remove the element with the most violations
            if S:
                max_viol = max(violation_count.values())
                if max_viol > 0:
                    worst = [x for x in S if violation_count[x] == max_viol]
                    x_to_remove = worst[0]
                    S.remove(x_to_remove)
                    S_set.remove(x_to_remove)
                    changed = True
        return S
    
    def local_search(S, max_iters=5000):
        S = list(S)
        S_set = set(S)
        best = len(S)
        rng = random.Random(42)
        
        for _ in range(max_iters):
            # Try to add a random removed element
            all_elements = set(instance_cache)
            removed = all_elements - S_set
            if not removed:
                break
            x = rng.choice(list(removed))
            # Check if adding x creates violations
            creates_violation = False
            for a in S:
                if a + x in S_set or x + a in S_set:
                    creates_violation = True
                    break
                if 2*x == a:  # x + x = a
                    creates_violation = True
                    break
            # Also check if x + x is in S
            if 2*x in S_set:
                creates_violation = True
            
            if not creates_violation:
                S.append(x)
                S_set.add(x)
                if len(S) > best:
                    best = len(S)
            else:
                # Try swapping: remove one element and add x
                # Find elements that conflict with x
                conflicts = set()
                for a in S:
                    if a + x in S_set:
                        conflicts.add(a)
                    if x + a in S_set:
                        conflicts.add(a)
                    if 2*x == a:
                        conflicts.add(a)
                if 2*x in S_set:
                    conflicts.add(2*x)
                
                if conflicts:
                    # Try removing one conflict and adding x
                    for c in list(conflicts):
                        # Check if removing c and adding x keeps things sum-free
                        new_S_set = S_set - {c} | {x}
                        # Quick check: is the new set sum-free?
                        valid = True
                        for a in new_S_set:
                            for b in new_S_set:
                                if a + b in new_S_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            S = list(new_S_set)
                            S_set = new_S_set
                            if len(S) > best:
                                best = len(S)
                            break
        
        return S
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        global instance_cache
        instance_cache = S
        
        # Greedy removal to get a sum-free set
        S_greedy = greedy_remove(S)
        
        # Local search to improve
        S_local = local_search(S_greedy)
        
        # Verify it's sum-free
        if not is_sum_free(S_local):
            # Fallback to greedy
            S_local = S_greedy
        
        results.append(S_local)
    
    return results
# EVOLVE-BLOCK-END
