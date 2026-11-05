"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def find_violations(S):
        """Return list of (a, b, c) such that a+b=c, all in S."""
        S_set = set(S)
        viol = []
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i, n):
                b = S[j]
                c = a + b
                if c in S_set:
                    viol.append((a, b, c))
        return viol

    def can_add(S_set, x):
        """Check if x can be added to sum-free set S without violation."""
        for a in S_set:
            if a + x in S_set:
                return False
            if x - a in S_set:
                return False
        if 2 * x in S_set:
            return False
        return True

    def greedy_add(S, V, order=None):
        """Add elements from V to S in the given order (or sorted by magnitude descending)."""
        S_set = set(S)
        if order is None:
            order = sorted(V, key=lambda x: abs(x), reverse=True)
        for x in order:
            if x not in S_set:
                if can_add(S_set, x):
                    S_set.add(x)
        return list(S_set)

    def local_search(S, V, max_steps=200):
        """Perform local search: try swaps and additions to improve size."""
        S_set = set(S)
        V_set = set(V)
        
        # First, do a greedy add pass
        S_set = set(greedy_add(list(S_set), V_set - S_set))
        
        improved = True
        steps = 0
        while improved and steps < max_steps:
            improved = False
            steps += 1
            
            # Try to add any element from V not in S
            candidates = [x for x in V_set if x not in S_set]
            for x in candidates:
                if can_add(S_set, x):
                    S_set.add(x)
                    improved = True
                    break
            
            # Try swaps: remove a, add b
            if not improved:
                # For each element in S, try removing it and adding something else
                # This is expensive, so limit it
                S_list = list(S_set)
                random.shuffle(S_list)
                for a in S_list[:20]:  # Limit to 20 elements to try
                    S_set.discard(a)
                    # Now try to add something
                    candidates = [x for x in V_set if x not in S_set]
                    random.shuffle(candidates)
                    added = False
                    for x in candidates[:50]:  # Limit to 50 candidates
                        if can_add(S_set, x):
                            S_set.add(x)
                            added = True
                            improved = True
                            break
                    if not added:
                        S_set.add(a)  # Put it back
                    if improved:
                        break
        
        return list(S_set)

    def repair_and_expand(S, V):
        """Improve S by repairing violations and adding safe elements."""
        S_list = list(S)
        S_set = set(S_list)
        
        # Phase 1: Remove elements to fix violations
        while True:
            viol = find_violations(S_list)
            if not viol:
                break
            
            # Count how many times each element appears in violations
            counts = Counter()
            for a, b, c in viol:
                counts[a] += 1
                counts[b] += 1
                counts[c] += 1
            
            # Remove the element with the highest count
            if counts:
                worst = max(counts, key=counts.get)
                if worst in S_set:
                    S_list.remove(worst)
                    S_set.discard(worst)
                else:
                    break
            else:
                break

        # Phase 2: Add safe elements using greedy
        S_list = greedy_add(S_list, V)
        
        # Phase 3: Local search
        S_list = local_search(S_list, V, max_steps=100)
        
        return S_list

    def search_instance(elements, seed_offset=0):
        """Search for a large sum-free subset of elements."""
        rng = random.Random(seed_offset)
        best = []
        best_size = 0
        
        # Generate multiple initial candidates
        candidates = []
        
        # 1. Odds
        odd = [x for x in elements if x % 2 != 0]
        candidates.append(odd)
        
        # 2. Largest values
        sorted_desc = sorted(elements, reverse=True)
        for frac in [0.3, 0.4, 0.5]:
            k = int(len(elements) * frac)
            candidates.append(sorted_desc[:k])
        
        # 3. Random samples
        for _ in range(5):
            k = int(len(elements) * 0.5)
            candidates.append(rng.sample(elements, k))
            
        # 4. Positive numbers
        pos = [x for x in elements if x > 0]
        candidates.append(pos)
        
        # 5. Negative numbers
        neg = [x for x in elements if x < 0]
        candidates.append(neg)
        
        # 6. Large magnitude first
        sorted_mag = sorted(elements, key=lambda x: abs(x), reverse=True)
        for frac in [0.3, 0.4, 0.5]:
            k = int(len(elements) * frac)
            candidates.append(sorted_mag[:k])
        
        # Process each candidate
        for cand in candidates:
            S = repair_and_expand(cand, elements)
            if len(S) > best_size:
                best_size = len(S)
                best = S
        
        # Additional random restarts if time permits
        start_restarts = time.time()
        restart_count = 0
        while time.time() - start_restarts < 8:
            k = int(len(elements) * rng.uniform(0.3, 0.8))
            cand = rng.sample(elements, k)
            S = repair_and_expand(cand, elements)
            if len(S) > best_size:
                best_size = len(S)
                best = S
            restart_count += 1
            if restart_count > 50:
                break
                
        return best

    # Generate instances
    instances = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        instances.append(instance(size, seed))
    
    results = []
    for i, elems in enumerate(instances):
        if time.time() - start_time > DEADLINE:
            # Fallback: odd numbers
            results.append([x for x in elems if x % 2 != 0])
            continue
        result = search_instance(elems, seed_offset=801 + i)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
