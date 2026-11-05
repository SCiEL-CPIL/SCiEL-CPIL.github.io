import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 180
    # Generate all forbidden triples and build conflict graph
    triples = []
    for a in range(1, N + 1):
        for d in range(1, a):
            if (a * a) % d == 0:
                b = a + d
                c = (a * b) // d
                if c <= N and c > b:
                    triples.append((a, b, c))
    
    if not triples:
        return list(range(1, N + 1))
    
    # Build conflict graph: edge between i,j if they appear together in some forbidden triple
    # We want maximum independent set (no two adjacent vertices both selected)
    # Equivalently, we want to select a subset S such that no forbidden triple has all 3 in S
    
    # Actually, the constraint is: no THREE DISTINCT elements a,b,c in S with 1/a=1/b+1/c
    # This is a 3-uniform hypergraph independent set problem, not 2-uniform
    # But we can still use hitting set / vertex cover on the hypergraph
    
    T = len(triples)
    elem_to_triples = defaultdict(list)
    for idx, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].append(idx)
    
    # Greedy hitting set with multiple strategies
    def run_greedy_random(seed):
        rng = random.Random(seed)
        uncovered = [True] * T
        R = set()
        
        while True:
            if time.time() > deadline:
                return R
            
            any_uncovered = False
            for i in range(T):
                if uncovered[i]:
                    any_uncovered = True
                    break
            if not any_uncovered:
                break
            
            candidates = set()
            for i in range(T):
                if uncovered[i]:
                    for x in triples[i]:
                        candidates.add(x)
                        
            counts = {x: 0 for x in candidates}
            for i in range(T):
                if uncovered[i]:
                    for x in triples[i]:
                        counts[x] += 1
                        
            max_count = max(counts.values())
            best_elems = [x for x, c in counts.items() if c == max_count]
            
            # Random pick with slight bias toward lower degree elements
            # to leave more flexibility
            best_elem = rng.choice(best_elems)
            
            R.add(best_elem)
            
            for idx in elem_to_triples[best_elem]:
                uncovered[idx] = False
                
        return R
    
    def local_search_remove(R):
        R = set(R)
        changed = True
        while changed:
            changed = False
            for x in list(R):
                needed = False
                for idx in elem_to_triples[x]:
                    covered = False
                    for y in triples[idx]:
                        if y != x and y in R:
                            covered = True
                            break
                    if not covered:
                        needed = True
                        break
                if not needed:
                    R.discard(x)
                    changed = True
        return R
    
    def swap_improve(R):
        """Try to swap elements to reduce size or maintain size with better structure."""
        R = set(R)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in list(R):
                # Try removing x and adding a different element y not in R
                R_without_x = R - {x}
                # Find triples that become uncovered
                uncovered_triples = []
                for idx in elem_to_triples[x]:
                    covered = False
                    for y in triples[idx]:
                        if y in R_without_x:
                            covered = True
                            break
                    if not covered:
                        uncovered_triples.append(idx)
                
                if not uncovered_triples:
                    R = R_without_x
                    improved = True
                    break
                
                # Find y not in R that covers most uncovered triples
                best_y = None
                best_count = -1
                for y in range(1, N + 1):
                    if y in R:
                        continue
                    count = 0
                    for idx in uncovered_triples:
                        if y in triples[idx]:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_y = y
                
                if best_y is not None and best_count > 0:
                    # Only swap if it helps (covers all uncovered or we can iterate)
                    if best_count == len(uncovered_triples):
                        R = R_without_x | {best_y}
                        improved = True
                        break
        
        return R
    
    best_R = None
    best_R_size = N + 1
    
    # Phase 1: Multiple greedy restarts
    num_restarts = 300
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
        R = run_greedy_random(i)
        R = local_search_remove(R)
        if len(R) < best_R_size:
            best_R_size = len(R)
            best_R = R
    
    # Phase 2: Local improvement on best solution
    if best_R is not None:
        best_R = swap_improve(best_R)
        best_R = local_search_remove(best_R)
        best_R_size = len(best_R)
    
    # Phase 3: More aggressive local search with random perturbations
    if best_R is not None and time.time() < deadline - 10:
        current_best = set(best_R)
        current_size = len(current_best)
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Perturb: remove 2-3 random elements, then repair
            perturbed = set(current_best)
            num_remove = min(3, len(perturbed))
            to_remove = random.sample(list(perturbed), num_remove)
            for x in to_remove:
                perturbed.discard(x)
            
            # Repair: add elements to cover uncovered triples
            uncovered = [True] * T
            for x in perturbed:
                for idx in elem_to_triples[x]:
                    uncovered[idx] = False
            
            while True:
                if time.time() > deadline - 3:
                    break
                any_uncovered = False
                for i in range(T):
                    if uncovered[i]:
                        any_uncovered = True
                        break
                if not any_uncovered:
                    break
                
                candidates = set()
                for i in range(T):
                    if uncovered[i]:
                        for x in triples[i]:
                            candidates.add(x)
                
                counts = {x: 0 for x in candidates}
                for i in range(T):
                    if uncovered[i]:
                        for x in triples[i]:
                            counts[x] += 1
                
                max_count = max(counts.values())
                best_elems = [x for x, c in counts.items() if c == max_count]
                best_elem = random.choice(best_elems)
                
                perturbed.add(best_elem)
                
                for idx in elem_to_triples[best_elem]:
                    uncovered[idx] = False
            
            perturbed = local_search_remove(perturbed)
            
            if len(perturbed) < current_size:
                current_best = perturbed
                current_size = len(perturbed)
                best_R = perturbed
                best_R_size = current_size
    
    if best_R is None:
        # Fallback: use all elements (should not happen)
        best_R = set()
    
    S = list(set(range(1, N + 1)) - best_R)
    return S
