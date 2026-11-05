import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 180
    # 1. Generate all forbidden triples
    triples = []
    for a in range(1, N + 1):
        for d in range(1, a):
            if (a * a) % d == 0:
                b = a + d
                c = (a * b) // d
                if c <= N and c > b:
                    triples.append((a, b, c))
                    
    # If no triples, return all
    if not triples:
        return list(range(1, N + 1))
        
    best_R_size = N
    best_R = set(range(1, N + 1))
    
    # Greedy Hitting Set with random restarts
    # We want to minimize |R|
    
    def run_greedy(shuffle_seed=None):
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            
        uncovered = [set(t) for t in triples]
        R = set()
        
        # To speed up, we can count coverage
        # But N=180, T~1000, so O(N*T) is fine.
        
        while uncovered:
            if time.time() > deadline:
                return R
            
            # Find element with max coverage
            max_count = -1
            best_elem = -1
            
            # Optimize: only consider elements in uncovered triples
            candidates = set()
            for t in uncovered:
                candidates.update(t)
                
            for x in candidates:
                count = 0
                for t in uncovered:
                    if x in t:
                        count += 1
                if count > max_count:
                    max_count = count
                    best_elem = x
                    
            if best_elem == -1:
                break
                
            R.add(best_elem)
            
            # Remove covered triples
            new_uncovered = []
            for t in uncovered:
                if best_elem not in t:
                    new_uncovered.append(t)
            uncovered = new_uncovered
            
        return R

    # Run multiple restarts
    # Simple greedy is deterministic if we don't shuffle tie-breaking.
    # Let's introduce randomness in tie-breaking.
    
    def run_greedy_random():
        uncovered = [set(t) for t in triples]
        R = set()
        
        while uncovered:
            if time.time() > deadline:
                return R
            
            candidates = set()
            for t in uncovered:
                candidates.update(t)
                
            counts = {x: 0 for x in candidates}
            for t in uncovered:
                for x in t:
                    counts[x] += 1
                    
            max_count = max(counts.values())
            best_elems = [x for x, c in counts.items() if c == max_count]
            
            # Random pick among best
            best_elem = random.choice(best_elems)
            
            R.add(best_elem)
            
            new_uncovered = []
            for t in uncovered:
                if best_elem not in t:
                    new_uncovered.append(t)
            uncovered = new_uncovered
            
        return R

    # Try a few times
    for i in range(100):
        if time.time() > deadline - 5:
            break
        R = run_greedy_random()
        if len(R) < best_R_size:
            best_R_size = len(R)
            best_R = R
            # If we found a perfect hitting set (size 0? impossible unless no triples)
            # We can also try local search here
            
    # Local search on best_R
    # Try to remove elements from R
    # If removing y breaks coverage, try to add z to cover them?
    # This is complex. Let's stick to greedy.
    
    # Actually, we can try to improve by checking if we can swap
    # But greedy is usually very good for this.
    
    # One more improvement: 
    # The greedy algorithm picks the element covering the most.
    # Sometimes picking an element that covers slightly less but is "more central" might be better?
    # Hard to define.
    
    # Let's try a simple local search: 
    # For each y in R, try to remove it. 
    # If it fails, try to find a z not in R that can replace it?
    # No, we want to minimize size.
    
    # If we have time, try to add elements back to S (remove from R) if possible.
    # This is just checking if R \ {y} is a hitting set.
    
    # Let's do a pass to remove redundant elements from best_R
    # A hitting set R is minimal if no element can be removed.
    # Greedy produces a minimal hitting set? 
    # Not necessarily. It produces an irredundant one?
    # Actually, if we remove an element from a greedy hitting set, we might uncover triples.
    # So it is minimal in the sense that no single element can be removed.
    # Wait, if I pick x to cover max, and later pick y, maybe y covers triples that x also covered?
    # No, x was picked because it covered max.
    # But it's possible that a different combination of elements covers the same set with fewer elements.
    
    # Let's try a few more random restarts if time permits.
    
    S = list(set(range(1, N + 1)) - best_R)
    return S
