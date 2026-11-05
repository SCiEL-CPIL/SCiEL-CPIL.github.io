import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _score(s):
    """Calculate the number of distinct d > 0 such that a, a+d, a+2d are in s."""
    s_set = set(s)
    diffs = set()
    elements = sorted(s_set)
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            d = b - a
            c = b + d
            if c in s_set:
                diffs.add(d)
    return len(diffs)

def _score_with_diffs(s):
    """Return (score, set_of_diffs) for a set."""
    s_set = set(s)
    diffs = set()
    elements = sorted(s_set)
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            d = b - a
            c = b + d
            if c in s_set:
                diffs.add(d)
    return len(diffs), diffs

def _hill_climb(current_set, current_score, max_steps=5000, time_limit=None):
    """
    Perform hill climbing from current_set.
    Returns (best_set, best_score).
    """
    best_set = set(current_set)
    best_score = current_score
    steps = 0
    no_improve_steps = 0
    
    while steps < max_steps:
        if time_limit and time.time() > time_limit:
            break
        
        # Pick a random element to remove and a random element to add
        elem_out = random.choice(list(current_set))
        elem_in = random.randint(1, 512)
        
        if elem_in in current_set:
            steps += 1
            continue
            
        new_set = set(current_set)
        new_set.remove(elem_out)
        new_set.add(elem_in)
        
        new_score, _ = _score_with_diffs(new_set)
        delta = new_score - current_score
        
        if delta > 0:
            current_set = new_set
            current_score = new_score
            steps = 0
            no_improve_steps = 0
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        else:
            no_improve_steps += 1
            steps += 1
            
        if no_improve_steps > 1000:
            break
            
    return best_set, best_score

def _greedy_coverage(initial_set, time_limit=None):
    """
    Maximum Coverage greedy heuristic:
    Start from initial_set, then iteratively swap out elements that
    contribute least to new difference coverage and swap in elements
    that cover the most new differences.
    """
    current_set = set(initial_set)
    current_score, current_diffs = _score_with_diffs(current_set)
    best_set = set(current_set)
    best_score = current_score
    no_improve = 0
    max_no_improve = 500
    
    while time_limit is None or time.time() < time_limit:
        if no_improve > max_no_improve:
            break
        
        # For each element in current_set, compute how many unique diffs it "owns"
        # (diffs that would be lost if we removed it)
        elem_contrib = {}
        for elem in current_set:
            temp_set = set(current_set)
            temp_set.remove(elem)
            _, temp_diffs = _score_with_diffs(temp_set)
            lost = current_diffs - temp_diffs
            elem_contrib[elem] = len(lost)
        
        # Find element with minimum contribution
        min_elem = min(elem_contrib, key=lambda e: elem_contrib[e])
        
        # For each candidate not in current_set, compute how many new diffs it adds
        # when replacing min_elem
        best_new_elem = None
        best_new_score = -1
        best_new_diffs = None
        
        # Sample candidates to keep this tractable
        candidates = [x for x in range(1, 513) if x not in current_set]
        random.shuffle(candidates)
        candidates = candidates[:200]  # Limit to 200 candidates for speed
        
        for cand in candidates:
            new_set = set(current_set)
            new_set.remove(min_elem)
            new_set.add(cand)
            new_score, new_diffs = _score_with_diffs(new_set)
            if new_score > best_new_score:
                best_new_score = new_score
                best_new_elem = cand
                best_new_diffs = new_diffs
        
        if best_new_elem is None:
            break
            
        if best_new_score > current_score:
            current_set.remove(min_elem)
            current_set.add(best_new_elem)
            current_score = best_new_score
            current_diffs = best_new_diffs
            no_improve = 0
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        else:
            no_improve += 1
            
            # Try a random swap as perturbation
            if random.random() < 0.3:
                e_out = random.choice(list(current_set))
                e_in = random.randint(1, 512)
                if e_in not in current_set:
                    current_set.remove(e_out)
                    current_set.add(e_in)
                    current_score, current_diffs = _score_with_diffs(current_set)
                    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_set = None
    best_score = 0
    
    # Phase 1: Diverse Initializations with Multiple Strategies
    restarts = 0
    max_restarts = 80
    
    while time.time() < deadline - 50.0 and restarts < max_restarts:
        restarts += 1
        
        # Generate initial set using diverse strategies
        r = random.random()
        if r < 0.2:
            # Random
            current_set = set(random.sample(range(1, 513), 32))
        elif r < 0.4:
            # Interval + Perturbation
            base = random.randint(1, 480)
            current_set = set(base + i for i in range(32))
            for _ in range(5):
                e_out = random.choice(list(current_set))
                e_in = random.randint(1, 512)
                if e_in not in current_set:
                    current_set.remove(e_out)
                    current_set.add(e_in)
        elif r < 0.6:
            # Two clusters
            c1 = random.randint(1, 200)
            c2 = random.randint(250, 450)
            current_set = set(c1 + i for i in range(16)) | set(c2 + i for i in range(16))
        elif r < 0.8:
            # Three clusters
            c1 = random.randint(1, 150)
            c2 = random.randint(200, 300)
            c3 = random.randint(350, 450)
            current_set = set(c1 + i for i in range(11)) | set(c2 + i for i in range(11)) | set(c3 + i for i in range(10))
        else:
            # Geometric-ish spread
            current_set = set()
            base = random.randint(1, 100)
            ratio = random.uniform(1.5, 2.5)
            val = base
            while len(current_set) < 32 and val <= 512:
                current_set.add(int(val))
                val *= ratio
            # Fill remaining randomly
            while len(current_set) < 32:
                current_set.add(random.randint(1, 512))
            current_set = set(list(current_set)[:32])
        
        if len(current_set) != 32:
            continue
            
        current_score, _ = _score_with_diffs(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
            
        # Hill Climb first
        hc_set, hc_score = _hill_climb(current_set, current_score, max_steps=1500, time_limit=deadline - 50.0)
        
        if hc_score > best_score:
            best_score = hc_score
            best_set = set(hc_set)
            
        # Then apply greedy coverage on the result
        gc_set, gc_score = _greedy_coverage(hc_set, time_limit=deadline - 50.0)
        
        if gc_score > best_score:
            best_score = gc_score
            best_set = set(gc_set)
            
    # Phase 2: Simulated Annealing on the best found so far
    if best_set is not None:
        current_set = set(best_set)
        current_score, _ = _score_with_diffs(current_set)
        
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        stuck_count = 0
        max_stuck = 5000
        
        while time.time() < deadline - 1.0:
            elem_out = random.choice(list(current_set))
            elem_in = random.randint(1, 512)
            
            if elem_in in current_set:
                continue
                
            new_set = set(current_set)
            new_set.remove(elem_out)
            new_set.add(elem_in)
            
            new_score, _ = _score_with_diffs(new_set)
            delta = new_score - current_score
            
            if delta > 0:
                current_set = new_set
                current_score = new_score
                stuck_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
            else:
                if random.random() < np.exp(delta / T):
                    current_set = new_set
                    current_score = new_score
                else:
                    stuck_count += 1
            
            T *= cooling
            if T < min_T:
                T = min_T
            
            if stuck_count > max_stuck:
                # Restart SA with a kick
                for _ in range(10):
                    e_out = random.choice(list(current_set))
                    e_in = random.randint(1, 512)
                    if e_in not in current_set:
                        current_set = set(current_set)
                        current_set.remove(e_out)
                        current_set.add(e_in)
                current_score, _ = _score_with_diffs(current_set)
                stuck_count = 0
                T = 20.0

    # Phase 3: Final polish with greedy coverage on the best result
    if best_set is not None and time.time() < deadline - 0.5:
        final_set, final_score = _greedy_coverage(best_set, time_limit=deadline - 0.5)
        if final_score > best_score:
            best_score = final_score
            best_set = set(final_set)

    return list(best_set) if best_set else list(range(1, 33))

# EVOLVE-BLOCK-END
