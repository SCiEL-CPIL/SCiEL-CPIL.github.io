import time
import random
from collections import defaultdict

def _triples(N=180):
    """Precompute all forbidden triples (a,b,c) with a<b<c and 1/a=1/b+1/c."""
    triples = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            s = a + b
            p = a * b
            if p % s == 0:
                c = p // s
                if c > b and c <= N:
                    triples.append((a, b, c))
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    N = 180
    triples = _triples(N)
    n_triples = len(triples)
    
    # Build element-to-triples mapping for fast validity checks
    elem_triples = defaultdict(list)
    for i, (a, b, c) in enumerate(triples):
        elem_triples[a].append((i, b, c))
        elem_triples[b].append((i, a, c))
        elem_triples[c].append((i, a, b))
    
    # Precompute for each element, the pairs (other1, other2) it forms triples with
    elem_pairs = defaultdict(list)
    for (a, b, c) in triples:
        elem_pairs[a].append((b, c))
        elem_pairs[b].append((a, c))
        elem_pairs[c].append((a, b))
    
    # Precompute conflict counts per element (total number of triples each element is in)
    elem_conflict_count = [0] * (N+1)
    for (a, b, c) in triples:
        elem_conflict_count[a] += 1
        elem_conflict_count[b] += 1
        elem_conflict_count[c] += 1
    
    def count_violations(state):
        """Count number of violated triples in a binary state array (1-indexed)."""
        cnt = 0
        for (a, b, c) in triples:
            if state[a] and state[b] and state[c]:
                cnt += 1
        return cnt
    
    def can_add(elem, state):
        """Check if adding elem to current state creates any violation."""
        for (o1, o2) in elem_pairs[elem]:
            if state[o1] and state[o2]:
                return False
        return True
    
    # --- Phase 1: Greedy construction ---
    # Sort elements by conflict count (fewest conflicts first)
    elements_sorted = sorted(range(1, N+1), key=lambda x: (elem_conflict_count[x], -x))
    
    state = [False] * (N+1)
    for elem in elements_sorted:
        if can_add(elem, state):
            state[elem] = True
    
    best_state = state[:]
    best_size = sum(state)
    
    # --- Phase 2: Try to add back excluded elements with random restarts ---
    random.seed(12345)
    
    for trial in range(30):
        if time.time() > deadline - 15:
            break
        
        # Greedy with random tie-breaking
        shuffled_elements = elements_sorted[:]
        # Add some randomness: occasionally swap nearby elements
        for _ in range(20):
            i = random.randint(0, N-2)
            shuffled_elements[i], shuffled_elements[i+1] = shuffled_elements[i+1], shuffled_elements[i]
        
        trial_state = [False] * (N+1)
        for elem in shuffled_elements:
            if can_add(elem, trial_state):
                trial_state[elem] = True
        
        trial_size = sum(trial_state)
        if trial_size > best_size:
            best_size = trial_size
            best_state = trial_state[:]
    
    # --- Phase 3: Local search - single flips and swaps ---
    # Try adding excluded elements
    for trial in range(100):
        if time.time() > deadline - 10:
            break
        
        improved = False
        excluded = [i for i in range(1, N+1) if not best_state[i]]
        random.shuffle(excluded)
        
        for elem in excluded:
            if can_add(elem, best_state):
                best_state[elem] = True
                best_size += 1
                improved = True
        
        if not improved:
            break
    
    # --- Phase 4: Simulated Annealing ---
    # Objective: maximize size - penalty * violations
    # We'll use a smooth-ish approach: track violations and size separately
    
    if time.time() < deadline - 5:
        current_state = best_state[:]
        current_size = best_size
        current_violations = count_violations(current_state)
        
        # Temperature schedule
        T_start = 5.0
        T_end = 0.01
        n_steps = 5000
        
        step = 0
        for step in range(n_steps):
            if time.time() > deadline - 5:
                break
            
            # Cooling
            T = T_start * (T_end / T_start) ** (step / max(n_steps - 1, 1))
            
            # Choose a random move: flip one element or swap two elements
            move_type = random.random()
            
            if move_type < 0.4:
                # Single flip
                elem = random.randint(1, N)
                old_val = current_state[elem]
                current_state[elem] = not old_val
                
                # Compute delta in violations
                # Only need to check triples containing elem
                delta_viol = 0
                for (ti, o1, o2) in elem_triples[elem]:
                    if old_val:
                        # elem was in, now out: violations decrease if o1,o2 both in
                        if current_state[o1] and current_state[o2]:
                            delta_viol -= 1
                    else:
                        # elem was out, now in: violations increase if o1,o2 both in
                        if current_state[o1] and current_state[o2]:
                            delta_viol += 1
                
                delta_size = 1 if not old_val else -1
                current_size += delta_size
                current_violations += delta_viol
                
                # Acceptance: we want to maximize size, minimize violations
                # Score = size - lambda * violations
                # Delta score = delta_size - lambda * delta_viol
                lam = 2.0
                delta_score = delta_size - lam * delta_viol
                
                if delta_score >= 0 or random.random() < _exp_neg(delta_score / T):
                    # Accept
                    pass
                else:
                    # Reject
                    current_state[elem] = old_val
                    current_size -= delta_size
                    current_violations -= delta_viol
            
            elif move_type < 0.8:
                # Swap: remove one included, add one excluded
                included = [i for i in range(1, N+1) if current_state[i]]
                excluded = [i for i in range(1, N+1) if not current_state[i]]
                
                if not included or not excluded:
                    continue
                
                rem = random.choice(included)
                add = random.choice(excluded)
                
                # Remove rem, add add
                current_state[rem] = False
                current_state[add] = True
                
                # Compute delta violations
                delta_viol = 0
                for (ti, o1, o2) in elem_triples[rem]:
                    if current_state[o1] and current_state[o2]:
                        # This triple was violated (rem was in, now out) -> violation removed
                        delta_viol -= 1
                for (ti, o1, o2) in elem_triples[add]:
                    if current_state[o1] and current_state[o2]:
                        # This triple now violated (add was out, now in)
                        delta_viol += 1
                
                delta_size = 0  # swap keeps size same
                current_violations += delta_viol
                
                lam = 2.0
                delta_score = delta_size - lam * delta_viol
                
                if delta_score >= 0 or random.random() < _exp_neg(delta_score / T):
                    pass
                else:
                    current_state[rem] = True
                    current_state[add] = False
                    current_violations -= delta_viol
            
            else:
                # 2-opt: remove two, add two (keep size)
                included = [i for i in range(1, N+1) if current_state[i]]
                excluded = [i for i in range(1, N+1) if not current_state[i]]
                
                if len(included) < 2 or len(excluded) < 2:
                    continue
                
                rem1, rem2 = random.sample(included, 2)
                add1, add2 = random.sample(excluded, 2)
                
                current_state[rem1] = False
                current_state[rem2] = False
                current_state[add1] = True
                current_state[add2] = True
                
                delta_viol = 0
                for (ti, o1, o2) in elem_triples[rem1]:
                    if current_state[o1] and current_state[o2]:
                        delta_viol -= 1
                for (ti, o1, o2) in elem_triples[rem2]:
                    if current_state[o1] and current_state[o2]:
                        delta_viol -= 1
                for (ti, o1, o2) in elem_triples[add1]:
                    if current_state[o1] and current_state[o2]:
                        delta_viol += 1
                for (ti, o1, o2) in elem_triples[add2]:
                    if current_state[o1] and current_state[o2]:
                        delta_viol += 1
                
                delta_size = 0
                current_violations += delta_viol
                
                lam = 2.0
                delta_score = delta_size - lam * delta_viol
                
                if delta_score >= 0 or random.random() < _exp_neg(delta_score / T):
                    pass
                else:
                    current_state[rem1] = True
                    current_state[rem2] = True
                    current_state[add1] = False
                    current_state[add2] = False
                    current_violations -= delta_viol
            
            # Track best
            if current_violations == 0 and current_size > best_size:
                best_size = current_size
                best_state = current_state[:]
    
    # --- Phase 5: Final polish - try to add any remaining excluded elements ---
    excluded = [i for i in range(1, N+1) if not best_state[i]]
    random.shuffle(excluded)
    for elem in excluded:
        if can_add(elem, best_state):
            best_state[elem] = True
            best_size += 1
    
    # Verify no violations
    final_violations = count_violations(best_state)
    if final_violations > 0:
        # Emergency: remove elements from violating triples
        for _ in range(100):
            if count_violations(best_state) == 0:
                break
            # Find a violating triple
            for (a, b, c) in triples:
                if best_state[a] and best_state[b] and best_state[c]:
                    # Remove the one with most conflicts
                    options = [(elem_conflict_count[a], a), (elem_conflict_count[b], b), (elem_conflict_count[c], c)]
                    options.sort(reverse=True)
                    best_state[options[0][1]] = False
                    best_size -= 1
                    break
    
    result = [i for i in range(1, N+1) if best_state[i]]
    return result

def _exp_neg(x):
    """Safe exp(-x) for SA acceptance."""
    if x > 500:
        return 0.0
    import math
    return math.exp(-x)
