"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s kill)
    
    # Precompute all reciprocal triples (a, b, c) with a < b < c
    # where 1/a = 1/b + 1/c, i.e., bc = a(b+c)
    # a = bc/(b+c) must be integer in [1,180], distinct from b,c
    
    # For each pair (b,c) with b<c, compute a = bc/(b+c) if integer
    triple_map = {}  # (b, c) -> a
    all_triples = []  # list of (a, b, c)
    for b in range(1, 181):
        for c in range(b + 1, 181):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= 180 and a != b and a != c:
                    triple_map[(b, c)] = a
                    all_triples.append((a, b, c))
    
    # Precompute for each element x, the list of triples it participates in
    # This allows faster violation checking
    triples_by_element = {i: [] for i in range(1, 181)}
    for (a, b, c) in all_triples:
        triples_by_element[a].append((a, b, c))
        triples_by_element[b].append((a, b, c))
        triples_by_element[c].append((a, b, c))
    
    # Conflict count for each element
    conflict_count = [0] * 181
    for (a, b, c) in all_triples:
        conflict_count[a] += 1
        conflict_count[b] += 1
        conflict_count[c] += 1
    
    def is_valid(subset_set):
        """Check if subset has no reciprocal triple."""
        for (a, b, c) in all_triples:
            if a in subset_set and b in subset_set and c in subset_set:
                return False
        return True
    
    def count_violations(subset_set):
        count = 0
        for (a, b, c) in all_triples:
            if a in subset_set and b in subset_set and c in subset_set:
                count += 1
        return count
    
    # Find a violation: return (a,b,c) or None
    def find_violation(subset_set):
        for (a, b, c) in all_triples:
            if a in subset_set and b in subset_set and c in subset_set:
                return (a, b, c)
        return None
    
    # Greedy construction: add elements in order of fewest conflicts
    order = sorted(range(1, 181), key=lambda x: conflict_count[x])
    
    greedy_set = set()
    for x in order:
        trial = greedy_set | {x}
        if is_valid(trial):
            greedy_set = trial
    
    best_set = set(greedy_set)
    best_size = len(greedy_set)
    
    # Local search with add and swap moves
    def local_search(current_set, max_iter=20000):
        current = set(current_set)
        current_size = len(current)
        excluded = set(range(1, 181)) - current
        
        # Track violations for faster checking
        # Precompute: for each element, which other elements are "dangerous" with it
        # i.e., for each pair (x,y), check if there exists z such that (x,y,z) is a triple
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            made_progress = False
            
            # Try adding a random excluded element
            if excluded:
                candidates = list(excluded)
                random.shuffle(candidates)
                for x in candidates[:20]:
                    trial = current | {x}
                    if is_valid(trial):
                        current = trial
                        current_size += 1
                        excluded.remove(x)
                        made_progress = True
                        break
            
            if not excluded:
                break
            
            # Try swap: remove y from current, add x from excluded
            if excluded and current:
                remove_candidates = list(current)
                random.shuffle(remove_candidates)
                add_candidates = list(excluded)
                random.shuffle(add_candidates)
                for x in add_candidates[:10]:
                    found_swap = False
                    for y in remove_candidates[:10]:
                        if x == y:
                            continue
                        trial = (current - {y}) | {x}
                        if is_valid(trial):
                            current = trial
                            excluded.remove(x)
                            excluded.add(y)
                            found_swap = True
                            made_progress = True
                            break
                    if found_swap:
                        break
            
            if not made_progress:
                # Stuck, try a more aggressive approach
                # Try removing up to 2 elements to free up space
                if excluded and len(current) > 2:
                    remove_candidates = list(current)
                    random.shuffle(remove_candidates)
                    add_candidates = list(excluded)
                    random.shuffle(add_candidates)
                    for x in add_candidates[:5]:
                        found_swap2 = False
                        for i in range(min(5, len(remove_candidates))):
                            for j in range(i+1, min(10, len(remove_candidates))):
                                y1, y2 = remove_candidates[i], remove_candidates[j]
                                if x == y1 or x == y2:
                                    continue
                                trial = (current - {y1, y2}) | {x}
                                if is_valid(trial):
                                    current = trial
                                    excluded.remove(x)
                                    excluded.add(y1)
                                    excluded.add(y2)
                                    found_swap2 = True
                                    made_progress = True
                                    break
                            if found_swap2:
                                break
                        if found_swap2:
                            break
    
        return current, len(current)
    
    # Run local search from greedy
    improved_set, improved_size = local_search(greedy_set)
    if improved_size > best_size:
        best_set = improved_set
        best_size = improved_size
    
    # Final polish: systematically try adding each excluded element
    def final_polish(current_set):
        current = set(current_set)
        excluded = set(range(1, 181)) - current
        
        # Repeatedly try to add each excluded element until no more can be added
        progress = True
        while progress and time.time() < deadline:
            progress = False
            for x in sorted(excluded):
                if time.time() > deadline:
                    break
                trial = current | {x}
                if is_valid(trial):
                    current = trial
                    excluded.remove(x)
                    progress = True
        
        return current, len(current)
    
    polished_set, polished_size = final_polish(best_set)
    if polished_size > best_size:
        best_set = polished_set
        best_size = polished_size
    
    # Multi-restart with different initializations
    restarts = [
        set(range(91, 181)),           # upper half
        set(range(2, 181, 2)),         # even numbers
        set(range(61, 181)),           # numbers > 60
        set(range(121, 181)),          # top 60
        set(range(1, 181, 3)),         # multiples of 3
        set(range(4, 181, 4)),         # multiples of 4
    ]
    
    for idx, init_set in enumerate(restarts):
        if time.time() > deadline:
            break
        
        # Clean up violations
        cleaned = set(init_set)
        while True:
            if time.time() > deadline:
                break
            v = find_violation(cleaned)
            if v is None:
                break
            a, b, c = v
            # Remove the element with highest conflict count
            max_conf = max([a, b, c], key=lambda x: conflict_count[x])
            cleaned.remove(max_conf)
        
        # Local search
        ls_set, ls_size = local_search(cleaned, max_iter=10000)
        if ls_size > best_size:
            best_set = ls_set
            best_size = ls_size
        
        # Final polish
        polished_set, polished_size = final_polish(best_set)
        if polished_size > best_size:
            best_set = polished_set
            best_size = polished_size
    
    # Random restarts
    for _ in range(5):
        if time.time() > deadline:
            break
        size = random.randint(80, 120)
        init_set = set(random.sample(range(1, 181), size))
        
        # Clean up violations
        cleaned = set(init_set)
        while True:
            if time.time() > deadline:
                break
            v = find_violation(cleaned)
            if v is None:
                break
            a, b, c = v
            max_conf = max([a, b, c], key=lambda x: conflict_count[x])
            cleaned.remove(max_conf)
        
        # Local search
        ls_set, ls_size = local_search(cleaned, max_iter=8000)
        if ls_size > best_size:
            best_set = ls_set
            best_size = ls_size
        
        # Final polish
        polished_set, polished_size = final_polish(best_set)
        if polished_size > best_size:
            best_set = polished_set
            best_size = polished_size
    
    return list(best_set)

# EVOLVE-BLOCK-END
