import math
import time
import random

def _score(u):
    """Compute the score for set u."""
    if not u:
        return 0.0
    max_u = max(u)
    if max_u == 0:
        return 1.0
    diffs = set()
    sums = set()
    for a in u:
        for b in u:
            diffs.add(a - b)
            sums.add(a + b)
    num_diffs = len(diffs)
    num_sums = len(sums)
    if num_sums == 0:
        return 0.0
    ratio = num_diffs / num_sums
    denom = 2 * max_u + 1
    if denom <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(denom)

def _score_fast(u):
    """Score without sorting (assumes 0 in u, u is a set or list)."""
    if not u:
        return 0.0
    u_set = u if isinstance(u, set) else set(u)
    max_u = max(u_set)
    if max_u == 0:
        return 1.0
    diffs = set()
    sums = set()
    for a in u_set:
        for b in u_set:
            diffs.add(a - b)
            sums.add(a + b)
    num_diffs = len(diffs)
    num_sums = len(sums)
    if num_sums == 0:
        return 0.0
    ratio = num_diffs / num_sums
    denom = 2 * max_u + 1
    if denom <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(denom)

def _count_new_pairs(u_set, x):
    """Count new sums and diffs when adding x to u_set."""
    new_sums = 0
    new_diffs = 0
    existing_sums = set(a + b for a in u_set for b in u_set)
    existing_diffs = set(a - b for a in u_set for b in u_set)
    
    for a in u_set:
        s = x + a
        if s not in existing_sums:
            new_sums += 1
        d1 = x - a
        if d1 not in existing_diffs:
            new_diffs += 1
        d2 = a - x
        if d2 not in existing_diffs:
            new_diffs += 1
    # x+x and x-x=0
    if (x + x) not in existing_sums:
        new_sums += 1
    # 0 is always in existing_diffs if 0 in u_set
    
    return new_sums, new_diffs

def _greedy_build(max_val, target_size, time_limit):
    """Greedy construction: add elements that maximize score improvement."""
    start_time = time.time()
    u = set([0])
    current_score = 1.0
    
    while time.time() - start_time < time_limit and len(u) < target_size:
        best_x = None
        best_score = current_score
        
        # Sample candidates
        candidates = random.sample(range(1, max_val + 1), min(50, max_val))
        
        for x in candidates:
            if x in u:
                continue
            
            new_u = u | {x}
            s = _score_fast(new_u)
            if s > best_score:
                best_score = s
                best_x = x
        
        if best_x is not None:
            u.add(best_x)
            current_score = best_score
        else:
            break
    
    return list(u), current_score

def _local_search(u, time_limit):
    """Local search: try single add/remove/swap moves."""
    start_time = time.time()
    current = set(u)
    current_score = _score_fast(current)
    max_val = max(current) if current else 100
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try removing an element
        for x in list(current):
            if x == 0:
                continue
            if time.time() - start_time > time_limit:
                break
            new_set = current - {x}
            s = _score_fast(new_set)
            if s > current_score:
                current = new_set
                current_score = s
                improved = True
                break
        
        if improved:
            continue
        
        # Try adding an element
        candidates = random.sample(range(1, max_val + 50), min(30, max_val + 49))
        for x in candidates:
            if x in current:
                continue
            if time.time() - start_time > time_limit:
                break
            new_set = current | {x}
            s = _score_fast(new_set)
            if s > current_score:
                current = new_set
                current_score = s
                max_val = max(current)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove one, add another
        if len(current) > 2:
            for x in list(current):
                if x == 0:
                    continue
                if time.time() - start_time > time_limit:
                    break
                base = current - {x}
                for y in random.sample(range(1, max_val + 50), min(20, max_val + 49)):
                    if y in base or y == x:
                        continue
                    new_set = base | {y}
                    s = _score_fast(new_set)
                    if s > current_score:
                        current = new_set
                        current_score = s
                        max_val = max(current)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Perturbation: random restart with small perturbation
            if len(current) > 3:
                # Remove 2 random elements, add 2 random new ones
                to_remove = random.sample([x for x in current if x != 0], 2)
                current -= set(to_remove)
                for _ in range(2):
                    y = random.randint(1, max_val + 100)
                    if y not in current:
                        current.add(y)
                current_score = _score_fast(current)
                max_val = max(current)
            else:
                break
    
    return list(current), current_score

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the sumset/difference-set exponent lower bound.
    
    Strategy: 
    1. Use greedy construction to build initial sets of various sizes.
    2. Refine with local search including add/remove/swap moves.
    3. Try multiple random restarts with different max_val.
    """
    start_time = time.time()
    total_time = 110  # Safety margin
    
    best_set = [0]
    best_score = 1.0
    
    # Try different configurations
    configs = [
        (200, 30),   # max_val, target_size
        (500, 40),
        (1000, 50),
        (2000, 60),
        (5000, 70),
    ]
    
    for max_val, target_size in configs:
        if time.time() - start_time > total_time - 15:
            break
        
        time_left = total_time - (time.time() - start_time)
        greedy_time = time_left * 0.3
        search_time = time_left * 0.6
        
        # Greedy construction
        u, score = _greedy_build(max_val, target_size, greedy_time)
        
        # Local search refinement
        u, score = _local_search(u, search_time)
        
        if score > best_score:
            best_score = score
            best_set = u
    
    # Final refinement with remaining time
    if time.time() - start_time < total_time - 5:
        time_left = total_time - (time.time() - start_time)
        best_set, best_score = _local_search(best_set, time_left)
    
    return best_set

if __name__ == "__main__":
    u = construct()
    print(f"Best set size: {len(u)}")
    print(f"Best set max: {max(u)}")
    print(f"Score: {_score(u)}")
