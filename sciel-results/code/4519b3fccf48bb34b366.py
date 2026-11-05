import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    n_points = 16
    all_points = list(range(n_points))
    
    # Precompute all triples
    all_triples = list(itertools.combinations(all_points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute all 8-subsets
    all_8sets = list(itertools.combinations(all_points, 8))
    num_8sets = len(all_8sets)
    
    # Map triples to bitmasks for fast subset check? 
    # Or just use index mapping for hit counts.
    
    # For each triple, which 8-sets does it hit?
    # T hits S if T is subset of S.
    triple_hits_8sets = [[] for _ in range(num_triples)]
    # For each 8-set, which triples hit it?
    # 8set_hits_triples = [[] for _ in range(num_8sets)]
    
    # Efficiently populate triple_hits_8sets
    # For each 8-set S, iterate all subsets T of S of size 3.
    # There are C(8,3) = 56 such subsets.
    # Total iterations: 12870 * 56 = ~720k. Fast.
    for s_idx, s in enumerate(all_8sets):
        s_set = set(s)
        for t in itertools.combinations(s, 3):
            t_idx = triple_to_idx[t]
            triple_hits_8sets[t_idx].append(s_idx)
            
    # Precompute conflicts (sharing pairs)
    # pair_to_triples: dict from frozenset(pair) to list of triple indices
    pair_to_triples = {}
    for i, t in enumerate(all_triples):
        for p in itertools.combinations(t, 2):
            fp = frozenset(p)
            if fp not in pair_to_triples:
                pair_to_triples[fp] = []
            pair_to_triples[fp].append(i)
            
    # blocked_count[i] = number of triples in current system that conflict with i
    blocked_count = [0] * num_triples
    in_system = [False] * num_triples
    current_system = []
    
    # hit_count[j] = number of triples in current system that hit 8-set j
    hit_count = [0] * num_8sets
    # min_hit = min(hit_count)
    # We want min_hit > 0
    
    def add_triple(t_idx):
        nonlocal current_system, blocked_count, in_system, hit_count
        if in_system[t_idx]:
            return
        in_system[t_idx] = True
        current_system.append(t_idx)
        
        # Update hit_count
        for s_idx in triple_hits_8sets[t_idx]:
            hit_count[s_idx] += 1
            
        # Update blocked_count
        t = all_triples[t_idx]
        for p in itertools.combinations(t, 2):
            fp = frozenset(p)
            for other_idx in pair_to_triples[fp]:
                if other_idx != t_idx:
                    blocked_count[other_idx] += 1

    def remove_triple(t_idx):
        nonlocal current_system, blocked_count, in_system, hit_count
        if not in_system[t_idx]:
            return
        in_system[t_idx] = False
        current_system.remove(t_idx)
        
        # Update hit_count
        for s_idx in triple_hits_8sets[t_idx]:
            hit_count[s_idx] -= 1
            
        # Update blocked_count
        t = all_triples[t_idx]
        for p in itertools.combinations(t, 2):
            fp = frozenset(p)
            for other_idx in pair_to_triples[fp]:
                if other_idx != t_idx:
                    blocked_count[other_idx] -= 1

    def get_unhitted_8sets():
        return [j for j in range(num_8sets) if hit_count[j] == 0]

    def is_solved():
        return all(h > 0 for h in hit_count)

    # Greedy initialization
    # Add random triples that are not blocked
    available = [i for i in range(num_triples)]
    random.shuffle(available)
    for i in available:
        if blocked_count[i] == 0:
            add_triple(i)
        if time.time() > deadline - 20:
            break
            
    # If not solved, try local search / repair
    # We will try to fix unhitted 8-sets by swapping triples
    
    for _ in range(1000):
        if time.time() > deadline - 10:
            break
            
        if is_solved():
            break
            
        unhitted = get_unhitted_8sets()
        if not unhitted:
            break
            
        # Pick a random unhitted 8-set
        target_s_idx = random.choice(unhitted)
        target_s = all_8sets[target_s_idx]
        
        # Find candidates T subset of target_s
        candidates = []
        for t in itertools.combinations(target_s, 3):
            t_idx = triple_to_idx[t]
            if not in_system[t_idx] and blocked_count[t_idx] == 0:
                candidates.append(t_idx)
                
        if candidates:
            # Add one randomly
            new_t = random.choice(candidates)
            add_triple(new_t)
            continue
            
        # No unblocked candidate. Need to swap.
        # Find T subset of target_s that has minimal blocking
        # If blocked_count[T] == 1, we can remove the blocker and add T.
        swap_candidates = []
        for t in itertools.combinations(target_s, 3):
            t_idx = triple_to_idx[t]
            if in_system[t_idx]:
                continue
            if blocked_count[t_idx] == 1:
                # Find the blocker
                t = all_triples[t_idx]
                blockers = []
                for p in itertools.combinations(t, 2):
                    fp = frozenset(p)
                    for other_idx in pair_to_triples[fp]:
                        if other_idx != t_idx and in_system[other_idx]:
                            blockers.append(other_idx)
                if len(blockers) == 1:
                    swap_candidates.append((t_idx, blockers[0]))
                    
        if swap_candidates:
            new_t, old_t = random.choice(swap_candidates)
            # Check if swap is beneficial? 
            # Removing old_t might unhitt some 8-sets. Adding new_t hits target_s.
            # Just try it.
            remove_triple(old_t)
            add_triple(new_t)
        else:
            # Stuck. Try removing a random triple to reset some blockages?
            # Or just break and restart?
            # Let's try a random removal to explore
            if current_system:
                remove_triple(random.choice(current_system))

    # Final check
    if is_solved():
        # Return the triples as lists
        return [list(all_triples[i]) for i in current_system]
        
    # If not solved, return the best we have (which might have alpha > 7)
    # But we need to return a valid linear system.
    # The current_system is always linear by construction.
    return [list(all_triples[i]) for i in current_system]
