# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _is_bad_triple(a, b, c):
    return _lcm(a, b) == _lcm(a, c) == _lcm(b, c)

# Precomputation
BAD_TRIPLES = []
CONFLICTS_WITH = {i: [] for i in range(1, 129)}

def _precompute():
    for a in range(1, 129):
        for b in range(a + 1, 129):
            for c in range(b + 1, 129):
                if _is_bad_triple(a, b, c):
                    BAD_TRIPLES.append((a, b, c))
                    CONFLICTS_WITH[a].append((b, c))
                    CONFLICTS_WITH[b].append((a, c))
                    CONFLICTS_WITH[c].append((a, b))

_precompute()

def _count_bad_triples_in_set(S):
    """Count how many bad triples are fully contained in S."""
    if not S:
        return 0
    S_set = set(S)
    count = 0
    for t in BAD_TRIPLES:
        if t[0] in S_set and t[1] in S_set and t[2] in S_set:
            count += 1
    return count

def _greedy_construct(ordering):
    """Greedy construction following given ordering."""
    selected = []
    selected_set = set()
    for x in ordering:
        can_add = True
        for i, a in enumerate(selected):
            if not can_add:
                break
            for j in range(i + 1, len(selected)):
                b = selected[j]
                if _is_bad_triple(x, a, b):
                    can_add = False
                    break
        if can_add:
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_search_add(S):
    """Try to add elements not in S to S."""
    S = list(S)
    S_set = set(S)
    n = len(S)
    changed = True
    while changed and n < 128:
        changed = False
        for x in range(1, 129):
            if x in S_set:
                continue
            can_add = True
            for i, a in enumerate(S):
                if not can_add:
                    break
                for j in range(i + 1, len(S)):
                    b = S[j]
                    if _is_bad_triple(x, a, b):
                        can_add = False
                        break
            if can_add:
                S.append(x)
                S_set.add(x)
                n += 1
                changed = True
                break
    return S

def _sa_find_valid(start_S, time_limit):
    """
    Simulated Annealing to find a large valid set.
    """
    S = list(start_S)
    S_set = set(S)
    current_cost = _count_bad_triples_in_set(S)
    
    best_valid_S = list(S) if current_cost == 0 else []
    best_valid_size = len(best_valid_S)
    
    T = 1.0
    cooling_rate = 0.9995
    end_time = time.time() + time_limit
    
    while time.time() < end_time:
        if T < 0.005:
            break
            
        move_type = random.choice(['add', 'remove', 'swap'])
        new_S = list(S)
        new_S_set = set(S)
        
        if move_type == 'add':
            candidates = [x for x in range(1, 129) if x not in new_S_set]
            if not candidates:
                T *= cooling_rate
                continue
            x = random.choice(candidates)
            new_S.append(x)
            new_S_set.add(x)
            # Incremental cost update
            # Cost increases by number of bad triples {x, u, v} with u, v in S
            delta = 0
            for u, v in CONFLICTS_WITH[x]:
                if u in S_set and v in S_set:
                    delta += 1
            new_cost = current_cost + delta
            
        elif move_type == 'remove':
            if not new_S:
                T *= cooling_rate
                continue
            x = random.choice(new_S)
            new_S.remove(x)
            new_S_set.discard(x)
            # Cost decreases by number of bad triples {x, u, v} with u, v in S\{x}
            delta = 0
            for u, v in CONFLICTS_WITH[x]:
                if u in S_set and v in S_set:
                    delta += 1
            new_cost = current_cost - delta
            
        else: # swap
            if len(new_S) < 1:
                T *= cooling_rate
                continue
            x_out = random.choice(new_S)
            candidates_in = [x for x in range(1, 129) if x not in new_S_set]
            if not candidates_in:
                T *= cooling_rate
                continue
            x_in = random.choice(candidates_in)
            new_S.remove(x_out)
            new_S_set.discard(x_out)
            new_S.append(x_in)
            new_S_set.add(x_in)
            
            # Cost change:
            # Remove triples with x_out: {x_out, u, v} with u,v in S\{x_out}
            delta_out = 0
            for u, v in CONFLICTS_WITH[x_out]:
                if u in S_set and v in S_set:
                    delta_out += 1
            
            # Add triples with x_in: {x_in, u, v} with u,v in S\{x_out}
            delta_in = 0
            for u, v in CONFLICTS_WITH[x_in]:
                if u in S_set and v in S_set and u != x_out and v != x_out:
                    # Note: u,v are in S_set (which includes x_out).
                    # We need u,v in S \ {x_out}.
                    # Since u,v are distinct from x_in, and x_in is not in S, 
                    # u,v are in S.
                    # If u == x_out, it's not in S \ {x_out}.
                    # So we must check u != x_out and v != x_out.
                    delta_in += 1
            
            new_cost = current_cost - delta_out + delta_in

        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            S = new_S
            S_set = new_S_set
            current_cost = new_cost
            
            if current_cost == 0:
                expanded = _local_search_add(S)
                if len(expanded) > best_valid_size:
                    best_valid_size = len(expanded)
                    best_valid_S = expanded
                    S = expanded
                    S_set = set(S)
                    current_cost = 0
        
        T *= cooling_rate

    return best_valid_S

def construct():
    deadline = time.time() + 85.0
    
    best_result = _greedy_construct(list(range(1, 129)))
    best_result = _local_search_add(best_result)
    best_size = len(best_result)
    
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 5.0:
            break
        perm = list(range(1, 129))
        random.shuffle(perm)
        S = _greedy_construct(perm)
        S = _local_search_add(S)
        if len(S) > best_size:
            best_size = len(S)
            best_result = S

    remaining_time = deadline - time.time() - 2.0
    if remaining_time > 1.0:
        num_sa_runs = 5
        time_per_sa = remaining_time / num_sa_runs
        
        for i in range(num_sa_runs):
            if time.time() > deadline - 2.0:
                break
            
            if i == 0:
                start_S = list(best_result)
            else:
                start_S = random.sample(range(1, 129), min(best_size, 128))
            
            S_valid = _sa_find_valid(start_S, time_per_sa)
            
            if S_valid and len(S_valid) > best_size:
                best_size = len(S_valid)
                best_result = S_valid

    if not best_result:
        best_result = [1]
        
    return best_result
# EVOLVE-BLOCK-END
