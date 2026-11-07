"""Combined program: Max-SAT-style clause scoring with local search refinement."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt
from itertools import combinations

def _can_add(S_list, x):
    """
    Check if x can be added to S (given as S_list).
    For every three distinct members a, b, c in S ∪ {x}, a does not divide b*c.
    """
    n = len(S_list)
    
    # 1. x divides a*b for distinct a,b in S
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            if (a * b) % x == 0:
                return False
    
    # 2. a divides x*b for distinct a,b in S
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if i == j:
                continue
            b = S_list[j]
            if (x * b) % a == 0:
                return False
    
    return True

def _compute_clause_score(S_list, all_nums, clause_map):
    """
    Compute the Max-SAT clause satisfaction score for a set.
    For each clause (¬a ∨ ¬b ∨ ¬c) where a|bc, it's satisfied if at least one of a,b,c is not in S.
    Score = number of satisfied clauses + bonus for |S|.
    """
    n = len(S_list)
    S_set = set(S_list)
    
    # Count satisfied clauses
    satisfied = 0
    for clause in clause_map:
        a, b, c = clause
        # Clause (¬a ∨ ¬b ∨ ¬c) is satisfied if at least one is not in S
        if a not in S_set or b not in S_set or c not in S_set:
            satisfied += 1
    
    # Total clauses
    total_clauses = len(clause_map)
    
    # Score: prioritize satisfying clauses, then maximize size
    # Use a weighted score: satisfied clauses * 1000 + size
    return satisfied * 1000 + n

def _build_clause_map(all_nums):
    """
    Build the clause map: for every triple (a,b,c) with a|bc, add clause (¬a ∨ ¬b ∨ ¬c).
    Return a list of tuples (a,b,c) representing clauses.
    """
    clauses = []
    n = len(all_nums)
    num_set = set(all_nums)
    
    for i in range(n):
        a = all_nums[i]
        for j in range(i + 1, n):
            b = all_nums[j]
            for k in range(j + 1, n):
                c = all_nums[k]
                # Check if a divides b*c
                if (b * c) % a == 0:
                    clauses.append((a, b, c))
    
    return clauses

def _maxsat_local_search(S_list, all_nums, clause_map, deadline, rng):
    """
    Max-SAT-style local search: try to flip variables (add/remove integers) to maximize
    the number of satisfied clauses while maintaining high cardinality.
    
    Uses a scoring function that balances clause satisfaction and set size.
    """
    current_set = set(S_list)
    current_list = list(S_list)
    
    # Precompute clause membership for fast scoring
    # For each number, track which clauses it appears in
    num_to_clauses = {x: [] for x in all_nums}
    for idx, clause in enumerate(clause_map):
        a, b, c = clause
        num_to_clauses[a].append(idx)
        num_to_clauses[b].append(idx)
        num_to_clauses[c].append(idx)
    
    # Track which clauses are currently unsatisfied
    unsatisfied_clauses = set()
    for idx, clause in enumerate(clause_map):
        a, b, c = clause
        if a in current_set and b in current_set and c in current_set:
            unsatisfied_clauses.add(idx)
    
    best_score = _compute_clause_score(current_list, all_nums, clause_map)
    best_set = current_set.copy()
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
        
        # Try random moves: add, remove, or swap
        move_type = rng.random()
        
        if move_type < 0.3 and len(current_set) < len(all_nums):
            # Try to add a random non-member
            non_members = [x for x in all_nums if x not in current_set]
            if not non_members:
                no_improve += 1
                continue
            x = rng.choice(non_members)
            
            # Check if adding x improves the score
            # New unsatisfied clauses would be those where x appears and the other two are in current_set
            new_unsat = 0
            for idx in num_to_clauses[x]:
                clause = clause_map[idx]
                a, b, c = clause
                others = [y for y in (a, b, c) if y != x]
                if all(y in current_set for y in others):
                    new_unsat += 1
            
            # Score change: -new_unsat * 1000 + 1 (size increase)
            if new_unsat == 0:
                # Adding x doesn't create new unsatisfied clauses
                current_set.add(x)
                current_list.append(x)
                # Update unsatisfied_clauses
                for idx in num_to_clauses[x]:
                    clause = clause_map[idx]
                    a, b, c = clause
                    if a in current_set and b in current_set and c in current_set:
                        unsatisfied_clauses.add(idx)
                no_improve = 0
            else:
                no_improve += 1
                
        elif move_type < 0.6 and len(current_set) > 1:
            # Try to remove a random member
            y = rng.choice(current_list)
            
            # Check if removing y reduces unsatisfied clauses
            # Clauses that become satisfied: those where y appears and all three were in current_set
            freed_clauses = 0
            for idx in num_to_clauses[y]:
                if idx in unsatisfied_clauses:
                    freed_clauses += 1
            
            # Score change: +freed_clauses * 1000 - 1 (size decrease)
            if freed_clauses >= 1:
                current_set.remove(y)
                current_list = [v for v in current_list if v != y]
                # Update unsatisfied_clauses
                for idx in num_to_clauses[y]:
                    unsatisfied_clauses.discard(idx)
                no_improve = 0
            else:
                no_improve += 1
                
        else:
            # Try a swap: remove one, add one
            if len(current_set) > 0:
                non_members = [x for x in all_nums if x not in current_set]
                if non_members:
                    y = rng.choice(current_list)
                    x = rng.choice(non_members)
                    
                    # Compute score change
                    # Clauses freed by removing y
                    freed = 0
                    for idx in num_to_clauses[y]:
                        if idx in unsatisfied_clauses:
                            freed += 1
                    
                    # New unsatisfied clauses from adding x (considering y is removed)
                    new_unsat = 0
                    temp_set = current_set - {y}
                    for idx in num_to_clauses[x]:
                        clause = clause_map[idx]
                        a, b, c = clause
                        others = [z for z in (a, b, c) if z != x]
                        if all(z in temp_set for z in others):
                            new_unsat += 1
                    
                    # Net score change: (freed - new_unsat) * 1000
                    if freed > new_unsat:
                        current_set.remove(y)
                        current_set.add(x)
                        current_list = [v for v in current_list if v != y] + [x]
                        # Update unsatisfied_clauses
                        for idx in num_to_clauses[y]:
                            unsatisfied_clauses.discard(idx)
                        for idx in num_to_clauses[x]:
                            clause = clause_map[idx]
                            a, b, c = clause
                            if a in current_set and b in current_set and c in current_set:
                                unsatisfied_clauses.add(idx)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            # Track best
            score = _compute_clause_score(current_list, all_nums, clause_map)
            if score > best_score:
                best_score = score
                best_set = current_set.copy()
                no_improve = 0
    
    return list(best_set)

def _can_add_fast(S_list, x, pair_products, pair_indices):
    """
    Fast can_add using precomputed pair products.
    pair_products[i] = S_list[i] * S_list[j] for all i<j
    pair_indices[i] = (i, j)
    """
    n = len(S_list)
    
    # 1. x divides a*b for distinct a,b in S
    for prod, (i, j) in zip(pair_products, pair_indices):
        if prod % x == 0:
            return False
    
    # 2. a divides x*b for distinct a,b in S
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if i == j:
                continue
            b = S_list[j]
            if (x * b) % a == 0:
                return False
    
    return True

def _is_valid(S_list):
    """
    Check if S is a valid set.
    For every three distinct members a, b, c in S, a does not divide b*c.
    """
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if i == j:
                continue
            b = S_list[j]
            for k in range(n):
                if i == k or j == k:
                    continue
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _greedy_construct(order, deadline):
    """
    Greedily construct a set by iterating through 'order' and adding
    if the resulting set remains valid.
    """
    S = set()
    S_list = []
    
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(S_list, x):
            S.add(x)
            S_list.append(x)
            
    return S, S_list

def _compute_pair_data(S_list):
    """Compute pair products and indices for fast can_add."""
    n = len(S_list)
    pair_products = []
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_products.append(S_list[i] * S_list[j])
            pair_indices.append((i, j))
    return pair_products, pair_indices

def _find_blocking_pairs(S_list, x):
    """
    Find which pairs in S block x from being added.
    Returns list of (a, b) pairs that conflict with x.
    """
    blockers = []
    n = len(S_list)
    
    # 1. x divides a*b
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            if (a * b) % x == 0:
                blockers.append((a, b))
    
    # 2. a divides x*b
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if i == j:
                continue
            b = S_list[j]
            if (x * b) % a == 0:
                blockers.append((a, b))
    
    return blockers

def _polish_pass(current_set, current_list, non_members, deadline):
    """
    Final high-precision polish: systematically try all adds, then all 1-for-1 swaps,
    then all 2-for-2 swaps until no improvement.
    """
    improved = True
    max_rounds = 50
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        if not improved:
            break
        improved = False
        
        # Try all adds
        for idx in range(len(non_members)):
            if time.time() > deadline:
                break
            x = non_members[idx]
            if _can_add(current_list, x):
                current_set.add(x)
                current_list.append(x)
                non_members.pop(idx)
                improved = True
                break
        
        if improved:
            continue
        
        # Try all 1-for-1 swaps
        if len(current_set) > 0:
            for i in range(len(current_list)):
                if time.time() > deadline:
                    break
                y = current_list[i]
                temp_list = [v for v in current_list if v != y]
                for j in range(len(non_members)):
                    if time.time() > deadline:
                        break
                    x = non_members[j]
                    if _can_add(temp_list, x):
                        current_set.remove(y)
                        current_set.add(x)
                        current_list = temp_list + [x]
                        non_members[j] = y
                        improved = True
                        break
                if improved:
                    break
        
        if improved:
            continue
        
        # Try all 2-for-2 swaps
        if len(current_set) >= 2 and len(non_members) >= 2:
            found = False
            for i in range(len(current_list)):
                if time.time() > deadline:
                    break
                y1 = current_list[i]
                for j in range(i + 1, len(current_list)):
                    if time.time() > deadline:
                        break
                    y2 = current_list[j]
                    temp_list = [v for v in current_list if v != y1 and v != y2]
                    for k in range(len(non_members)):
                        if time.time() > deadline:
                            break
                        x1 = non_members[k]
                        if _can_add(temp_list, x1):
                            temp_list2 = temp_list + [x1]
                            for m in range(k + 1, len(non_members)):
                                if time.time() > deadline:
                                    break
                                x2 = non_members[m]
                                if _can_add(temp_list2, x2):
                                    current_set.discard(y1)
                                    current_set.discard(y2)
                                    current_set.add(x1)
                                    current_set.add(x2)
                                    current_list = temp_list + [x1, x2]
                                    non_members[k] = y1
                                    non_members[m] = y2
                                    found = True
                                    break
                    if found:
                        break
                if found:
                    break
            if found:
                improved = True
    
    return current_set, current_list, non_members

def construct():
    deadline = time.time() + 55.0
    start_time = time.time()
    
    all_nums = list(range(2, 257))
    
    best_set = set()
    best_size = 0
    
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0: return False
        return True
            
    primes = [x for x in all_nums if is_prime(x)]
    composites = [x for x in all_nums if not is_prime(x)]
    
    orders = [
        list(range(2, 257)),
        list(range(256, 1, -1)),
        primes + composites,
        composites + primes,
    ]
    
    rng = random.Random(42)
    for _ in range(30):
        shuffled = all_nums.copy()
        rng.shuffle(shuffled)
        orders.append(shuffled)
        
    # Phase 1: Greedy construction to get initial solution
    for order in orders:
        if time.time() > deadline:
            break
        S, S_list = _greedy_construct(order, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Phase 2: Build clause map for Max-SAT-style scoring
    if time.time() < deadline - 10.0:
        clause_map = _build_clause_map(all_nums)
        
        # Phase 3: Max-SAT local search to improve
        current_list = list(best_set)
        
        # Run multiple Max-SAT local search iterations with different random seeds
        maxsat_deadline = deadline - 5.0
        for seed in range(10):
            if time.time() > maxsat_deadline:
                break
            rng_ms = random.Random(seed)
            improved_list = _maxsat_local_search(
                current_list, all_nums, clause_map, maxsat_deadline, rng_ms
            )
            
            # Validate the result
            if _is_valid(improved_list):
                if len(improved_list) > best_size:
                    best_size = len(improved_list)
                    best_set = set(improved_list)
                    current_list = improved_list
        
        # Phase 4: Final polish with traditional local search
        if time.time() < deadline - 2.0:
            current_set = best_set.copy()
            current_list = list(current_set)
            non_members = [x for x in all_nums if x not in current_set]
            
            # Quick polish pass
            no_improve = 0
            while time.time() < deadline - 2.0 and no_improve < 20:
                improved = False
                
                # Try adds
                for idx in range(len(non_members)):
                    if time.time() > deadline - 2.0:
                        break
                    x = non_members[idx]
                    if _can_add(current_list, x):
                        current_set.add(x)
                        current_list.append(x)
                        non_members.pop(idx)
                        improved = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = current_set.copy()
                        break
                
                if improved:
                    no_improve = 0
                    continue
                
                # Try 1-for-1 swaps
                found_swap = False
                for i in range(len(current_list)):
                    if time.time() > deadline - 2.0:
                        break
                    y = current_list[i]
                    temp_list = [v for v in current_list if v != y]
                    for j in range(len(non_members)):
                        if time.time() > deadline - 2.0:
                            break
                        x = non_members[j]
                        if _can_add(temp_list, x):
                            current_set.remove(y)
                            current_set.add(x)
                            current_list = temp_list + [x]
                            non_members[j] = y
                            found_swap = True
                            if len(current_set) > best_size:
                                best_size = len(current_set)
                                best_set = current_set.copy()
                            break
                    if found_swap:
                        break
                
                if found_swap:
                    no_improve = 0
                    continue
                
                no_improve += 1
    
    result = list(best_set)
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
