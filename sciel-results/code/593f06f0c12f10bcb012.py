import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    numbers = list(range(1, n + 1))
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Precompute LCM table for all pairs
    lcm_table = {}
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[(i, j)] = lcm(i, j)
    
    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]
    
    # Check if adding x to set S creates a bad triple
    def can_add_fast(x, S):
        groups = {}
        for a in S:
            L = get_lcm(x, a)
            if L not in groups:
                groups[L] = []
            groups[L].append(a)
        
        for L, elems in groups.items():
            if len(elems) < 2:
                continue
            for i in range(len(elems)):
                for j in range(i + 1, len(elems)):
                    if get_lcm(elems[i], elems[j]) == L:
                        return False
        return True
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(n)
    prime_powers = set()
    for p in primes:
        power = p
        while power <= n:
            prime_powers.add(power)
            power *= p
    
    core_set = prime_powers | {1}
    
    def num_distinct_primes(x):
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    remaining = [x for x in range(1, n + 1) if x not in core_set]
    
    # Multiple randomized greedy passes
    best_set = set(core_set)
    
    orderings = []
    orderings.append(sorted(remaining, key=lambda x: (num_distinct_primes(x), x)))
    orderings.append(sorted(remaining, key=lambda x: (num_distinct_primes(x), -x)))
    orderings.append(sorted(remaining, key=lambda x: x))
    orderings.append(sorted(remaining, key=lambda x: -x))
    
    for _ in range(8):
        rand_order = remaining[:]
        random.shuffle(rand_order)
        orderings.append(rand_order)
    
    for ordering in orderings:
        if time.time() > deadline - 15.0:
            break
        
        current = set(core_set)
        for x in ordering:
            if time.time() > deadline - 15.0:
                break
            if x in current:
                continue
            if can_add_fast(x, current):
                current.add(x)
        
        if len(current) > len(best_set):
            best_set = set(current)
    
    # WalkSAT-style clause-weighting local search
    all_numbers = set(range(1, n + 1))
    
    # Precompute all forbidden triples
    forbidden_triples = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lab = get_lcm(i, j)
            for k in range(j + 1, n + 1):
                if get_lcm(i, k) == lab and get_lcm(j, k) == lab:
                    forbidden_triples.append((i, j, k))
    
    # Build adjacency: for each element, which triples it participates in
    triple_index = {i: [] for i in range(1, n + 1)}
    for idx, (a, b, c) in enumerate(forbidden_triples):
        triple_index[a].append(idx)
        triple_index[b].append(idx)
        triple_index[c].append(idx)
    
    # Count how many triples each pair is in (for fast violation check)
    pair_triple_count = {}
    for (a, b, c) in forbidden_triples:
        for p, q in [(a, b), (a, c), (b, c)]:
            if p <= q:
                pair_triple_count[(p, q)] = pair_triple_count.get((p, q), 0) + 1
    
    # WalkSAT: flip one variable at a time, guided by clause weights
    def count_violations(S):
        """Count number of violated triples in set S."""
        count = 0
        S_set = set(S)
        for (a, b, c) in forbidden_triples:
            if a in S_set and b in S_set and c in S_set:
                count += 1
        return count
    
    def walk_sat_local_search(current_set, max_steps=50000):
        """WalkSAT-style local search with clause weighting."""
        current = set(current_set)
        best = set(current)
        best_size = len(best)
        
        # Initial violation count
        cur_viol = count_violations(current)
        
        # Clause weights (soft constraints)
        clause_weight = {idx: 1.0 for idx in range(len(forbidden_triples))}
        
        steps_since_improve = 0
        max_steps_no_improve = 5000
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            if steps_since_improve > max_steps_no_improve:
                # Reset clause weights and try a random restart
                for idx in clause_weight:
                    clause_weight[idx] = 1.0
                # Random perturbation: flip 2-3 random elements
                all_nums = list(all_numbers)
                for _ in range(random.randint(2, 3)):
                    x = random.choice(all_nums)
                    if x in current:
                        current.discard(x)
                    else:
                        current.add(x)
                cur_viol = count_violations(current)
                steps_since_improve = 0
                continue
            
            # Choose a flip: try adding or removing a random element
            # Weight by how many clauses it affects
            candidates = []
            for x in all_numbers:
                if x in current:
                    # Removing x: how many violated triples does it fix?
                    fix_count = 0
                    for tidx in triple_index[x]:
                        a, b, c = forbidden_triples[tidx]
                        if a in current and b in current and c in current:
                            fix_count += clause_weight[tidx]
                    candidates.append((x, 'remove', fix_count))
                else:
                    # Adding x: how many new violations does it create?
                    create_count = 0
                    for tidx in triple_index[x]:
                        a, b, c = forbidden_triples[tidx]
                        other1, other2 = (a, b) if x == c else ((a, c) if x == b else (b, c))
                        if other1 in current and other2 in current:
                            create_count += clause_weight[tidx]
                    candidates.append((x, 'add', create_count))
            
            if not candidates:
                break
            
            # Weighted random selection: prefer flips that fix more or create fewer
            # For removals, higher fix_count is better
            # For additions, lower create_count is better (0 is ideal)
            removals = [(x, f) for x, action, f in candidates if action == 'remove']
            additions = [(x, f) for x, action, f in candidates if action == 'add']
            
            # 60% chance to try an addition if any are safe (create_count == 0)
            safe_adds = [x for x, f in additions if f == 0]
            
            if safe_adds and random.random() < 0.7:
                x = random.choice(safe_adds)
                current.add(x)
                cur_viol = count_violations(current)
            elif removals and (not safe_adds or random.random() < 0.3):
                # Pick removal that fixes most violations
                x, f = max(removals, key=lambda t: t[1])
                if f > 0:
                    current.discard(x)
                    cur_viol = count_violations(current)
            elif additions:
                # Pick addition that creates fewest violations
                x, f = min(additions, key=lambda t: t[1])
                current.add(x)
                cur_viol = count_violations(current)
            else:
                break
            
            # Update clause weights for violated clauses
            for tidx in range(len(forbidden_triples)):
                a, b, c = forbidden_triples[tidx]
                if a in current and b in current and c in current:
                    clause_weight[tidx] = min(clause_weight[tidx] * 1.2, 100.0)
                else:
                    clause_weight[tidx] = max(clause_weight[tidx] * 0.9, 0.1)
            
            # Track best
            if len(current) > best_size:
                best = set(current)
                best_size = len(current)
                steps_since_improve = 0
            else:
                steps_since_improve += 1
        
        return best
    
    # Run WalkSAT from current best
    improved = True
    ws_passes = 0
    while improved and time.time() < deadline - 5.0 and ws_passes < 10:
        improved = False
        ws_passes += 1
        ws_result = walk_sat_local_search(best_set, max_steps=30000)
        if len(ws_result) > len(best_set):
            best_set = ws_result
            improved = True
    
    # Final validation: remove any elements causing violations
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        S = sorted(best_set)
        for i in range(len(S)):
            if time.time() > deadline - 1.0:
                break
            for j in range(i + 1, len(S)):
                lab = get_lcm(S[i], S[j])
                for k in range(j + 1, len(S)):
                    if get_lcm(S[i], S[k]) == lab and get_lcm(S[j], S[k]) == lab:
                        # Remove the element in the triple with highest triple_index count
                        counts = [len(triple_index[S[i]]), len(triple_index[S[j]]), len(triple_index[S[k]])]
                        max_idx = counts.index(max(counts))
                        to_remove = S[i] if max_idx == 0 else (S[j] if max_idx == 1 else S[k])
                        best_set.discard(to_remove)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    
    return list(best_set) if len(best_set) > 0 else list(core_set)
