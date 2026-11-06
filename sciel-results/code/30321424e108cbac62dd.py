"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 55
    TARGET = 200
    
    # Helper to check if a set of numbers has a subset summing to TARGET
    # Uses bitset-like logic with a set of reachable sums
    def is_valid_subset(S):
        sums = {0}
        for x in S:
            new_sums = set()
            for s in sums:
                ns = s + x
                if ns == TARGET:
                    return False
                if ns < TARGET:
                    new_sums.add(ns)
            sums |= new_sums
        return True

    def get_reachable_sums(S):
        sums = {0}
        for x in S:
            new_sums = set()
            for s in sums:
                ns = s + x
                if ns < TARGET:
                    new_sums.add(ns)
            sums |= new_sums
        return sums

    # 1. Generate initial solution using greedy hitting set on [51, 100]
    UNIVERSE_HIGH = list(range(51, 101))
    bad_triples = []
    for i in range(len(UNIVERSE_HIGH)):
        a = UNIVERSE_HIGH[i]
        for j in range(i + 1, len(UNIVERSE_HIGH)):
            b = UNIVERSE_HIGH[j]
            rem = TARGET - a - b
            if b < rem <= 100 and rem in UNIVERSE_HIGH:
                bad_triples.append((a, b, rem))
                
    def greedy_hitting_set(triples, rng):
        H = set()
        active = list(triples)
        while active:
            counts = {}
            for t in active:
                for v in t:
                    counts[v] = counts.get(v, 0) + 1
            max_count = max(counts.values())
            candidates = [v for v, c in counts.items() if c == max_count]
            best_elem = rng.choice(candidates)
            H.add(best_elem)
            active = [t for t in active if best_elem not in t]
            if time.time() > deadline - 10:
                break
        return H

    def is_valid_hitting_set(H, triples):
        for t in triples:
            if not any(v in H for v in t):
                return False
        return True

    rng = random.Random(42)
    best_H = None
    best_size = float('inf')
    
    # Run a few trials to get a good starting point
    for i in range(50):
        if time.time() > deadline - 10:
            break
        H = greedy_hitting_set(bad_triples, rng)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for h in list(H):
                H_test = H - {h}
                if is_valid_hitting_set(H_test, bad_triples):
                    H = H_test
                    improved = True
                    break
        if len(H) < best_size:
            best_size = len(H)
            best_H = H.copy()

    if best_H is not None:
        S = set(x for x in UNIVERSE_HIGH if x not in best_H)
    else:
        S = set(UNIVERSE_HIGH)
        
    # Ensure S is valid (it should be, but just in case)
    if not is_valid_subset(S):
        # Fallback: remove elements until valid
        sums = get_reachable_sums(S)
        # This is a simple repair
        for x in sorted(S, reverse=True):
            if is_valid_subset(S):
                break
            S.remove(x)

    # 2. Local Search: Try to add elements from [1, 50]
    # We want to maximize cardinality.
    # We can also try to swap elements to free up space or allow additions.
    
    # First, try to add any element from 1..50
    added_any = True
    while added_any and time.time() < deadline - 5:
        added_any = False
        candidates = [x for x in range(1, 51) if x not in S]
        # Sort by value? Maybe larger values are "safer" or "riskier"?
        # Let's try random order to explore
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if x in S:
                continue
            S_test = S | {x}
            if is_valid_subset(S_test):
                S = S_test
                added_any = True
                # Restart loop to check other candidates with new S
                break

    # 3. Local Search: Swaps
    # Try to replace an element y in S with x not in S if it allows adding more later?
    # Or just if it maintains validity. But since we want max cardinality,
    # a 1-for-1 swap doesn't change size.
    # However, removing a "blocking" element might allow adding TWO elements.
    # So we should try: Remove y, then try to add as many as possible.
    # If net gain > 0, keep it.
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        S_list = list(S)
        rng.shuffle(S_list)
        for y in S_list:
            if time.time() > deadline - 2:
                break
            if y not in S:
                continue
            S_temp = S - {y}
            # Try to add elements to S_temp
            added = 0
            candidates = [x for x in range(1, 101) if x not in S_temp and x != y]
            # Prioritize adding small numbers? Or large?
            # Let's try all
            for x in candidates:
                if time.time() > deadline - 2:
                    break
                S_test = S_temp | {x}
                if is_valid_subset(S_test):
                    S_temp = S_temp | {x}
                    added += 1
            # If we added more than we removed (1), we have a gain
            if added > 1:
                S = S_temp
                improved = True
                break # Restart while loop

    return list(S)
# EVOLVE-BLOCK-END
