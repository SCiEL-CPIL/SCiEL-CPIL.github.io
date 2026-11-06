"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
    N = 180
    
    # Precompute all bad triples: distinct a,b,c in [1,180] with 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c => c = ab/(b-a) where b > a
    # a is always the smallest since 1/a > 1/b and 1/a > 1/c
    
    bad_triples = set()
    for a in range(1, 181):
        for b in range(a + 1, 181):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c > a and c <= 180 and c != b:
                    bad_triples.add(frozenset((a, b, c)))
    
    triples_list = list(bad_triples)
    n_triples = len(triples_list)
    
    # For each element, store indices of triples containing it
    triples_by_elem = defaultdict(list)
    for i, t in enumerate(triples_list):
        for elem in t:
            triples_by_elem[elem].append(i)
    
    # For each triple, store its three elements as a tuple
    triple_elems = [tuple(sorted(t)) for t in triples_list]
    
    def is_valid(S):
        """Check if set S has no bad triple."""
        S_set = set(S)
        for t in triples_list:
            if t.issubset(S_set):
                return False
        return True
    
    def count_conflicts(S, x):
        """Count how many bad triples would be formed if we add x to S."""
        S_set = set(S)
        count = 0
        for idx in triples_by_elem[x]:
            t = triples_list[idx]
            others = t - {x}
            if others.issubset(S_set):
                count += 1
        return count
    
    def greedy_construct(order):
        S = set()
        for x in order:
            if count_conflicts(S, x) == 0:
                S.add(x)
        return S
    
    # --- Phase 1: Try MILP via scipy to get a strong initial solution ---
    best_S = set(range(91, 181))  # fallback
    best_size = len(best_S)
    
    # Quick greedy baselines
    S_greedy_large = set(range(91, 181))
    for x in range(1, 91):
        if count_conflicts(S_greedy_large, x) == 0:
            S_greedy_large.add(x)
    if len(S_greedy_large) > best_size:
        best_S = S_greedy_large
        best_size = len(best_S)
    
    random.seed(42)
    for trial in range(100):
        if time.time() > deadline - 30:
            break
        order = list(range(1, 181))
        random.shuffle(order)
        S = greedy_construct(order)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
    
    # Try MILP if time permits
    try:
        from scipy.optimize import linprog
        if time.time() < deadline - 40:
            # Maximize sum(x_i) subject to: for each triple (a,b,c), x_a + x_b + x_c <= 2
            # Variables x_i in {0,1}, but we relax to [0,1] and round
            c_obj = [-1.0] * N  # minimize -sum = maximize sum
            A_ub = []
            b_ub = []
            for t in triples_list:
                row = [0.0] * N
                for elem in t:
                    row[elem - 1] = 1.0
                A_ub.append(row)
                b_ub.append(2.0)
            
            bounds = [(0, 1)] * N
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success and res.x is not None:
                # Round the LP solution: take elements with x_i > 0.5
                S_lp = set()
                for i in range(N):
                    if res.x[i] > 0.5:
                        S_lp.add(i + 1)
                
                # Fix any violations by removing elements
                changed = True
                while changed and time.time() < deadline - 35:
                    changed = False
                    for t in triples_list:
                        if t.issubset(S_lp):
                            # Remove the element in this triple with highest LP value (least "important")
                            min_val_elem = min(t, key=lambda e: res.x[e - 1])
                            S_lp.discard(min_val_elem)
                            changed = True
                
                if len(S_lp) > best_size:
                    best_S = S_lp
                    best_size = len(S_lp)
    except Exception:
        pass
    
    # --- Phase 2: Local search to improve ---
    
    # Phase 2a: Try to add elements not in best_S
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        candidates = [x for x in range(1, 181) if x not in best_S]
        random.shuffle(candidates)
        for x in candidates:
            if count_conflicts(best_S, x) == 0:
                best_S.add(x)
                improved = True
                best_size = len(best_S)
    
    # Phase 2b: Swap-based local search (remove one, add another)
    for _ in range(200):
        if time.time() > deadline - 8:
            break
        S = set(best_S)
        elems = list(S)
        random.shuffle(elems)
        for rem in elems[:30]:
            S_test = set(S)
            S_test.discard(rem)
            # Try to add elements not in S_test
            added_any = False
            for add in range(1, 181):
                if add not in S_test and add != rem and count_conflicts(S_test, add) == 0:
                    S_test.add(add)
                    added_any = True
            if len(S_test) > len(S):
                best_S = S_test
                best_size = len(S_test)
                break
    
    # Phase 2c: Multi-element removal and re-add
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        S = set(best_S)
        elems = list(S)
        random.shuffle(elems)
        for num_remove in [1, 2, 3]:
            if time.time() > deadline - 5:
                break
            S_test = set(S)
            for rem in elems[:num_remove]:
                S_test.discard(rem)
            # Re-add greedily
            candidates = [x for x in range(1, 181) if x not in S_test]
            random.shuffle(candidates)
            for x in candidates:
                if count_conflicts(S_test, x) == 0:
                    S_test.add(x)
            if len(S_test) > len(S):
                best_S = S_test
                best_size = len(S_test)
                break
    
    # Phase 2d: Simulated-annealing-like local search with pair swaps
    for _ in range(300):
        if time.time() > deadline - 3:
            break
        S = set(best_S)
        elems_in = list(S)
        elems_out = [x for x in range(1, 181) if x not in S]
        if not elems_in or not elems_out:
            break
        rem = random.choice(elems_in)
        add = random.choice(elems_out)
        S_test = set(S)
        S_test.discard(rem)
        if count_conflicts(S_test, add) == 0:
            S_test.add(add)
            # Try to add more
            for x in elems_out:
                if x not in S_test and x != add and count_conflicts(S_test, x) == 0:
                    S_test.add(x)
        if len(S_test) > len(S):
            best_S = S_test
            best_size = len(S_test)
    
    # Phase 2e: Final polish - try all single-element additions
    for x in range(1, 181):
        if x not in best_S and count_conflicts(best_S, x) == 0:
            best_S.add(x)
            best_size = len(best_S)
    
    # Final verification
    if not is_valid(best_S):
        # Remove elements from violated triples
        changed = True
        while changed:
            changed = False
            for t in triples_list:
                if t.issubset(best_S):
                    # Remove the largest element from the triple
                    best_S.discard(max(t))
                    changed = True
        if not best_S:
            best_S = set(range(91, 181))
    
    return list(best_S)

# EVOLVE-BLOCK-END
