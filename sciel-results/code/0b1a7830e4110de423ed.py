"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from random import Random

DEADLINE = time.time() + 85  # safety margin before 90s kill


def _bad_triples(n=128):
    """Precompute all bad triples as sets of indices."""
    nums = list(range(1, n + 1))
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = nums[i], nums[j]
            gab = gcd(a, b)
            for k in range(j + 1, n):
                c = nums[k]
                if gcd(a, c) == gab and gcd(b, c) == gab:
                    triples.append((i, j, k))
    return nums, triples


def _check_add(S, x, triples_by_pair):
    """Check if adding index x to set S is valid. triples_by_pair maps (i,j) to set of k."""
    for i in S:
        for j in S:
            if i >= j:
                continue
            if x in triples_by_pair.get((i, j), set()):
                return False
    return True


def _check_add_fast(S, x, pair_conflicts):
    """Fast check using pair_conflicts matrix."""
    for i in S:
        if i >= x:
            break
        if x in pair_conflicts[i][x]:
            return False
    return True


def _build_pair_conflicts(n=128):
    nums = list(range(1, n + 1))
    pair_conflicts = [[set() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = nums[i], nums[j]
            gab = gcd(a, b)
            for k in range(n):
                if k == i or k == j:
                    continue
                c = nums[k]
                if gcd(a, c) == gab and gcd(b, c) == gab:
                    pair_conflicts[i][j].add(k)
                    pair_conflicts[j][i].add(k)
    return nums, pair_conflicts


def _greedy(order, pair_conflicts, n):
    S = set()
    for x in order:
        if x not in S:
            if _check_add_fast(S, x, pair_conflicts):
                S.add(x)
    return S


def _local_search(S, rng, pair_conflicts, n, max_iter=3000):
    S = set(S)
    not_in = [x for x in range(n) if x not in S]
    improved = True
    while improved and not_in:
        improved = False
        for _ in range(max_iter):
            if time.time() > DEADLINE:
                return S
            if not not_in:
                break
            x = rng.choice(not_in)
            if _check_add_fast(S, x, pair_conflicts):
                S.add(x)
                not_in.remove(x)
                improved = True
    return S


def _simulate_annealing(S_init, rng, pair_conflicts, n, steps=80000):
    S = set(S_init)
    all_idx = list(range(n))
    current_size = len(S)
    best_S = set(S)
    best_size = current_size
    T = 5.0
    cooling = 0.99995
    
    for step in range(steps):
        if time.time() > DEADLINE:
            break
        
        move_type = rng.random()
        if move_type < 0.5:
            # Try to add
            candidates = [x for x in all_idx if x not in S]
            if not candidates:
                continue
            x = rng.choice(candidates)
            if _check_add_fast(S, x, pair_conflicts):
                S.add(x)
                new_size = len(S)
                delta = -(new_size - current_size)
                if delta <= 0 or rng.random() < pow(2.718, -delta / T):
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_S = set(S)
                else:
                    S.discard(x)
        elif move_type < 0.8:
            # Try to remove
            if S:
                x = rng.choice(list(S))
                S.discard(x)
                new_size = len(S)
                delta = -(new_size - current_size)
                if delta <= 0 or rng.random() < pow(2.718, -delta / T):
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_S = set(S)
                else:
                    S.add(x)
        else:
            # Swap
            if S:
                x_rm = rng.choice(list(S))
                S.discard(x_rm)
                candidates = [x for x in all_idx if x not in S]
                if candidates:
                    x_add = rng.choice(candidates)
                    if _check_add_fast(S, x_add, pair_conflicts):
                        S.add(x_add)
                        new_size = len(S)
                        delta = -(new_size - current_size)
                        if delta <= 0 or rng.random() < pow(2.718, -delta / T):
                            current_size = new_size
                            if current_size > best_size:
                                best_size = current_size
                                best_S = set(S)
                        else:
                            S.discard(x_add)
                            S.add(x_rm)
                    else:
                        S.add(x_rm)
        
        T *= cooling
        if T < 0.01:
            break
    
    return best_S


def _branch_and_bound(pair_conflicts, n, initial_lower_bound):
    """
    Branch and bound for maximum independent set in 3-uniform hypergraph.
    Uses a greedy upper bound based on degree heuristics.
    """
    best_set = set()
    best_size = initial_lower_bound
    
    # Precompute degree: number of bad triples each element participates in
    degrees = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            deg_ij = len(pair_conflicts[i][j])
            degrees[i] += deg_ij
            degrees[j] += deg_ij
    
    def upper_bound(candidates):
        """Greedy upper bound: repeatedly pick min-degree vertex."""
        if not candidates:
            return 0
        cand = set(candidates)
        bound = 0
        # Simple bound: |candidates| is trivially an upper bound
        # Better: use a greedy packing
        # For 3-uniform, a simple bound is |cand| but we can do better
        # Actually, let's just use |cand| as a weak bound and rely on pruning
        return len(cand)
    
    def bnb(candidates, current_set, depth=0):
        nonlocal best_set, best_size
        if time.time() > DEADLINE:
            return
        
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        # Upper bound: can we add all candidates?
        ub = len(candidates) + len(current_set)
        if ub <= best_size:
            return
        
        # Pick a vertex to branch on: choose one with max degree in candidates
        v = -1
        max_deg = -1
        for x in candidates:
            # Degree of x within candidates: number of other pairs that conflict
            deg = 0
            for y in candidates:
                if y != x and x in pair_conflicts[y][x]:
                    deg += 1
            if deg > max_deg:
                max_deg = deg
                v = x
        
        if v == -1:
            return
        
        # Branch 1: Include v
        # Check if v is compatible with current_set
        compatible = True
        for i in current_set:
            for j in current_set:
                if i >= j:
                    continue
                if v in pair_conflicts[i][j]:
                    compatible = False
                    break
            if not compatible:
                break
        
        if compatible:
            new_candidates = set(candidates)
            new_candidates.discard(v)
            # Remove all elements that conflict with v
            for x in list(new_candidates):
                # x conflicts with v if there exists y in new_candidates such that {v,x,y} is bad
                # i.e., v in pair_conflicts[x][y] for some y, OR x in pair_conflicts[v][y]
                # Actually, if we include v, we must exclude all x such that there's a y
                # where {v,x,y} is a bad triple. But also x must be compatible with current_set.
                # More precisely: after including v, x can be included only if:
                # 1. x is compatible with current_set ∪ {v}
                # 2. x is not in any bad triple with v and another candidate
                # Condition 2: for all y in new_candidates, {v,x,y} is not bad
                # i.e., x not in pair_conflicts[v][y] for all y in new_candidates
                # AND y not in pair_conflicts[v][x] (same thing)
                # AND v not in pair_conflicts[x][y] for all y
                # Simplification: remove x if exists y in candidates such that 
                # v in pair_conflicts[x][y] or x in pair_conflicts[v][y]
                # Actually the cleanest: after fixing v in the set, x is removable from candidates
                # if there exists y such that {v,x,y} is a bad triple.
                # {v,x,y} bad means v in pair_conflicts[x][y] AND x in pair_conflicts[v][y] AND y in pair_conflicts[v][x]
                # But we can just check: is there a y in new_candidates with x in pair_conflicts[v][y]?
                # That means {v,y,x} could be bad if also gcd conditions hold... 
                # Actually pair_conflicts[v][y] contains x iff {v,y,x} is a bad triple.
                # So x must be removed if x in pair_conflicts[v][y] for some y in new_candidates,
                # OR y in pair_conflicts[v][x] for some y (equivalent).
                # Wait, pair_conflicts[i][j] contains k iff {i,j,k} is bad.
                # So x in pair_conflicts[v][y] iff {v,y,x} is bad.
                # If x is in pair_conflicts[v][y] for some y in candidates, then {v,x,y} is bad,
                # so x cannot be in the set with v and y. But y might not be in the set.
                # This is the tricky part of hypergraph independent sets.
                # For correctness: x can be in the set with v only if for ALL y in the set,
                # {v,x,y} is not bad. Since we're branching, we need to be careful.
                # Simplest correct approach: just check pairwise with current_set ∪ {v}
                # and don't do the hyperedge removal (this makes it a standard graph MIS on
                # the "conflict graph" where x and y conflict if there's a bad triple containing both... no that's not right either).
                
                # Let me reconsider. The constraint is: no three elements form a bad triple.
                # If I fix v in the set, then for any pair (x,y) in the remaining set,
                # I need {v,x,y} to not be bad, AND {x,y,z} for any z in set to not be bad.
                # The second condition is handled recursively. The first condition means:
                # x and y can both be in the set only if {v,x,y} is not a bad triple.
                # So I should create a "conflict" between x and y if {v,x,y} is bad.
                # This effectively turns the remaining problem into a 2-uniform problem!
                
                # So: after fixing v, two elements x,y conflict if {v,x,y} is bad.
                # x in pair_conflicts[v][y] means {v,y,x} is bad, i.e., {v,x,y} is bad.
                # So the conflict graph on remaining candidates has edge (x,y) iff x in pair_conflicts[v][y].
                
                # For now, let's just do a simpler but correct check:
                # Remove x from new_candidates if x conflicts with v in the sense that
                # adding x would immediately create a bad triple with v and some element
                # already committed... but we haven't committed other elements yet.
                # 
                # Actually the simplest correct approach: just check compatibility with
                # current_set ∪ {v} and recurse. The hyperedge constraints involving
                # two future elements are handled by the recursion.
                pass
            
            # Simplified: just check pairwise compatibility with current_set + v
            final_new_candidates = set()
            for x in new_candidates:
                # Check x against current_set ∪ {v}
                ok = True
                for i in current_set:
                    if x in pair_conflicts[i][v] or x in pair_conflicts[v][i]:
                        # This doesn't directly tell us... 
                        # We need: for each pair (i,j) in current_set∪{v}, x not in pair_conflicts[i][j]
                        pass
                # Redo properly
                ok = True
                for i in current_set:
                    for j in current_set:
                        if i >= j:
                            continue
                        if x in pair_conflicts[i][j]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    # Also check with v
                    for i in current_set:
                        if x in pair_conflicts[i][v] or x in pair_conflicts[v][i]:
                            # {i,v,x} is bad? pair_conflicts[i][v] contains x iff {i,v,x} is bad
                            if x in pair_conflicts[i][v]:
                                ok = False
                                break
                if ok:
                    final_new_candidates.add(x)
            
            current_set.append(v)
            bnb(final_new_candidates, current_set, depth + 1)
            current_set.pop()
        
        # Branch 2: Exclude v
        new_candidates = set(candidates)
        new_candidates.discard(v)
        bnb(new_candidates, current_set, depth + 1)
    
    # Start BnB
    all_candidates = set(range(n))
    bnb(all_candidates, [], 0)
    
    return best_set


def construct():
    n = 128
    nums, pair_conflicts = _build_pair_conflicts(n)
    
    rng = Random(42)
    
    # Strategy 1: Multiple greedy orderings
    orders = [
        list(range(n)),
        list(range(n - 1, -1, -1)),
        sorted(range(n), key=lambda x: -nums[x]),
        sorted(range(n), key=lambda x: nums[x]),
    ]
    
    best = set()
    best_len = 0
    
    for order in orders:
        S = _greedy(order, pair_conflicts, n)
        if len(S) > best_len:
            best = S
            best_len = len(S)
    
    # Strategy 2: Random greedy + local search
    for i in range(200):
        if time.time() > DEADLINE:
            break
        order = list(range(n))
        rng.shuffle(order)
        S = _greedy(order, pair_conflicts, n)
        S = _local_search(S, rng, pair_conflicts, n, max_iter=2000)
        if len(S) > best_len:
            best = S
            best_len = len(S)
    
    # Strategy 3: Simulated annealing
    for i in range(10):
        if time.time() > DEADLINE:
            break
        order = list(range(n))
        rng.shuffle(order)
        S_init = _greedy(order, pair_conflicts, n)
        S = _simulate_annealing(S_init, rng, pair_conflicts, n, steps=60000)
        if len(S) > best_len:
            best = S
            best_len = len(S)
    
    # Strategy 4: SA from best
    if best:
        S = _simulate_annealing(set(best), rng, pair_conflicts, n, steps=80000)
        if len(S) > best_len:
            best = S
            best_len = len(S)
    
    # Strategy 5: Branch and bound with time limit
    if time.time() < DEADLINE - 5:
        bnb_result = _branch_and_bound(pair_conflicts, n, best_len)
        if len(bnb_result) > best_len:
            best = bnb_result
            best_len = len(bnb_result)
    
    # Strategy 6: Final SA polish
    if best and time.time() < DEADLINE - 2:
        S = _simulate_annealing(set(best), Random(999), pair_conflicts, n, steps=50000)
        if len(S) > best_len:
            best = S
            best_len = len(S)
    
    if not best:
        return [1]
    
    result = sorted(nums[i] for i in best)
    return result
# EVOLVE-BLOCK-END
