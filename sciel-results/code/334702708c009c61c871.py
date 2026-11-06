import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if len(S) < 3:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def build_conflicts(values):
        """Precompute all conflict triples and per-vertex conflict sets."""
        n = len(values)
        val_to_idx = {v: i for i, v in enumerate(values)}
        # conflicts[i] = set of (j, k) such that values[i]+values[j]=values[k] or values[i]+values[k]=values[j]
        # For can_add: adding vertex v creates conflict if there exist i,j in S with values[i]+values[j]=values[v] or values[i]+values[v]=values[j] or values[v]+values[j]=values[i]
        # Equivalent: v conflicts with pair (i,j) if values[i]+values[j]==values[v], or values[i]+values[v]==values[j], or values[v]+values[j]==values[i]
        
        # Precompute: for each vertex v, the set of pairs (i,j) that would conflict with v
        # Also precompute: for each pair (i,j), which vertex k satisfies values[i]+values[j]==values[k]
        
        pair_to_third = {}  # (i,j) -> k if values[i]+values[j]==values[k]
        for i in range(n):
            for j in range(i, n):
                s = values[i] + values[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    pair_to_third[(i, j)] = k
        
        # conflict_pairs[v] = set of (i,j) where adding v would create a conflict
        # Adding v conflicts if:
        #   1. exists i,j in S: values[i]+values[j]==values[v]  -> (i,j) in pair_to_third with pair_to_third[(i,j)]==v
        #   2. exists i in S: values[i]+values[v]==values[j] for some j in S -> values[j]-values[v]==values[i], i.e., (i, j-v) ... 
        #      Actually: values[i] + values[v] = values[j] means values[j] - values[v] = values[i]
        #      So for fixed v, the pairs (i,j) such that values[i]+values[v]==values[j] are those where values[j]-values[i]==values[v]
        #   3. exists j in S: values[v]+values[j]==values[i] for some i in S -> values[i]-values[j]==values[v], same as case 2 with i,j swapped
        
        # Simpler: for each v, conflict_pairs[v] = {(i,j) : values[i]+values[j]==values[v]} ∪ {(i,j) : values[i]+values[v]==values[j]} ∪ {(i,j) : values[v]+values[j]==values[i]}
        # Note cases 2 and 3 are the same set of unordered pairs {i,j} where |values[i]-values[j]|==|values[v]|... no wait.
        
        # Let me think differently. A set S is sum-free iff for all distinct (or not) a,b,c in S, a+b≠c.
        # Adding v to S creates a conflict iff:
        #   ∃ a,b ∈ S: a+b = v  (i.e., (a,b) is a pair summing to v)
        #   ∃ a ∈ S: a+v = b for some b ∈ S  (i.e., v = b-a for some a,b ∈ S)
        #   ∃ b ∈ S: v+b = a for some a ∈ S  (same as above)
        
        # So: conflict if v is in {a+b : a,b ∈ S} or v is in {b-a : a,b ∈ S}
        
        # Precompute for each v:
        #   sum_pairs[v] = {(a_idx, b_idx) : values[a_idx] + values[b_idx] == values[v]}
        #   diff_pairs[v] = {(a_idx, b_idx) : values[b_idx] - values[a_idx] == values[v]}
        
        sum_pairs = defaultdict(set)  # v_idx -> set of (a_idx, b_idx)
        diff_pairs = defaultdict(set)  # v_idx -> set of (a_idx, b_idx) where values[b]-values[a]==values[v]
        
        for i in range(n):
            for j in range(i, n):
                s = values[i] + values[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    sum_pairs[k].add((i, j))
                
                d = values[j] - values[i]
                if d in val_to_idx:
                    k = val_to_idx[d]
                    diff_pairs[k].add((i, j))
                d2 = values[i] - values[j]
                if d2 in val_to_idx:
                    k = val_to_idx[d2]
                    diff_pairs[k].add((i, j))
        
        # conflict_pairs[v] = sum_pairs[v] ∪ diff_pairs[v]
        conflict_pairs = {}
        for v in range(n):
            cp = set(sum_pairs.get(v, set())) | set(diff_pairs.get(v, set()))
            conflict_pairs[v] = cp
        
        # degree[v] = number of pairs that conflict with v (rough measure of how "constrained" v is)
        degree = np.array([len(conflict_pairs[v]) for v in range(n)], dtype=np.int32)
        
        return conflict_pairs, degree, val_to_idx

    def can_add_idx(v, active, conflict_pairs):
        """Check if vertex v can be added to active set without creating conflicts."""
        for (i, j) in conflict_pairs[v]:
            if i in active and j in active:
                return False
        return True

    def greedy_add_idx(values, order, conflict_pairs):
        """Greedy add in given order, using index-based conflict check."""
        active = set()
        for v in order:
            if v in active:
                continue
            if can_add_idx(v, active, conflict_pairs):
                active.add(v)
        return active

    def local_search_idx(active, n, values, conflict_pairs, degree, time_limit=5.0, seed=42):
        """Local search: try to add inactive vertices, swap high-degree for low-degree."""
        active = set(active)
        rng = random.Random(seed)
        start = time.time()
        best_active = set(active)
        best_size = len(active)
        
        # Phase 1: Try adding all inactive vertices greedily
        improved = True
        while improved and (time.time() - start) < time_limit:
            improved = False
            for v in range(n):
                if v not in active:
                    if can_add_idx(v, active, conflict_pairs):
                        active.add(v)
                        improved = True
                        if len(active) > best_size:
                            best_size = len(active)
                            best_active = set(active)
        
        # Phase 2: Swap operations - remove one vertex, try to add others
        improved = True
        while improved and (time.time() - start) < time_limit:
            improved = False
            # Sort active by degree descending (most constrained first)
            active_sorted = sorted(active, key=lambda x: degree[x], reverse=True)
            for rem in active_sorted:
                if (time.time() - start) >= time_limit:
                    break
                test_active = active - {rem}
                added = 0
                # Try adding all inactive in order of increasing degree (least constrained first)
                inactive_sorted = sorted([v for v in range(n) if v not in test_active], key=lambda x: degree[x])
                for v in inactive_sorted:
                    if can_add_idx(v, test_active, conflict_pairs):
                        test_active.add(v)
                        added += 1
                if len(test_active) > len(active):
                    active = test_active
                    if len(active) > best_size:
                        best_size = len(active)
                        best_active = set(active)
                    improved = True
                    break
        
        # Phase 3: Random restarts within local search
        if (time.time() - start) < time_limit:
            for _ in range(5):
                if (time.time() - start) >= time_limit:
                    break
                # Perturb: remove a random subset, re-add
                perturbed = set(active)
                if perturbed:
                    to_remove = rng.sample(list(perturbed), min(3, len(perturbed)))
                    for v in to_remove:
                        perturbed.discard(v)
                # Re-add greedily
                improved2 = True
                while improved2 and (time.time() - start) < time_limit:
                    improved2 = False
                    for v in range(n):
                        if v not in perturbed:
                            if can_add_idx(v, perturbed, conflict_pairs):
                                perturbed.add(v)
                                improved2 = True
                if len(perturbed) > best_size:
                    best_size = len(perturbed)
                    best_active = set(perturbed)
        
        return best_active

    def spectral_independent_set(values, conflict_pairs, degree, time_limit=3.0):
        """Use spectral methods to find a good initial independent set."""
        start = time.time()
        n = len(values)
        if n <= 2:
            return set(range(n))
        
        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=np.float64)
        for v in range(n):
            for (i, j) in conflict_pairs[v]:
                adj[v, i] = 1
                adj[v, j] = 1
        
        # Compute Laplacian
        deg = adj.sum(axis=1)
        L = np.diag(deg) - adj
        
        try:
            evals, evecs = np.linalg.eigh(L)
            fiedler = evecs[:, 1]
        except:
            return set(range(n))
        
        # Partition based on Fiedler vector
        median_val = np.median(fiedler)
        part_A = set(i for i in range(n) if fiedler[i] > median_val)
        part_B = set(i for i in range(n) if fiedler[i] <= median_val)
        
        best_indep = set()
        for part in [part_A, part_B]:
            if not part:
                continue
            current = set(part)
            improved = True
            while improved and (time.time() - start) < time_limit:
                improved = False
                if not current:
                    break
                max_deg = -1
                max_v = None
                for v in current:
                    internal_deg = sum(1 for u in current if u != v and adj[v, u] == 1)
                    if internal_deg > max_deg:
                        max_deg = internal_deg
                        max_v = v
                if max_deg == 0:
                    break
                if max_v is not None:
                    current.discard(max_v)
                    improved = True
            if len(current) > len(best_indep):
                best_indep = current
        
        return best_indep

    def solve_one(values, time_budget=15.0):
        if time.time() > deadline:
            return []
        
        n = len(values)
        if n == 0:
            return []
        
        # Precompute conflict structure
        conflict_pairs, degree, val_to_idx = build_conflicts(values)
        
        best_active = set()
        
        # Generate multiple initial solutions using different orderings
        orderings = []
        
        # Strategy 1: Odd numbers first
        odds = [val_to_idx[x] for x in values if x % 2 != 0]
        evens = [val_to_idx[x] for x in values if x % 2 == 0]
        orderings.append(odds + evens)
        
        # Strategy 2: Large positive, then large negative
        max_v = max(values)
        min_v = min(values)
        large_pos = [val_to_idx[x] for x in values if x > max_v / 2.0]
        large_neg = [val_to_idx[x] for x in values if x < min_v / 2.0]
        rest = [val_to_idx[x] for x in values if x <= max_v / 2.0 and x >= min_v / 2.0]
        orderings.append(large_pos + large_neg + rest)
        
        # Strategy 3: Sorted by absolute value descending
        orderings.append(sorted(range(n), key=lambda i: abs(values[i]), reverse=True))
        
        # Strategy 4: Sorted by value ascending
        orderings.append(sorted(range(n), key=lambda i: values[i]))
        
        # Strategy 5: Sorted by value descending
        orderings.append(sorted(range(n), key=lambda i: values[i], reverse=True))
        
        # Strategy 6: Low degree first (least constrained)
        orderings.append(sorted(range(n), key=lambda i: degree[i]))
        
        # Strategy 7: High degree first (most constrained, to see if we can include them)
        orderings.append(sorted(range(n), key=lambda i: degree[i], reverse=True))
        
        # Strategy 8-10: Random shuffles
        rng = random.Random(42)
        for _ in range(3):
            shuffled = list(range(n))
            rng.shuffle(shuffled)
            orderings.append(shuffled)
        
        # Evaluate each ordering
        for order in orderings:
            active = greedy_add_idx(values, order, conflict_pairs)
            if len(active) > len(best_active):
                best_active = set(active)
        
        # Spectral method if time permits
        remaining = deadline - time.time()
        if remaining > 8 and n > 30:
            spec_time = min(3.0, remaining * 0.3)
            spec_active = spectral_independent_set(values, conflict_pairs, degree, time_limit=spec_time)
            if len(spec_active) > len(best_active):
                best_active = set(spec_active)
        
        # Local search on best
        remaining = deadline - time.time()
        if remaining > 5:
            ls_time = min(time_budget * 0.5, remaining - 3)
            if ls_time > 1:
                improved = local_search_idx(best_active, n, values, conflict_pairs, degree, time_limit=ls_time)
                if len(improved) > len(best_active):
                    best_active = improved
        
        # Additional random restarts if time permits
        remaining = deadline - time.time()
        if remaining > 8:
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                shuffled = list(range(n))
                rng.shuffle(shuffled)
                active = greedy_add_idx(values, shuffled, conflict_pairs)
                if len(active) >= len(best_active):
                    ls_time = min(2.0, deadline - time.time() - 3)
                    if ls_time > 0.5:
                        improved = local_search_idx(active, n, values, conflict_pairs, degree, time_limit=ls_time, seed=rng.randint(0, 100000))
                        if len(improved) > len(best_active):
                            best_active = improved
        
        # Convert back to values
        result = [values[i] for i in best_active]
        
        # Final verification
        if not is_sum_free(result):
            # Fallback to odds
            result = [x for x in values if x % 2 != 0]
        
        return result

    results = []
    time_alloc = {60: 12.0, 120: 25.0, 240: 70.0}
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            results.append([])
            continue
        
        values = instance(size, seed)
        budget = time_alloc.get(size, 15.0)
        remaining = deadline - time.time()
        budget = min(budget, max(2.0, remaining - 3.0))
        
        subset = solve_one(values, time_budget=budget)
        
        results.append(subset)
    
    return results
