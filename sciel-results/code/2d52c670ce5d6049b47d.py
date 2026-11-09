import time
import math
import random

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin for 120s limit
    
    N = 512
    # Candidates 1..511, want max sum 1/(512-a) subject to pairwise coprime
    
    # Precompute prime factor sets
    primes = []
    for i in range(2, 512):
        is_p = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_p = 0
                break
        if is_p:
            primes.append(i)
    
    factors = [set() for _ in range(N)]
    for a in range(2, N):
        x = a
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                factors[a].add(p)
                while x % p == 0:
                    x //= p
        if x > 1:
            factors[a].add(x)
    # factors[1] = empty set (coprime with everything)
    
    def is_coprime_with_all(a, current_set):
        """Check if a is coprime with every element in current_set."""
        fa = factors[a]
        if not fa:
            return True
        for b in current_set:
            if fa & factors[b]:
                return False
        return True
    
    weights = {a: 1.0 / (N - a) for a in range(1, N)}
    
    # ---- Phase 1: Greedy with multiple orderings ----
    # Sort by weight descending (prefer numbers close to 512)
    sorted_by_weight = sorted(range(1, N), key=lambda x: weights[x], reverse=True)
    
    # Also try random perturbation orderings
    def greedy_solve(order):
        s = set()
        for a in order:
            if is_coprime_with_all(a, s):
                s.add(a)
        return s, sum(weights[a] for a in s)
    
    best_set, best_score = greedy_solve(sorted_by_weight)
    
    # Try several random orderings (biased toward high weight)
    for restart in range(20):
        # Perturbed ordering: mostly weight-sorted, but shuffle within groups
        order = list(range(1, N))
        random.shuffle(order)
        order.sort(key=lambda x: weights[x] + random.gauss(0, 0.01), reverse=True)
        s, sc = greedy_solve(order)
        if sc > best_score:
            best_score = sc
            best_set = s
        if time.time() > deadline - 60:
            break
    
    # ---- Phase 2: Local search (add, remove, swap) ----
    current_set = set(best_set)
    current_score = best_score
    
    all_candidates = list(range(1, N))
    
    def compatible_with(a, ref_set):
        fa = factors[a]
        if not fa:
            return True
        for b in ref_set:
            if fa & factors[b]:
                return False
        return True
    
    # Local search with restarts
    ls_restarts = 30
    steps_per_restart = 500
    
    for rs in range(ls_restarts):
        if time.time() > deadline - 30:
            break
        
        if rs > 0:
            # Sometimes restart from best, sometimes with random perturbation
            if random.random() < 0.5:
                current_set = set(best_set)
                current_score = best_score
            else:
                current_set = set(best_set)
                # Random perturbation: add a random compatible element
                for _ in range(3):
                    a = random.choice(all_candidates)
                    if a not in current_set and compatible_with(a, current_set):
                        current_set.add(a)
                        current_score = sum(weights[x] for x in current_set)
                        break
        
        improved = True
        while improved and steps_per_restart > 0:
            if time.time() > deadline - 30:
                break
            improved = False
            steps_per_restart -= 1
            
            # Move type selection
            move_type = random.random()
            
            if move_type < 0.45:
                # Add
                candidates_to_add = [a for a in all_candidates if a not in current_set]
                if candidates_to_add:
                    a = random.choice(candidates_to_add)
                    if compatible_with(a, current_set):
                        new_set = set(current_set)
                        new_set.add(a)
                        delta = weights[a]
                        new_score = current_score + delta
                        if new_score > current_score or (delta < 0 and random.random() < math.exp(delta / max(0.1, (1 - rs / ls_restarts) * 0.5))):
                            current_set = new_set
                            current_score = new_score
                            improved = True
            elif move_type < 0.65:
                # Remove
                if current_set:
                    a = random.choice(list(current_set))
                    new_set = set(current_set)
                    new_set.remove(a)
                    new_score = current_score - weights[a]
                    # Always accept improvement, else accept with prob (for escape)
                    if random.random() < math.exp(-(weights[a]) / max(0.1, (1 - rs / ls_restarts) * 0.3)):
                        current_set = new_set
                        current_score = new_score
                        improved = True
            else:
                # Swap
                if current_set:
                    a_rm = random.choice(list(current_set))
                    candidates_to_add = [a for a in all_candidates if a not in current_set]
                    if candidates_to_add:
                        a_add = random.choice(candidates_to_add)
                        temp_set = set(current_set)
                        temp_set.remove(a_rm)
                        if compatible_with(a_add, temp_set):
                            new_set = temp_set
                            new_set.add(a_add)
                            delta = weights[a_add] - weights[a_rm]
                            new_score = current_score + delta
                            if new_score > current_score or (delta < 0 and random.random() < math.exp(delta / max(0.1, (1 - rs / ls_restarts) * 0.5))):
                                current_set = new_set
                                current_score = new_score
                                improved = True
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
    
    # ---- Phase 3: Iterated greedy with random restarts ----
    ig_count = 0
    while time.time() < deadline - 5:
        ig_count += 1
        # Randomized greedy: at each step, among all compatible candidates,
        # pick one from top-k by weight with some randomness
        s = set()
        remaining = list(range(1, N))
        random.shuffle(remaining)
        remaining.sort(key=lambda x: weights[x], reverse=True)
        
        while remaining:
            best_pick = None
            best_w = -1
            # Consider top candidates (limit search for speed)
            for a in remaining[:50]:
                if is_coprime_with_all(a, s):
                    if weights[a] > best_w:
                        best_w = weights[a]
                        best_pick = a
            if best_pick is None:
                break
            s.add(best_pick)
            remaining = [x for x in remaining if x != best_pick]
        
        sc = sum(weights[a] for a in s)
        if sc > best_score:
            best_score = sc
            best_set = s
        
        # Also try: start from current best_set, randomly remove one, then greedily add
        if len(best_set) > 3:
            s2 = set(best_set)
            a_rm = random.choice(list(s2))
            s2.remove(a_rm)
            # Greedily add back compatible high-weight elements
            for a in sorted_by_weight:
                if a not in s2 and is_coprime_with_all(a, s2):
                    s2.add(a)
            sc2 = sum(weights[a] for a in s2)
            if sc2 > best_score:
                best_score = sc2
                best_set = s2
    
    # ---- Phase 4: Final polish - try each single element addition/removal ----
    # Try adding each non-selected candidate if compatible
    for a in range(1, N):
        if a not in best_set and is_coprime_with_all(a, best_set):
            test_set = set(best_set)
            test_set.add(a)
            sc = sum(weights[x] for x in test_set)
            if sc > best_score:
                best_score = sc
                best_set = test_set
    
    # Try removing low-weight elements that conflict with nothing (free) but
    # check if removing one allows adding two better ones
    for a in sorted(best_set, key=lambda x: weights[x]):
        if time.time() > deadline - 3:
            break
        test_set = set(best_set)
        test_set.remove(a)
        # Try to add the best compatible element
        best_add = None
        best_add_w = weights[a]  # must exceed what we removed
        for b in sorted_by_weight:
            if b not in test_set and is_coprime_with_all(b, test_set):
                if weights[b] > best_add_w:
                    best_add = b
                    best_add_w = weights[b]
                else:
                    break
        if best_add is not None:
            test_set.add(best_add)
            sc = sum(weights[x] for x in test_set)
            if sc > best_score:
                best_score = sc
                best_set = test_set
    
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(f"Selected: {res}")
    score = sum(1.0 / (512 - a) for a in res)
    print(f"Score: {score}")
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if math.gcd(res[i], res[j]) != 1:
                print(f"ERROR: {res[i]} and {res[j]} not coprime")
    print("Done")
