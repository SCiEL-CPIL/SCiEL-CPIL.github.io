# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Optimized search for a set of up to 48 distinct integers in [1, 192]
    such that no element divides the sum of any 3 distinct OTHER elements.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    def time_left():
        return deadline - time.time()

    N = 192
    ALL = list(range(1, N + 1))

    # Precompute multiples for fast divisibility check
    max_sum3 = 3 * N
    multiples_of = {}
    for x in range(1, N + 1):
        multiples_of[x] = set(range(x, max_sum3 + 1, x))

    def is_valid_full(s):
        """
        Full validity check for set s.
        """
        lst = sorted(s)
        n = len(lst)
        if n < 4:
            return True
        
        for idx_a in range(n):
            a = lst[idx_a]
            others = [lst[i] for i in range(n) if i != idx_a]
            m = len(others)
            if m < 3:
                continue
            for i in range(m):
                for j in range(i + 1, m):
                    s2 = others[i] + others[j]
                    for k in range(j + 1, m):
                        if (s2 + others[k]) % a == 0:
                            return False
        return True

    def recompute_state(s):
        """
        Recompute sums3 and forbidden from scratch for set s.
        """
        lst = sorted(s)
        n = len(lst)
        sums3 = set()
        if n >= 3:
            for i in range(n):
                for j in range(i+1, n):
                    s2 = lst[i] + lst[j]
                    for k in range(j+1, n):
                        sums3.add(s2 + lst[k])
        
        forbidden = {}
        for idx_a in range(n):
            a = lst[idx_a]
            forb = set()
            for i in range(n):
                if i == idx_a: continue
                for j in range(i+1, n):
                    if j == idx_a: continue
                    forb.add(-(lst[i]+lst[j]) % a)
            forbidden[a] = forb
        return sums3, forbidden

    def can_add_fast(x, s, sums3, forbidden):
        """
        Fast check if x can be added to s.
        """
        # Check 1: Is any 3-sum from s divisible by x?
        for m in multiples_of[x]:
            if m in sums3:
                return False
        
        # Check 2: For each a in s, is x % a in forbidden[a]?
        for a in s:
            if x % a in forbidden[a]:
                return False
        
        return True

    def update_state_add(s, x, sums3, forbidden):
        """
        Update sums3 and forbidden after adding x to s.
        s is the set BEFORE adding x.
        """
        lst = sorted(s)
        n = len(lst)
        
        # Update sums3
        new_sums = set(sums3)
        for i in range(n):
            for j in range(i + 1, n):
                new_sums.add(x + lst[i] + lst[j])
        
        # Update forbidden
        new_forbidden = {a: set(f) for a, f in forbidden.items()}
        
        # For each existing a, add new forbidden residues caused by x
        for a in s:
            forb_a = new_forbidden[a]
            for b in s:
                if b == a:
                    continue
                forb_a.add(-(x + b) % a)
            
        # Add x to forbidden
        forb_x = set()
        for i in range(n):
            for j in range(i + 1, n):
                forb_x.add(-(lst[i] + lst[j]) % x)
        new_forbidden[x] = forb_x
        
        return new_sums, new_forbidden

    def greedy_build(order):
        selected = set()
        sums3 = set()
        forbidden = {}
        for x in order:
            if x in selected:
                continue
            if can_add_fast(x, selected, sums3, forbidden):
                selected.add(x)
                sums3, forbidden = update_state_add(selected - {x}, x, sums3, forbidden)
        return selected, sums3, forbidden

    def local_search(s, sums3, forbidden, rng, max_iters=100):
        s = set(s)
        sums3 = set(sums3)
        forbidden = {a: set(f) for a, f in forbidden.items()}
        
        for _ in range(max_iters):
            if time_left() < 2:
                break
            
            improved = False
            
            # Try additions
            candidates = [x for x in ALL if x not in s]
            rng.shuffle(candidates)
            for x in candidates[:80]:
                if can_add_fast(x, s, sums3, forbidden):
                    s.add(x)
                    sums3, forbidden = update_state_add(s - {x}, x, sums3, forbidden)
                    improved = True
                    break
            
            if improved:
                continue

            # Try swaps (remove 1, add 1)
            elems = list(s)
            rng.shuffle(elems)
            for rem in elems[:20]:
                candidates = [x for x in ALL if x not in s]
                rng.shuffle(candidates)
                for add in candidates[:80]:
                    new_s = (s - {rem}) | {add}
                    if is_valid_full(new_s):
                        s = new_s
                        sums3, forbidden = recompute_state(s)
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue

            # Try remove 1, add 2
            if time_left() < 5:
                break
            elems = list(s)
            rng.shuffle(elems)
            for rem in elems[:15]:
                candidates = [x for x in ALL if x not in s]
                rng.shuffle(candidates)
                # Use a larger pool for the second element to escape local optima
                for add1 in candidates[:30]:
                    for add2 in candidates[:30]:
                        if add1 == add2:
                            continue
                        new_s = (s - {rem}) | {add1, add2}
                        if is_valid_full(new_s):
                            s = new_s
                            sums3, forbidden = recompute_state(s)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return s, sums3, forbidden

    # Phase 1: Greedy starts
    rng = random.Random(42)
    best_s = set()
    best_size = 0
    best_sums3 = set()
    best_forbidden = {}

    orders = [
        list(range(N, 0, -1)),
        list(range(1, N + 1)),
    ]
    for _ in range(5):
        o = ALL[:]
        rng.shuffle(o)
        orders.append(o)
    
    for o in orders:
        s, sums3, forbidden = greedy_build(o)
        if len(s) > best_size:
            best_size = len(s)
            best_s = s
            best_sums3 = sums3
            best_forbidden = forbidden

    # Phase 2: Local search from best
    # We run multiple iterations to try to escape local optima
    # The "remove 1, add 2" move is key for finding larger sets
    for _ in range(30):
        if time_left() < 10:
            break
        s, sums3, forbidden = local_search(best_s, best_sums3, best_forbidden, rng, max_iters=80)
        if len(s) > best_size:
            best_size = len(s)
            best_s = s
            best_sums3 = sums3
            best_forbidden = forbidden
        else:
            # Perturb
            s = set(best_s)
            to_remove = rng.sample(list(s), min(3, len(s)))
            s -= set(to_remove)
            sums3, forbidden = recompute_state(s)
            
            s, sums3, forbidden = local_search(s, sums3, forbidden, rng, max_iters=80)
            if len(s) > best_size:
                best_size = len(s)
                best_s = s
                best_sums3 = sums3
                best_forbidden = forbidden

    # Final check
    if not is_valid_full(best_s):
        # Fallback
        best_s, _, _ = greedy_build(list(range(N, 0, -1)))
        while not is_valid_full(best_s):
            lst = sorted(best_s)
            best_s = set(lst[:-1])

    return list(best_s)
# EVOLVE-BLOCK-END
