import time
import random
from itertools import combinations

DEADLINE = 50.0  # safety margin before 60s kill (and 90s wall)

def _valid_add(S, a):
    # S is a set/list of current members (not containing a)
    # check for all distinct b,c in S: a does not divide b*c
    L = list(S)
    n = len(L)
    for i in range(n):
        b = L[i]
        for j in range(i+1, n):
            c = L[j]
            if (b * c) % a == 0:
                return False
    return True

def _is_valid_set(S):
    L = list(S)
    n = len(L)
    for i in range(n):
        a = L[i]
        for j in range(i+1, n):
            b = L[j]
            for k in range(j+1, n):
                c = L[k]
                if (b * c) % a == 0:
                    return False
    return True

def construct():
    t0 = time.time()
    universe = list(range(2, 257))
    
    # Precompute primes
    def is_prime(x):
        if x < 2: return False
        if x < 4: return True
        if x % 2 == 0: return False
        i = 3
        while i * i <= x:
            if x % i == 0: return False
            i += 2
        return True
    primes = [x for x in universe if is_prime(x)]
    
    best = set(primes)
    best_size = len(best)
    
    # Greedy from different orderings
    def greedy(order):
        S = []
        Sset = set()
        for a in order:
            if a in Sset: continue
            if _valid_add(Sset, a):
                S.append(a)
                Sset.add(a)
        return Sset
    
    # Orderings: ascending, descending, primes-first, random
    random.seed(12345)
    orders = [
        universe,
        sorted(universe, reverse=True),
        primes + [x for x in universe if not is_prime(x)],
        sorted(universe, key=lambda x: (-is_prime(x), x)),
        sorted(universe, key=lambda x: (x, )),
    ]
    # Add a few random orders
    for _ in range(3):
        o = universe[:]
        random.shuffle(o)
        orders.append(o)
    
    for o in orders:
        if time.time() - t0 > DEADLINE: break
        S = greedy(o)
        if len(S) > best_size:
            best = S
            best_size = len(S)
    
    # Local search / hill-climb on best
    def try_improve(S):
        # S is a set
        improved = True
        while improved and time.time() - t0 < DEADLINE:
            improved = False
            L = list(S)
            # Try additions
            for a in universe:
                if time.time() - t0 > DEADLINE: break
                if a in S: continue
                if _valid_add(S, a):
                    S.add(a)
                    improved = True
                    break
            if improved: continue
            # Try swaps: remove one, add one
            for rem in L:
                if time.time() - t0 > DEADLINE: break
                S2 = S - {rem}
                for a in universe:
                    if a in S2 or a == rem: continue
                    if _valid_add(S2, a):
                        # net size same (remove 1 add 1) -> no improvement in size
                        # skip since we only care about size
                        pass
        return S
    
    # Simulated annealing for size improvement via swaps that enable later adds
    # Simpler: do hill-climb additions from best, then try removing an element and re-greedy
    S = set(best)
    # add all possible
    changed = True
    while changed and time.time() - t0 < DEADLINE:
        changed = False
        for a in universe:
            if a in S: continue
            if _valid_add(S, a):
                S.add(a)
                changed = True
                break
    if len(S) > best_size:
        best = S
        best_size = len(S)
    
    # Try: for each element in best, remove it, then greedily add as much as possible
    def fill_set(S_base, order):
        S = set(S_base)
        for a in order:
            if a in S: continue
            if _valid_add(S, a):
                S.add(a)
        return S
    
    for rem in list(best):
        if time.time() - t0 > DEADLINE - 5: break
        S2_base = best - {rem}
        # Try multiple orderings for fill
        for o in [universe, sorted(universe, reverse=True)]:
            S2 = fill_set(S2_base, o)
            if len(S2) > best_size:
                best = S2
                best_size = len(S2)
    
    # Try removing 2 elements
    best_list = list(best)
    for i in range(len(best_list)):
        if time.time() - t0 > DEADLINE - 5: break
        for j in range(i+1, len(best_list)):
            if time.time() - t0 > DEADLINE - 5: break
            S2_base = best - {best_list[i], best_list[j]}
            for o in [universe, sorted(universe, reverse=True)]:
                S2 = fill_set(S2_base, o)
                if len(S2) > best_size:
                    best = S2
                    best_size = len(S2)
    
    result = sorted(best)
    # Ensure types
    result = [int(x) for x in result]
    # Final validity check
    if not _is_valid_set(result):
        # fallback to primes
        result = [int(x) for x in primes]
    return result
