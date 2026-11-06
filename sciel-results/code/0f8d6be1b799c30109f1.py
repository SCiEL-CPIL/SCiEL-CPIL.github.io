"""Equal-gcd-triple-free subsets of [1,128]: structural base + multi-order greedy
+ fast ball-based add checks + bounded 1-opt/2-opt local search.

Constraint: no three distinct a,b,c in S with gcd(a,b)==gcd(a,c)==gcd(b,c).
Adding x to a valid S is forbidden iff exists g and pair (a,b) in S with
gcd(x,a)=gcd(x,b)==gcd(a,b)==g.  We precompute, per (x,g), the list of y with
gcd(x,y)=g, so the check is: for some g, |S ∩ B_g(x)| >= 2 AND some pair in
that intersection has gcd g.  This is cheap and frees budget for restarts.
"""
import time
import random
import math
from math import gcd
from itertools import combinations

_DEADLINE = 60.0          # wall-clock safety margin under the 90 s kill
_START = None
_CAND = list(range(1, 129))
_PRE = None               # pairwise gcd matrix, indexed 0..128
_BALL = None              # BALL[x][g] = list of y in [1,128] with gcd(x,y)==g
_PRIMEP = None            # prime-power flag per value
_GCD1 = None              # GCD1[x] = list of y in [1,128] with gcd(x,y)==1
_MASKS = None             # per-gcd ball masks for fast intersection checks
_PAIRMASK = None          # PAIRMASK[x][g] = bitmask of y with gcd(x,y)==g AND y!=x, used for conflict counting


def _ensure_pre():
    global _PRE, _BALL, _PRIMEP, _GCD1, _MASKS, _PAIRMASK
    if _PRE is not None:
        return
    n = 128
    PRE = [[1] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            PRE[i][j] = PRE[j][i] = gcd(i, j)
    _PRE = PRE

    # Balls: for each x and each gcd value g, list of y with gcd(x,y)==g.
    BALL = {}
    for x in range(1, n + 1):
        d = {}
        for y in range(1, n + 1):
            g = PRE[x][y]
            d.setdefault(g, []).append(y)
        BALL[x] = d
    _BALL = BALL
    _GCD1 = {x: BALL[x].get(1, []) for x in range(1, 129)}

    # Bitmasks: MASKS[x][g] = bitmask of y with gcd(x,y)==g
    MASKS = {}
    for x in range(1, n + 1):
        d = {}
        for y in range(1, n + 1):
            g = PRE[x][y]
            d[g] = d.get(g, 0) | (1 << y)
        MASKS[x] = d
    _MASKS = MASKS

    # PAIRMASK[x][g] = bitmask of y (y != x) with gcd(x,y)==g
    # Used for O(1) conflict counting in simulated annealing
    PAIRMASK = {}
    for x in range(1, n + 1):
        d = {}
        for y in range(1, n + 1):
            if y == x:
                continue
            g = PRE[x][y]
            d[g] = d.get(g, 0) | (1 << y)
        PAIRMASK[x] = d
    _PAIRMASK = PAIRMASK

    # prime-power flags
    flag = [False] * (n + 1)
    for x in range(2, n + 1):
        v = x
        f0 = None
        ok = True
        p = 2
        while p * p <= v:
            if v % p == 0:
                if f0 is None:
                    f0 = p
                else:
                    ok = False
                    break
                while v % p == 0:
                    v //= p
            p += 1
        if ok and (v == 1 or v == f0):
            flag[x] = True
    _PRIMEP = flag


def _num_divisors(n):
    c = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            c += 1
            if i * i != n:
                c += 1
        i += 1
    return c


def _bad_triples_in(S):
    """Count equal-gcd triples within S (each triple counted once)."""
    P = _PRE
    s = list(S)
    n = len(s)
    cnt = 0
    for i in range(n):
        a = s[i]
        for j in range(i + 1, n):
            b = s[j]
            g_ab = P[a][b]
            for k in range(j + 1, n):
                c = s[k]
                if P[a][c] == g_ab and P[b][c] == g_ab:
                    cnt += 1
    return cnt


def _can_add(Sset, x):
    """True if adding x to valid set Sset creates no equal-gcd triple."""
    P = _PRE
    B = _BALL[x]
    for g, lst in B.items():
        # gather S elements in this ball (excluding x itself, which is not in S)
        inball = [u for u in Sset if u in set(lst)]
        if len(inball) < 2:
            continue
        # need a pair in inball whose gcd is g
        m = len(inball)
        for i in range(m):
            a = inball[i]
            for j in range(i + 1, m):
                b = inball[j]
                if P[a][b] == g:
                    return False
    return True


def _can_add_list(Slist, x):
    """List-based fast check (Slist is a plain list, no set membership)."""
    P = _PRE
    B = _BALL[x]
    for g, lst in B.items():
        if len(lst) < 2:
            continue
        inball = []
        for u in Slist:
            if u == x:
                continue
            if P[x][u] == g:
                inball.append(u)
        if len(inball) < 2:
            continue
        m = len(inball)
        for i in range(m):
            a = inball[i]
            for j in range(i + 1, m):
                b = inball[j]
                if P[a][b] == g:
                    return False
    return True


def _can_add_mask(Smask, x):
    """Bitmask-based fast check. Smask has bit y set iff y in S."""
    P = _PRE
    MK = _MASKS[x]
    for g, mask in MK.items():
        # Count elements of S in this ball (excluding x)
        inter = Smask & mask
        if inter & (1 << x):
            inter ^= (1 << x)
        # Need at least 2 elements
        if inter & (inter - 1) == 0:
            continue
        # Check if any pair in inter has gcd g
        # Iterate over set bits
        m = inter
        while m:
            a_bit = m & (-m)
            a = a_bit.bit_length() - 1
            m ^= a_bit
            rest = m
            while rest:
                b_bit = rest & (-rest)
                b = b_bit.bit_length() - 1
                if P[a][b] == g:
                    return False
                rest ^= b_bit
    return True


def _count_conflicts_with_x(Smask, x):
    """Count number of equal-gcd triples that x participates in, given S (x not in S).
    A triple (x, a, b) is bad iff gcd(x,a)==gcd(x,b)==gcd(a,b)==g for some g.
    For each g, let B = S ∩ {y : gcd(x,y)==g}. The number of bad pairs in B
    is the number of pairs (a,b) in B with gcd(a,b)==g.
    """
    P = _PRE
    MK = _MASKS[x]
    total = 0
    for g, mask in MK.items():
        inter = Smask & mask
        # inter has bits for elements y in S with gcd(x,y)==g
        if inter & (inter - 1) == 0:
            continue
        # Count pairs (a,b) in inter with gcd(a,b)==g
        m = inter
        while m:
            a_bit = m & (-m)
            a = a_bit.bit_length() - 1
            m ^= a_bit
            rest = m
            while rest:
                b_bit = rest & (-rest)
                b = b_bit.bit_length() - 1
                if P[a][b] == g:
                    total += 1
                rest ^= b_bit
    return total


def _swap_valid(Slist, r, x):
    """Check Slist - {r} + {x} is valid. Original set is valid, so only new
    triples involve x; check pairs (a,b) in Slist\\{r}."""
    P = _PRE
    B = _BALL[x]
    for g, lst in B.items():
        if len(lst) < 2:
            continue
        inball = []
        for u in Slist:
            if u == r or u == x:
                continue
            if P[x][u] == g:
                inball.append(u)
        if len(inball) < 2:
            continue
        m = len(inball)
        for i in range(m):
            a = inball[i]
            for j in range(i + 1, m):
                b = inball[j]
                if P[a][b] == g:
                    return False
    return True


def _find_bad_triple(S):
    """Find one bad triple (a,b,c) in S, or None if S is valid."""
    P = _PRE
    s = list(S)
    n = len(s)
    for i in range(n):
        a = s[i]
        for j in range(i + 1, n):
            b = s[j]
            g_ab = P[a][b]
            for k in range(j + 1, n):
                c = s[k]
                if P[a][c] == g_ab and P[b][c] == g_ab:
                    return (a, b, c)
    return None


def _two_opt_remove(S):
    """Remove two elements from a bad triple to restore validity.
    Returns (new_S, removed_pair) or (None, None) if no bad triple found."""
    triple = _find_bad_triple(S)
    if triple is None:
        return S, None
    a, b, c = triple
    # Try removing each pair from the triple, check if result is valid
    # and allows adding back some excluded element
    Sset = set(S)
    candidates = [(a, b), (a, c), (b, c)]
    best_new = None
    best_gain = -1
    for r1, r2 in candidates:
        new_S = [x for x in S if x != r1 and x != r2]
        new_Sset = set(new_S)
        # Check validity (should be valid since we removed a bad triple's elements)
        # Count how many excluded elements can now be added
        gain = 0
        for x in _CAND:
            if x not in new_Sset and _can_add_list(new_S, x):
                gain += 1
        if gain > best_gain:
            best_gain = gain
            best_new = new_S
    if best_new is not None:
        return best_new, (candidates[0][0], candidates[0][1])  # placeholder
    return S, None


def _greedy(order):
    S = []
    for x in order:
        if _can_add_list(S, x):
            S.append(x)
    return S


def _local_add(S):
    """Repeatedly add any excluded element that fits until none remain."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        for x in _CAND:
            if x not in S and _can_add_list(S, x):
                S.append(x)
                improved = True
    return S


def _local_search(S):
    """Bounded local search: adds, then targeted swaps to enable more adds."""
    P = _PRE
    S = list(S)
    Sset = set(S)
    max_rounds = 50
    for _ in range(max_rounds):
        if time.time() - _START > _DEADLINE - 6:
            break
        # (a) Try all adds until no more fit
        added = True
        while added:
            added = False
            for x in _CAND:
                if x not in Sset and _can_add_list(S, x):
                    S.append(x)
                    Sset.add(x)
                    added = True
        if time.time() - _START > _DEADLINE - 6:
            break
        # (b) 1-opt swap: find a swap that enables at least one new add
        swap_found = False
        for r in list(S):
            if time.time() - _START > _DEADLINE - 6:
                break
            for x in _CAND:
                if x in Sset or x == r:
                    continue
                if _swap_valid(S, r, x):
                    # Tentatively swap and check if it enables any add
                    S.remove(r)
                    Sset.discard(r)
                    S.append(x)
                    Sset.add(x)
                    # Check if any new element can be added
                    can_add_new = False
                    for y in _CAND:
                        if y not in Sset and _can_add_list(S, y):
                            can_add_new = True
                            break
                    if can_add_new:
                        swap_found = True
                        break
                    else:
                        # Revert
                        S.remove(x)
                        Sset.discard(x)
                        S.append(r)
                        Sset.add(r)
            if swap_found:
                break
        if not swap_found:
            break
    return S


def _simulated_annealing(S_init, max_time=25.0):
    """Simulated annealing to find large valid sets.
    State: bitmask of selected elements.
    Moves: add/remove/swap elements.
    Objective: maximize |S| - penalty * (# bad triples).
    Uses incremental conflict counting for speed.
    """
    P = _PRE
    MK = _MASKS
    rng = random.Random(42)
    
    # Build initial state
    Smask = 0
    for x in S_init:
        Smask |= (1 << x)
    
    # Count initial conflicts
    def _count_all_conflicts(mask):
        """Count total bad triples in the set represented by mask."""
        total = 0
        # For each element in mask, count conflicts it participates in, divide by 3
        m = mask
        while m:
            bit = m & (-m)
            x = bit.bit_length() - 1
            m ^= bit
            # Count conflicts x has with other elements in mask
            rest = mask & ~bit
            if rest & (rest - 1) == 0:
                continue
            for g, gmask in MK[x].items():
                inter = rest & gmask
                if inter & (inter - 1) == 0:
                    continue
                im = inter
                while im:
                    a_bit = im & (-im)
                    a = a_bit.bit_length() - 1
                    im ^= a_bit
                    rem = im
                    while rem:
                        b_bit = rem & (-rem)
                        b = b_bit.bit_length() - 1
                        if P[a][b] == g:
                            total += 1
                        rem ^= b_bit
        return total // 3  # each triple counted 3 times (once per element)
    
    # Faster: count conflicts per element, sum, divide by 3
    def _conflict_count(mask):
        """Total number of bad triples (each counted once)."""
        total = 0
        m = mask
        while m:
            bit = m & (-m)
            x = bit.bit_length() - 1
            m ^= bit
            rest = mask & ~bit
            if rest & (rest - 1) == 0:
                continue
            for g, gmask in MK[x].items():
                inter = rest & gmask
                if inter & (inter - 1) == 0:
                    continue
                im = inter
                while im:
                    a_bit = im & (-im)
                    a = a_bit.bit_length() - 1
                    im ^= a_bit
                    rem = im
                    while rem:
                        b_bit = rem & (-rem)
                        b = b_bit.bit_length() - 1
                        if P[a][b] == g:
                            total += 1
                        rem ^= b_bit
        return total // 3
    
    def _popcount(m):
        c = 0
        while m:
            m &= m - 1
            c += 1
        return c
    
    current_mask = Smask
    current_conflicts = _conflict_count(current_mask)
    current_size = _popcount(current_mask)
    
    best_mask = current_mask
    best_size = current_size if current_conflicts == 0 else -1
    best_conflicts = current_conflicts
    
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        if time.time() - _START > _DEADLINE - 8:
            break
        if T < T_min:
            break
        
        # Generate a random move
        move_type = rng.random()
        if move_type < 0.3:
            # Add a random element not in set
            not_in = ~current_mask & ((1 << 129) - 2)  # bits 1..128
            if not_in == 0:
                T *= cooling
                continue
            # Pick random bit from not_in
            m = not_in
            # Use a random offset to pick a bit
            r = rng.randint(1, 128)
            # Find a set bit: use a simple approach
            # Actually, let's just iterate
            candidates = []
            mm = not_in
            while mm:
                bit = mm & (-mm)
                candidates.append(bit)
                mm ^= bit
            if not candidates:
                T *= cooling
                continue
            new_bit = candidates[rng.randint(0, len(candidates) - 1)]
            x = new_bit.bit_length() - 1
            new_mask = current_mask | new_bit
            
            # Compute delta conflicts
            # New conflicts: triples (x, a, b) where a,b in current_mask
            # and gcd(x,a)==gcd(x,b)==gcd(a,b)
            delta_conf = 0
            for g, gmask in MK[x].items():
                inter = current_mask & gmask
                if inter & (inter - 1) == 0:
                    continue
                im = inter
                while im:
                    a_bit = im & (-im)
                    a = a_bit.bit_length() - 1
                    im ^= a_bit
                    rem = im
                    while rem:
                        b_bit = rem & (-rem)
                        b = b_bit.bit_length() - 1
                        if P[a][b] == g:
                            delta_conf += 1
                        rem ^= b_bit
            delta_conf //= 3  # Wait, this isn't right. Let me reconsider.
            
            # Actually, each new triple (x,a,b) is a distinct triple.
            # The conflict count increases by the number of such triples.
            # But _conflict_count divides by 3 because each triple is counted
            # from 3 different elements. When we add x, the new triples are
            # only counted from x's perspective in _conflict_count, so the
            # actual increase in _conflict_count is delta_conf // 3? No...
            # Let me just recompute. For simplicity and correctness,
            # let's use a different approach.
            
            # Recompute conflicts for new state
            new_conflicts = _conflict_count(new_mask)
            new_size = current_size + 1
            delta_obj = (new_size - 100 * new_conflicts) - (current_size - 100 * current_conflicts)
            
        elif move_type < 0.6:
            # Remove a random element from set
            if current_mask == 0:
                T *= cooling
                continue
            candidates = []
            mm = current_mask
            while mm:
                bit = mm & (-mm)
                candidates.append(bit)
                mm ^= bit
            new_bit = candidates[rng.randint(0, len(candidates) - 1)]
            x = new_bit.bit_length() - 1
            new_mask = current_mask & ~new_bit
            new_conflicts = _conflict_count(new_mask)
            new_size = current_size - 1
            delta_obj = (new_size - 100 * new_conflicts) - (current_size - 100 * current_conflicts)
            
        else:
            # Swap: remove one, add another
            if current_mask == 0:
                T *= cooling
                continue
            candidates_in = []
            mm = current_mask
            while mm:
                bit = mm & (-mm)
                candidates_in.append(bit)
                mm ^= bit
            if not candidates_in:
                T *= cooling
                continue
            remove_bit = candidates_in[rng.randint(0, len(candidates_in) - 1)]
            not_in = ~current_mask & ((1 << 129) - 2)
            if not_in == 0:
                T *= cooling
                continue
            candidates_out = []
            mm = not_in
            while mm:
                bit = mm & (-mm)
                candidates_out.append(bit)
                mm ^= bit
            if not candidates_out:
                T *= cooling
                continue
            add_bit = candidates_out[rng.randint(0, len(candidates_out) - 1)]
            new_mask = (current_mask & ~remove_bit) | add_bit
            new_conflicts = _conflict_count(new_mask)
            new_size = current_size
            delta_obj = (new_size - 100 * new_conflicts) - (current_size - 100 * current_conflicts)
        
        # Metropolis criterion
        if delta_obj >= 0:
            accept = True
        else:
            accept = rng.random() < math.exp(delta_obj / T)
        
        if accept:
            current_mask = new_mask
            current_conflicts = new_conflicts
            current_size = new_size
            if new_conflicts == 0 and new_size > best_size:
                best_size = new_size
                best_mask = new_mask
                best_conflicts = 0
        
        T *= cooling
    
    # Return the best valid set found
    if best_conflicts == 0:
        result = []
        m = best_mask
        while m:
            bit = m & (-m)
            result.append(bit.bit_length() - 1)
            m ^= bit
        return result
    else:
        # Return best valid set from initial or fallback
        return list(S_init)


def _two_opt_search(S):
    """2-opt local search: find bad triples, remove pairs, re-add greedily.
    This can escape local optima that 1-opt cannot."""
    P = _PRE
    S = list(S)
    Sset = set(S)
    max_rounds = 30
    for _ in range(max_rounds):
        if time.time() - _START > _DEADLINE - 6:
            break
        # First, try to add any elements
        added = True
        while added:
            added = False
            for x in _CAND:
                if x not in Sset and _can_add_list(S, x):
                    S.append(x)
                    Sset.add(x)
                    added = True
        if time.time() - _START > _DEADLINE - 6:
            break
        # Find a bad triple
        triple = _find_bad_triple(S)
        if triple is None:
            # No bad triples, try 1-opt swaps
            swap_found = False
            for r in list(S):
                if time.time() - _START > _DEADLINE - 6:
                    break
                for x in _CAND:
                    if x in Sset or x == r:
                        continue
                    if _swap_valid(S, r, x):
                        S.remove(r)
                        Sset.discard(r)
                        S.append(x)
                        Sset.add(x)
                        can_add_new = False
                        for y in _CAND:
                            if y not in Sset and _can_add_list(S, y):
                                can_add_new = True
                                break
                        if can_add_new:
                            swap_found = True
                            break
                        else:
                            S.remove(x)
                            Sset.discard(x)
                            S.append(r)
                            Sset.add(r)
                if swap_found:
                    break
            if not swap_found:
                break
            continue
        # We have a bad triple (a,b,c). Try removing each pair.
        a, b, c = triple
        pairs = [(a, b), (a, c), (b, c)]
        best_new_S = None
        best_len = -1
        for r1, r2 in pairs:
            new_S = [x for x in S if x != r1 and x != r2]
            new_Sset = set(new_S)
            # Greedily add back elements
            improved = True
            while improved:
                improved = False
                for x in _CAND:
                    if x not in new_Sset and _can_add_list(new_S, x):
                        new_S.append(x)
                        new_Sset.add(x)
                        improved = True
            if len(new_S) > best_len:
                best_len = len(new_S)
                best_new_S = new_S
        if best_new_S is not None and len(best_new_S) >= len(S) - 1:
            S = best_new_S
            Sset = set(S)
        else:
            break
    return S


def _construct():
    global _START
    _ensure_pre()
    _START = time.time()
    P = _PRE
    PRIMEP = _PRIMEP

    prime_powers = [x for x in range(1, 129) if PRIMEP[x]]
    divs = {x: _num_divisors(x) for x in range(1, 129)}

    best = []
    best_len = -1

    # --- Multi-order greedy (inspiration 1) + local add ---
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        sorted(range(1, 129), key=lambda x: divs[x]),
        sorted(range(1, 129), key=lambda x: divs[x], reverse=True),
        sorted(range(1, 129), key=lambda x: (0 if PRIMEP[x] else 1, x)),
        prime_powers + [x for x in range(1, 129) if not PRIMEP[x]],
        sorted(range(1, 129), key=lambda x: (divs[x], x)),
        sorted(range(1, 129), key=lambda x: (-divs[x], x)),
    ]
    for order in orders:
        if time.time() - _START > _DEADLINE - 8:
            break
        S = _greedy(order)
        S = _local_add(S)
        if len(S) > best_len:
            best = S
            best_len = len(S)

    # --- 2-opt search on the best base ---
    if time.time() - _START < _DEADLINE - 10:
        S = _two_opt_search(best)
        if len(S) > best_len:
            best = S
            best_len = len(S)

    # --- Also try 1-opt local search ---
    if time.time() - _START < _DEADLINE - 10:
        S = _local_search(best)
        if len(S) > best_len:
            best = S
            best_len = len(S)

    # --- Simulated annealing from multiple starting points ---
    rng = random.Random(999)
    sa_budget = max(1.0, _DEADLINE - (time.time() - _START) - 12)
    if sa_budget > 3.0:
        # Run SA from the best set found so far
        S_sa = _simulated_annealing(best, max_time=min(sa_budget * 0.4, 20.0))
        S_sa = _local_add(S_sa)
        if len(S_sa) > best_len:
            best = S_sa
            best_len = len(S_sa)
    
    # --- Random restarts: biased random orders, greedy + local search ---
    rng = random.Random(12345)
    while time.time() - _START < _DEADLINE - 10:
        # Mix of strategies: prime powers first, divisor-based, random
        strategy = rng.randint(0, 2)
        if strategy == 0:
            pp = list(prime_powers)
            rest = [x for x in range(1, 129) if not PRIMEP[x]]
            rng.shuffle(pp)
            rng.shuffle(rest)
            order = pp + rest
        elif strategy == 1:
            order = sorted(range(1, 129), key=lambda x: (divs[x], rng.random()))
        else:
            order = list(range(1, 129))
            rng.shuffle(order)
        S = _greedy(order)
        S = _local_add(S)
        if len(S) > best_len:
            best = S
            best_len = len(S)
        
        # Try SA from this starting point if time permits
        remaining = _DEADLINE - (time.time() - _START)
        if remaining > 15:
            S_sa = _simulated_annealing(S, max_time=min(remaining * 0.25, 10.0))
            S_sa = _local_add(S_sa)
            if len(S_sa) > best_len:
                best = S_sa
                best_len = len(S_sa)
        elif time.time() - _START < _DEADLINE - 12:
            # Use 2-opt for better exploration
            S = _two_opt_search(S)
            if len(S) > best_len:
                best = S
                best_len = len(S)

    # --- Final safety: ensure validity and nonempty ---
    if _bad_triples_in(best) != 0:
        # fall back to guaranteed-valid greedy
        best = _greedy(list(range(1, 129)))
        best = _local_add(best)
    if not best:
        best = [2]
    return best


def construct():
    from itertools import combinations  # noqa: F401
    from math import gcd  # noqa: F401
    try:
        res = _construct()
    except Exception:
        _ensure_pre()
        P = _PRE
        res = list(range(1, 129))
        while _bad_triples_in(res) != 0 and len(res) > 1:
            worst = res[0]
            worst_cnt = -1
            for cand in res:
                cnt = 0
                for a, b in combinations([y for y in res if y != cand], 2):
                    if P[cand][a] == P[cand][b] == P[a][b]:
                        cnt += 1
                if cnt > worst_cnt:
                    worst_cnt = cnt
                    worst = cand
            res.remove(worst)
        if not res:
            res = [2]
    return res
