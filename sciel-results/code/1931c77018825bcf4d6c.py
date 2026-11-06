"""Two Sidon sets with disjoint difference sets.
Optimized for speed: Uses randomized local search with incremental bitmask checks
for sums and efficient rebuilds for differences to stay within time limits."""

import time
import random
import math

N = 256
MAX_SIZE = 64
SUM_BITS = 512  # sums 2..512 inclusive -> 511 bits
DIFF_BITS = 512  # diffs -255..255 inclusive -> 511 bits


def _sum_bit(x):
    return 1 << (2 * x - 2)

def _sum_pair_bit(a, b):
    return 1 << (a + b - 2)

def _diff_bit(d):
    return 1 << (d + 255)

def _check_sidon(els):
    n = len(els)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _nonzero_diffs(els):
    d = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                d.add(els[j] - els[i])
    return d

def _valid(A, B):
    if not _check_sidon(A):
        return False
    if not _check_sidon(B):
        return False
    if _nonzero_diffs(A) & _nonzero_diffs(B):
        return False
    return True


class State:
    __slots__ = ("A", "B", "sums_A", "sums_B", "diffs_A", "diffs_B", "set_A", "set_B")

    def __init__(self, A=None, B=None):
        self.A = list(A) if A else []
        self.B = list(B) if B else []
        self.set_A = set(self.A)
        self.set_B = set(self.B)
        self.sums_A = 0
        self.sums_B = 0
        self.diffs_A = 0
        self.diffs_B = 0
        self._rebuild()

    def _rebuild(self):
        self.sums_A = 0
        for i in range(len(self.A)):
            ai = self.A[i]
            for j in range(i, len(self.A)):
                self.sums_A |= _sum_pair_bit(ai, self.A[j])
        
        self.sums_B = 0
        for i in range(len(self.B)):
            bi = self.B[i]
            for j in range(i, len(self.B)):
                self.sums_B |= _sum_pair_bit(bi, self.B[j])
                
        self.diffs_A = 0
        for i in range(len(self.A)):
            ai = self.A[i]
            for j in range(len(self.A)):
                if i != j:
                    self.diffs_A |= _diff_bit(self.A[j] - ai)
                    
        self.diffs_B = 0
        for i in range(len(self.B)):
            bi = self.B[i]
            for j in range(len(self.B)):
                if i != j:
                    self.diffs_B |= _diff_bit(self.B[j] - bi)

    def copy(self):
        s = State.__new__(State)
        s.A = list(self.A)
        s.B = list(self.B)
        s.set_A = set(self.A)
        s.set_B = set(self.B)
        s.sums_A = self.sums_A
        s.sums_B = self.sums_B
        s.diffs_A = self.diffs_A
        s.diffs_B = self.diffs_B
        return s


def _valid_add_A(st, x):
    if x in st.set_A:
        return False
    # Sidon check
    if st.sums_A & _sum_bit(x):
        return False
    for a in st.A:
        if st.sums_A & _sum_pair_bit(a, x):
            return False
    # Diff disjointness check
    for a in st.A:
        d1 = x - a
        d2 = a - x
        if d1 != 0 and (st.diffs_B & _diff_bit(d1)):
            return False
        if d2 != 0 and (st.diffs_B & _diff_bit(d2)):
            return False
    return True

def _valid_add_B(st, x):
    if x in st.set_B:
        return False
    if st.sums_B & _sum_bit(x):
        return False
    for b in st.B:
        if st.sums_B & _sum_pair_bit(b, x):
            return False
    for b in st.B:
        d1 = x - b
        d2 = b - x
        if d1 != 0 and (st.diffs_A & _diff_bit(d1)):
            return False
        if d2 != 0 and (st.diffs_A & _diff_bit(d2)):
            return False
    return True

def _commit_add_A(st, x):
    st.A.append(x)
    st.set_A.add(x)
    st.sums_A |= _sum_bit(x)
    for a in st.A[:-1]:
        st.sums_A |= _sum_pair_bit(a, x)
    for a in st.A[:-1]:
        st.diffs_A |= _diff_bit(x - a)
        st.diffs_A |= _diff_bit(a - x)

def _commit_add_B(st, x):
    st.B.append(x)
    st.set_B.add(x)
    st.sums_B |= _sum_bit(x)
    for b in st.B[:-1]:
        st.sums_B |= _sum_pair_bit(b, x)
    for b in st.B[:-1]:
        st.diffs_B |= _diff_bit(x - b)
        st.diffs_B |= _diff_bit(b - x)

def _valid_swap_A(st, idx, x):
    """Check if swapping A[idx] with x is valid."""
    old = st.A[idx]
    if x in st.set_A and x != old:
        return False
    
    # Temporarily remove old from sums/diffs? 
    # Easier: Create a temporary list and check validity of the new set A'
    # Since N is small, we can just check the new set A' against Sidon and B's diffs.
    new_A = st.A[:idx] + st.A[idx+1:] + [x]
    
    # Check Sidon for new_A
    sums = 0
    valid_sidon = True
    for i in range(len(new_A)):
        ai = new_A[i]
        if sums & _sum_bit(ai):
            valid_sidon = False
            break
        sums |= _sum_bit(ai)
        for j in range(i+1, len(new_A)):
            bit = _sum_pair_bit(ai, new_A[j])
            if sums & bit:
                valid_sidon = False
                break
            sums |= bit
        if not valid_sidon:
            break
    
    if not valid_sidon:
        return False
        
    # Check diff disjointness with B
    # We need diffs of new_A.
    # Instead of building full mask, just check each diff of new_A against diffs_B
    for i in range(len(new_A)):
        ai = new_A[i]
        for j in range(len(new_A)):
            if i != j:
                d = new_A[j] - ai
                if st.diffs_B & _diff_bit(d):
                    return False
    return True

def _valid_swap_B(st, idx, x):
    old = st.B[idx]
    if x in st.set_B and x != old:
        return False
    
    new_B = st.B[:idx] + st.B[idx+1:] + [x]
    
    sums = 0
    valid_sidon = True
    for i in range(len(new_B)):
        bi = new_B[i]
        if sums & _sum_bit(bi):
            valid_sidon = False
            break
        sums |= _sum_bit(bi)
        for j in range(i+1, len(new_B)):
            bit = _sum_pair_bit(bi, new_B[j])
            if sums & bit:
                valid_sidon = False
                break
            sums |= bit
        if not valid_sidon:
            break
            
    if not valid_sidon:
        return False
        
    for i in range(len(new_B)):
        bi = new_B[i]
        for j in range(len(new_B)):
            if i != j:
                d = new_B[j] - bi
                if st.diffs_A & _diff_bit(d):
                    return False
    return True

def _do_swap_A(st, idx, x):
    st.A[idx] = x
    st.set_A.add(x)
    # Note: set_A might contain old value, but we don't strictly need set_A to be perfect 
    # if we check `x in st.set_A` carefully or just use list membership for small N.
    # Actually, set_A should be updated.
    # Let's just rebuild set_A and set_B on swap to be safe, it's O(n).
    st.set_A = set(st.A)
    st._rebuild()

def _do_swap_B(st, idx, x):
    st.B[idx] = x
    st.set_B = set(st.B)
    st._rebuild()

def _greedy_grow(st, deadline, rng):
    candidates_A = list(range(1, N + 1))
    candidates_B = list(range(1, N + 1))
    rng.shuffle(candidates_A)
    rng.shuffle(candidates_B)

    for _ in range(500):
        if time.time() > deadline - 3:
            break
        progress = False

        if len(st.A) < MAX_SIZE:
            for x in candidates_A:
                if x in st.set_A:
                    continue
                if _valid_add_A(st, x):
                    _commit_add_A(st, x)
                    candidates_A.remove(x)
                    progress = True
                    break

        if len(st.B) < MAX_SIZE:
            for x in candidates_B:
                if x in st.set_B:
                    continue
                if _valid_add_B(st, x):
                    _commit_add_B(st, x)
                    candidates_B.remove(x)
                    progress = True
                    break

        if not progress:
            break

    return st

def _local_search(st, deadline, rng):
    """Randomized local search to escape local optima."""
    while time.time() < deadline - 2:
        action = rng.choice(['add_A', 'add_B', 'swap_A', 'swap_B'])
        
        if action == 'add_A' and len(st.A) < MAX_SIZE:
            x = rng.randint(1, N)
            if x not in st.set_A and _valid_add_A(st, x):
                _commit_add_A(st, x)
        elif action == 'add_B' and len(st.B) < MAX_SIZE:
            x = rng.randint(1, N)
            if x not in st.set_B and _valid_add_B(st, x):
                _commit_add_B(st, x)
        elif action == 'swap_A' and len(st.A) > 0:
            idx = rng.randint(0, len(st.A) - 1)
            x = rng.randint(1, N)
            if x not in st.set_A and _valid_swap_A(st, idx, x):
                _do_swap_A(st, idx, x)
        elif action == 'swap_B' and len(st.B) > 0:
            idx = rng.randint(0, len(st.B) - 1)
            x = rng.randint(1, N)
            if x not in st.set_B and _valid_swap_B(st, idx, x):
                _do_swap_B(st, idx, x)

    return st

def _structured_seed(rng):
    def is_prime(x):
        if x < 2: return False
        if x == 2: return True
        if x % 2 == 0: return False
        for i in range(3, int(math.sqrt(x)) + 1, 2):
            if x % i == 0: return False
        return True

    primes = [p for p in range(13, 60) if is_prime(p)]
    rng.shuffle(primes)

    for p in primes:
        qr = sorted(set((i * i) % p for i in range(p)))
        nqr = sorted(set(range(p)) - set(qr))
        scale = N // p
        A = []
        for r in qr:
            val = r * scale + 1
            if val <= N and val not in A:
                A.append(val)
        B = []
        for r in nqr:
            val = r * scale + 1
            if val <= N and val not in B:
                B.append(val)
        if A and B and _valid(A, B):
            return A, B

    A = [rng.randint(1, N)]
    B = [rng.randint(1, N)]
    return A, B

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    rng = random.Random(42)

    best_score = 0
    best_A = [1]
    best_B = [2]

    # Initial structured seed
    A0, B0 = _structured_seed(rng)
    if not A0: A0 = [1]
    if not B0: B0 = [2]

    st = State(A0, B0)
    _greedy_grow(st, deadline, rng)
    _local_search(st, deadline, rng)
    score = len(st.A) + len(st.B)
    if score > best_score and _valid(st.A, st.B):
        best_score = score
        best_A = list(st.A)
        best_B = list(st.B)

    # Multiple restarts
    for it in range(50): # Reduced restarts to allow longer search per restart
        if time.time() > deadline - 10:
            break

        strat = it % 4
        if strat == 0:
            A0 = [rng.randint(1, N)]
            B0 = [rng.randint(1, N)]
        elif strat == 1:
            sa = rng.randint(2, 4)
            sb = rng.randint(2, 4)
            A0 = sorted(rng.sample(range(1, N + 1), sa))
            B0 = sorted(rng.sample(range(1, N + 1), sb))
        elif strat == 2:
            A0 = [rng.randint(1, 128)]
            B0 = [rng.randint(129, N)]
        else:
            A0, B0 = _structured_seed(rng)

        st = State(A0, B0)
        _greedy_grow(st, deadline, rng)
        _local_search(st, deadline, rng)
        score = len(st.A) + len(st.B)
        if score > best_score and _valid(st.A, st.B):
            best_score = score
            best_A = list(st.A)
            best_B = list(st.B)

    # Final polish
    if time.time() < deadline - 5:
        st = State(best_A, best_B)
        _local_search(st, deadline, rng)
        score = len(st.A) + len(st.B)
        if score > best_score and _valid(st.A, st.B):
            best_A = list(st.A)
            best_B = list(st.B)
            best_score = score

    if not _valid(best_A, best_B):
        best_A = [1]
        best_B = [2]

    return [best_A, best_B]
