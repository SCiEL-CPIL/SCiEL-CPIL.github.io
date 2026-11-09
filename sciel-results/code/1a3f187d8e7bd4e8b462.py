"""Improved program: maximize |A|+|B| for two Sidon sets with disjoint
nonzero difference sets in [1,256]. Uses efficient incremental checks and
a greedy + local-search hybrid with multiple restarts."""

import time
import random

N = 256
MAX_EACH = 64
DEADLINE_SECONDS = 85  # safety margin before 90s kill

def _check_sidon(s):
    """Return True if s is a Sidon set (pair sums including doubles are unique)."""
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def _diffs(s):
    """Return set of nonzero absolute differences for sorted s."""
    d = set()
    n = len(s)
    for i in range(n):
        for j in range(i):
            d.add(s[i] - s[j])
    return d

def _valid_pair(A, B):
    """Check both are Sidon and their nonzero difference sets are disjoint."""
    if not _check_sidon(A) or not _check_sidon(B):
        return False
    dA = _diffs(A)
    dB = _diffs(B)
    return not (dA & dB)

def _score(A, B):
    return len(A) + len(B)

class _SidonState:
    """Maintains state for incremental Sidon set construction."""
    def __init__(self):
        self.elems = []
        self.sums = set()
        self.diffs = set()
    
    def can_add(self, x, other_diffs=None):
        """Check if x can be added to this set while maintaining Sidon
        property and (optionally) avoiding diffs in other_diffs."""
        # Check sums: x+x and x+a for each a in elems
        if x + x in self.sums:
            return False
        for a in self.elems:
            if x + a in self.sums:
                return False
        # Check diffs
        if other_diffs is not None:
            for a in self.elems:
                d = abs(x - a)
                if d in self.diffs or d in other_diffs:
                    return False
        else:
            for a in self.elems:
                d = abs(x - a)
                if d in self.diffs:
                    return False
        return True
    
    def add(self, x, other_diffs=None):
        """Add x to the set, updating sums and diffs."""
        self.elems.append(x)
        self.elems.sort()
        self.sums.add(x + x)
        for a in self.elems[:-1]:
            self.sums.add(x + a)
        if other_diffs is None:
            for a in self.elems[:-1]:
                self.diffs.add(abs(x - a))
        else:
            for a in self.elems[:-1]:
                self.diffs.add(abs(x - a))
    
    def remove(self, x):
        """Remove x from the set, updating sums and diffs."""
        idx = self.elems.index(x)
        self.elems.pop(idx)
        # Recompute sums and diffs from scratch (simpler and safer)
        self.sums = set()
        self.diffs = set()
        n = len(self.elems)
        for i in range(n):
            for j in range(i, n):
                self.sums.add(self.elems[i] + self.elems[j])
        for i in range(n):
            for j in range(i):
                self.diffs.add(self.elems[i] - self.elems[j])

def _greedy_build(pool, max_size, rng, other_diffs=None):
    """Greedily build a Sidon set from pool, avoiding conflicts with other_diffs."""
    state = _SidonState()
    pool = list(pool)
    rng.shuffle(pool)
    
    for x in pool:
        if len(state.elems) >= max_size:
            break
        if state.can_add(x, other_diffs):
            state.add(x, other_diffs)
    
    return list(state.elems)

def _greedy_build_interleaved(max_a, max_b, rng):
    """Build A and B interleaved, each avoiding the other's diffs."""
    state_a = _SidonState()
    state_b = _SidonState()
    pool = list(range(1, N + 1))
    rng.shuffle(pool)
    
    for x in pool:
        if len(state_a.elems) < max_a:
            if state_a.can_add(x, state_b.diffs):
                state_a.add(x, state_b.diffs)
        if len(state_b.elems) < max_b:
            if state_b.can_add(x, state_a.diffs):
                state_b.add(x, state_a.diffs)
    
    return list(state_a.elems), list(state_b.elems)

def _local_search(A, B, rng, time_limit):
    """Try to improve (A, B) via local search: add elements."""
    A = list(A)
    B = list(B)
    
    def try_add_to(A, B, side):
        """Try to add a random element to side 0 (A) or 1 (B)."""
        if side == 0:
            if len(A) >= MAX_EACH:
                return False
            other_diffs = _diffs(B)
            self_diffs = _diffs(A)
            self_sums = set()
            n = len(A)
            for i in range(n):
                for j in range(i, n):
                    self_sums.add(A[i] + A[j])
        else:
            if len(B) >= MAX_EACH:
                return False
            other_diffs = _diffs(A)
            self_diffs = _diffs(B)
            self_sums = set()
            n = len(B)
            for i in range(n):
                for j in range(i, n):
                    self_sums.add(B[i] + B[j])
        
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if side == 0:
                # Check if x can be added to A
                ok = True
                if x + x in self_sums:
                    ok = False
                else:
                    for a in A:
                        if x + a in self_sums:
                            ok = False
                            break
                if ok:
                    new_diffs = set()
                    for a in A:
                        d = abs(x - a)
                        new_diffs.add(d)
                    if not (new_diffs & (self_diffs | other_diffs)):
                        A.append(x)
                        A.sort()
                        return True
            else:
                # Check if x can be added to B
                ok = True
                if x + x in self_sums:
                    ok = False
                else:
                    for b in B:
                        if x + b in self_sums:
                            ok = False
                            break
                if ok:
                    new_diffs = set()
                    for b in B:
                        d = abs(x - b)
                        new_diffs.add(d)
                    if not (new_diffs & (self_diffs | other_diffs)):
                        B.append(x)
                        B.sort()
                        return True
        return False
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        if try_add_to(A, B, 0):
            improved = True
        if try_add_to(A, B, 1):
            improved = True
    
    return A, B

def _random_restart(rng, target_a, target_b):
    """One random restart: build A and B with greedy."""
    A, B = _greedy_build_interleaved(target_a, target_b, rng)
    return A, B

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_A, best_B = [], []
    best_score = 0
    
    rng = random.Random(42)
    
    # Baseline: seed program style (deterministic)
    sets = [[], []]
    for x in range(1, N + 1):
        for side in (0, 1):
            candidate = sets[side] + [x]
            candidate.sort()
            sums = set()
            n = len(candidate)
            ok_sidon = True
            for i in range(n):
                for j in range(i, n):
                    s = candidate[i] + candidate[j]
                    if s in sums:
                        ok_sidon = False
                        break
                    sums.add(s)
            if ok_sidon:
                diffs = set()
                for i in range(n):
                    for j in range(i):
                        diffs.add(candidate[i] - candidate[j])
                other_diffs = set()
                os = sets[1 - side]
                n2 = len(os)
                for i in range(n2):
                    for j in range(i):
                        other_diffs.add(os[i] - os[j])
                if not (diffs & other_diffs):
                    sets[side] = candidate
    
    if len(sets[0]) > 0 and len(sets[1]) > 0:
        if _valid_pair(sets[0], sets[1]):
            best_A, best_B = list(sets[0]), list(sets[1])
            best_score = _score(best_A, best_B)
    
    # Randomized search with multiple restarts
    # Try various target size pairs
    size_pairs = []
    # Symmetric
    for s in range(15, 45, 2):
        size_pairs.append((s, s))
    # Asymmetric
    for a in range(15, 40, 3):
        for b in range(15, 40, 3):
            if a + b > 60:
                continue
            size_pairs.append((a, b))
    
    rng.shuffle(size_pairs)
    
    for A_size, B_size in size_pairs:
        if time.time() > deadline - 3:
            break
        
        # Multiple restarts for each size pair
        for restart in range(5):
            if time.time() > deadline - 3:
                break
            
            A, B = _random_restart(rng, A_size, B_size)
            score = _score(A, B)
            
            if score > best_score and _valid_pair(A, B):
                best_A, best_B = list(A), list(B)
                best_score = score
            
            # Local search to improve
            if score >= best_score - 4 and time.time() < deadline - 5:
                A2, B2 = _local_search(A, B, rng, deadline - 3)
                score2 = _score(A2, B2)
                if score2 > best_score and _valid_pair(A2, B2):
                    best_A, best_B = list(A2), list(B2)
                    best_score = score2
    
    # Final attempt: try to improve best pair with local search
    if best_score < MAX_EACH * 2 and time.time() < deadline - 2:
        A2, B2 = _local_search(best_A, best_B, rng, deadline - 1)
        if _score(A2, B2) > best_score and _valid_pair(A2, B2):
            best_A, best_B = A2, B2
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [3]
    
    # Final verification
    if not _valid_pair(best_A, best_B):
        # Fallback: simple valid pair
        best_A = [1, 2, 4, 8, 16, 32, 64, 128]
        best_B = [3, 6, 12, 24, 48, 96, 192, 255]
        if not _valid_pair(best_A, best_B):
            best_A = [1]
            best_B = [3]
    
    return [best_A, best_B]
