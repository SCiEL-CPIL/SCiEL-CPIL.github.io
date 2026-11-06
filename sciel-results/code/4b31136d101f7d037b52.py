import time
import random
import sys

def _check_sidon_unique_sums(els):
    """Check that all pair sums a+b (a<=b) are unique."""
    sums = set()
    n = len(els)
    for i in range(n):
        for j in range(i, n):
            s = els[i] + els[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_nonzero_diffs(els):
    """Get set of nonzero differences (positive and negative)."""
    diffs = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = els[j] - els[i]
                if d != 0:
                    diffs.add(d)
    return diffs

def _is_valid_pair(A, B):
    """Check that A and B are Sidon and their nonzero difference sets are disjoint."""
    if not A or not B:
        return False
    if not _check_sidon_unique_sums(A):
        return False
    if not _check_sidon_unique_sums(B):
        return False
    diffsA = _get_nonzero_diffs(A)
    diffsB = _get_nonzero_diffs(B)
    if diffsA & diffsB:
        return False
    return True

class SidonState:
    """Incremental state for a Sidon set with O(k) add checks."""
    def __init__(self, n=256, max_size=64):
        self.n = n
        self.max_size = max_size
        self.els = []
        self.els_set = set()
        self.used_sums = set()
        self.used_diffs = set()
    
    def reset(self):
        self.els = []
        self.els_set = set()
        self.used_sums = set()
        self.used_diffs = set()
    
    def can_add(self, x, other_diffs=None):
        """Check if x can be added maintaining Sidon and diff constraints."""
        if x in self.els_set:
            return False
        if len(self.els) >= self.max_size:
            return False
        # Check Sidon sums
        if 2 * x in self.used_sums:
            return False
        for a in self.els:
            if (a + x) in self.used_sums:
                return False
        # Check differences against self
        for a in self.els:
            d1 = x - a
            d2 = a - x
            if d1 in self.used_diffs:
                return False
            if d2 in self.used_diffs:
                return False
        # Check differences against other set
        if other_diffs:
            for a in self.els:
                d1 = x - a
                d2 = a - x
                if d1 in other_diffs:
                    return False
                if d2 in other_diffs:
                    return False
        return True
    
    def add(self, x, other_diffs=None):
        """Add x to the set, updating all state."""
        self.els.append(x)
        self.els_set.add(x)
        self.used_sums.add(2 * x)
        for a in self.els[:-1]:
            self.used_sums.add(a + x)
            d1 = x - a
            d2 = a - x
            self.used_diffs.add(d1)
            self.used_diffs.add(d2)
    
    def remove(self, x):
        """Remove x from the set, updating all state."""
        self.els.remove(x)
        self.els_set.discard(x)
        # Rebuild state from scratch (simple but correct)
        self.used_sums = set()
        self.used_diffs = set()
        for i, a in enumerate(self.els):
            self.used_sums.add(2 * a)
            for j in range(i + 1, len(self.els)):
                b = self.els[j]
                self.used_sums.add(a + b)
                d1 = b - a
                d2 = a - b
                self.used_diffs.add(d1)
                self.used_diffs.add(d2)

def _greedy_build_A(B, n=256, max_size=64, rng=None):
    """Greedily build A given B, ensuring Sidon property and disjoint diffs with B."""
    if rng is None:
        rng = random.Random()
    B_diffs = _get_nonzero_diffs(B) if B else set()
    state = SidonState(n, max_size)
    candidates = list(range(1, n + 1))
    rng.shuffle(candidates)
    for x in candidates:
        if state.can_add(x, B_diffs):
            state.add(x, B_diffs)
    return state.els

def _greedy_build_B(A, n=256, max_size=64, rng=None):
    """Greedily build B given A, ensuring Sidon property and disjoint diffs with A."""
    if rng is None:
        rng = random.Random()
    A_diffs = _get_nonzero_diffs(A) if A else set()
    state = SidonState(n, max_size)
    candidates = list(range(1, n + 1))
    rng.shuffle(candidates)
    for x in candidates:
        if state.can_add(x, A_diffs):
            state.add(x, A_diffs)
    return state.els

def _build_pair_interleaved(n=256, max_size=64, rng=None, order='random'):
    """Build A and B interleaved for better diff coordination."""
    if rng is None:
        rng = random.Random()
    state_A = SidonState(n, max_size)
    state_B = SidonState(n, max_size)
    
    candidates = list(range(1, n + 1))
    if order == 'random':
        rng.shuffle(candidates)
    elif order == 'asc':
        candidates.sort()
    elif order == 'desc':
        candidates.sort(reverse=True)
    
    for x in candidates:
        if len(state_A.els) < max_size and state_A.can_add(x, state_B.used_diffs):
            state_A.add(x, state_B.used_diffs)
        elif len(state_B.els) < max_size and state_B.can_add(x, state_A.used_diffs):
            state_B.add(x, state_A.used_diffs)
        # Try both
        if len(state_A.els) < max_size and state_A.can_add(x, state_B.used_diffs):
            state_A.add(x, state_B.used_diffs)
        if len(state_B.els) < max_size and state_B.can_add(x, state_A.used_diffs):
            state_B.add(x, state_A.used_diffs)
    
    return state_A.els, state_B.els

def _local_search_improve(A, B, n=256, max_iters=50, rng=None):
    """
    Improve by adding/swapping elements using incremental state.
    """
    if rng is None:
        rng = random.Random()
    
    best_A = list(A)
    best_B = list(B)
    best_size = len(A) + len(B)
    
    # Build states for current best
    state_A = SidonState(n, 64)
    for x in best_A:
        state_A.add(x)
    state_B = SidonState(n, 64)
    for x in best_B:
        state_B.add(x)
    
    for _ in range(max_iters):
        improved = False
        
        # Try to add to A
        for _ in range(15):
            x = rng.randint(1, n)
            if x in state_A.els_set:
                continue
            if state_A.can_add(x, state_B.used_diffs):
                state_A.add(x, state_B.used_diffs)
                best_A = list(state_A.els)
                best_size = len(best_A) + len(best_B)
                improved = True
                break
        
        # Try to add to B
        for _ in range(15):
            x = rng.randint(1, n)
            if x in state_B.els_set:
                continue
            if state_B.can_add(x, state_A.used_diffs):
                state_B.add(x, state_A.used_diffs)
                best_B = list(state_B.els)
                best_size = len(best_A) + len(best_B)
                improved = True
                break
        
        # Try swap in A
        if len(best_A) > 0:
            idx = rng.randint(0, len(best_A) - 1)
            old_x = best_A[idx]
            for _ in range(15):
                new_x = rng.randint(1, n)
                if new_x in state_A.els_set:
                    continue
                # Temporarily remove old_x
                state_A.remove(old_x)
                if state_A.can_add(new_x, state_B.used_diffs):
                    state_A.add(new_x, state_B.used_diffs)
                    best_A = list(state_A.els)
                    improved = True
                    break
                else:
                    # Restore
                    state_A.add(old_x)
        
        # Try swap in B
        if len(best_B) > 0:
            idx = rng.randint(0, len(best_B) - 1)
            old_x = best_B[idx]
            for _ in range(15):
                new_x = rng.randint(1, n)
                if new_x in state_B.els_set:
                    continue
                state_B.remove(old_x)
                if state_B.can_add(new_x, state_A.used_diffs):
                    state_B.add(new_x, state_A.used_diffs)
                    best_B = list(state_B.els)
                    improved = True
                    break
                else:
                    state_B.add(old_x)
        
        if not improved:
            break
    
    return best_A, best_B

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 256
    max_size = 64
    
    best_A = [1]
    best_B = [2]
    best_score = 2
    
    rng = random.Random(42)
    
    # Strategy list
    strategies = [
        ('seq_AB', 'random'),
        ('seq_BA', 'random'),
        ('interleaved', 'random'),
        ('seq_AB', 'asc'),
        ('seq_BA', 'desc'),
        ('interleaved', 'asc'),
    ]
    
    for restart in range(100000):
        if time.time() > deadline:
            break
        
        strategy, order = strategies[restart % len(strategies)]
        
        if strategy == 'seq_AB':
            # Build A first (random seed), then B, then extend A
            A = _greedy_build_A([], n, max_size, rng)
            B = _greedy_build_B(A, n, max_size, rng)
            A2 = _greedy_build_A(B, n, max_size, rng)
            if len(A2) > len(A):
                A = A2
        elif strategy == 'seq_BA':
            # Build B first, then A, then extend B
            B = _greedy_build_B([], n, max_size, rng)
            A = _greedy_build_A(B, n, max_size, rng)
            B2 = _greedy_build_B(A, n, max_size, rng)
            if len(B2) > len(B):
                B = B2
        else:  # interleaved
            A, B = _build_pair_interleaved(n, max_size, rng, order)
        
        if not A:
            A = [1]
        if not B:
            B = [2]
        
        if _is_valid_pair(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = list(A)
                best_B = list(B)
        
        # Local search with incremental state
        if time.time() < deadline - 3:
            A_ls, B_ls = _local_search_improve(A, B, n, max_iters=30, rng=rng)
            if _is_valid_pair(A_ls, B_ls):
                score_ls = len(A_ls) + len(B_ls)
                if score_ls > best_score:
                    best_score = score_ls
                    best_A = list(A_ls)
                    best_B = list(B_ls)
    
    if not _is_valid_pair(best_A, best_B):
        best_A = [1]
        best_B = [2]
    
    return [best_A, best_B]
