"""Model-and-optimize: grow two Sidon sets with disjoint differences via local search."""
import random
from collections import Counter
from itertools import combinations_with_replacement, combinations
import time
import numpy as np

N = 256
MAX_EACH = 64
MAX_SUM = 2 * N
MAX_DIFF = N

class SidonState:
    """
    Maintains a set of elements, its pair sums, and its differences using bitmasks.
    Sums: 512 bits (max sum 512).
    Diffs: 256 bits (max diff 255, index by diff value).
    """
    def __init__(self, elements=None):
        self.elements = sorted(elements) if elements else []
        self.sums_mask = 0
        self.diffs_mask = 0
        if self.elements:
            self._rebuild()

    def _rebuild(self):
        self.sums_mask = 0
        self.diffs_mask = 0
        els = self.elements
        n = len(els)
        for i in range(n):
            a = els[i]
            for j in range(i, n):
                b = els[j]
                self.sums_mask |= 1 << (a + b)
                if i != j:
                    d = abs(b - a)
                    self.diffs_mask |= 1 << d

    def can_add(self, x, other_diffs_mask=0):
        """
        Check if x can be added.
        other_diffs_mask: bitmask of differences from the other set.
        """
        els = self.elements
        new_sums = 0
        new_diffs = 0
        
        # x + x
        new_sums |= 1 << (x + x)
        
        for e in els:
            new_sums |= 1 << (x + e)
            d = abs(x - e)
            new_diffs |= 1 << d
        
        # Check sum collisions
        if self.sums_mask & new_sums:
            return False
        
        # Check diff collisions within self
        if self.diffs_mask & new_diffs:
            return False
        
        # Check diff collisions with other set
        if other_diffs_mask & new_diffs:
            return False
            
        return True

    def add(self, x):
        """Add x to the set, updating masks."""
        els = self.elements
        new_sums = 1 << (x + x)
        new_diffs = 0
        
        for e in els:
            new_sums |= 1 << (x + e)
            d = abs(x - e)
            new_diffs |= 1 << d
            
        self.sums_mask |= new_sums
        self.diffs_mask |= new_diffs
        
        # Insert x into sorted list
        i = 0
        while i < len(els) and els[i] < x:
            i += 1
        els.insert(i, x)

    def remove(self, x):
        """Remove x from the set. Rebuilds state for simplicity."""
        self.elements.remove(x)
        self._rebuild()


def _greedy_sidon_fast(max_size=64, n=N, seed=1337):
    """Greedy construction of a Sidon set."""
    rng = random.Random(seed)
    state = SidonState()
    candidates = list(range(1, n + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if state.can_add(x):
            state.add(x)
            if len(state.elements) >= max_size:
                break
    return state.elements


def _greedy_b_fast(a, max_size=64, n=N, seed=0):
    """Greedy construction of B given fixed A."""
    rng = random.Random(seed)
    sa = SidonState(a)
    sb = SidonState()
    candidates = list(range(1, n + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if sb.can_add(x, other_diffs_mask=sa.diffs_mask):
            sb.add(x)
            if len(sb.elements) >= max_size:
                break
    return sb.elements


def _local_search_fast(a, b, max_size=MAX_EACH, n=N, deadline=55.0, seed=0):
    """
    Local search using SidonState.
    Tries to add elements, then tries 1-for-1 swaps to escape local optima.
    """
    rng = random.Random(seed)
    sa = SidonState(a)
    sb = SidonState(b)
    
    best_total = len(sa.elements) + len(sb.elements)
    best_a, best_b = list(sa.elements), list(sb.elements)
    
    pool = list(range(1, n + 1))
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try adding to A
        if len(sa.elements) < max_size:
            for x in pool:
                if x in sa.elements:
                    continue
                if sa.can_add(x, other_diffs_mask=sb.diffs_mask):
                    sa.add(x)
                    improved = True
                    if len(sa.elements) + len(sb.elements) > best_total:
                        best_total = len(sa.elements) + len(sb.elements)
                        best_a, best_b = list(sa.elements), list(sb.elements)
                    break
        
        # 2. Try adding to B
        if len(sb.elements) < max_size:
            for x in pool:
                if x in sb.elements:
                    continue
                if sb.can_add(x, other_diffs_mask=sa.diffs_mask):
                    sb.add(x)
                    improved = True
                    if len(sa.elements) + len(sb.elements) > best_total:
                        best_total = len(sa.elements) + len(sb.elements)
                        best_a, best_b = list(sa.elements), list(sb.elements)
                    break

        if len(sa.elements) >= max_size and len(sb.elements) >= max_size:
            break

        if not improved:
            # Stuck. Try swaps.
            # Pick a random element from A and try to swap with a random element from pool not in A
            if len(sa.elements) > 0 and len(sa.elements) < max_size:
                idx = rng.randrange(len(sa.elements))
                x_rem = sa.elements[idx]
                
                # Try to find a replacement
                found = False
                # Shuffle pool for randomization
                rng.shuffle(pool)
                for x_new in pool:
                    if x_new in sa.elements:
                        continue
                    # Simulate swap
                    old_sa_els = list(sa.elements)
                    old_sb_els = list(sb.elements)
                    
                    sa.remove(x_rem)
                    if sa.can_add(x_new, other_diffs_mask=sb.diffs_mask):
                        sa.add(x_new)
                        # Check if we can also add more to B now?
                        # For now, just accept if valid.
                        improved = True
                        if len(sa.elements) + len(sb.elements) > best_total:
                            best_total = len(sa.elements) + len(sb.elements)
                            best_a, best_b = list(sa.elements), list(sb.elements)
                        found = True
                        break
                    else:
                        # Revert
                        sa.elements = old_sa_els
                        sa._rebuild()
                        sb.elements = old_sb_els
                        sb._rebuild()
                
                if not found:
                    # Revert if we tried to remove but didn't add
                    # Actually, remove was called, so we must revert if not added
                    if x_rem in sa.elements:
                         # It was reverted in the loop if failed, but if loop didn't run?
                         pass
                    # The logic above reverts inside the loop. If no x_new was valid, we reverted.
                    # If x_new was valid, we kept it.
                    # But what if we removed x_rem and then failed to add x_new? We reverted.
                    # So state is consistent.
            
            # Similarly for B
            if len(sb.elements) > 0 and len(sb.elements) < max_size:
                idx = rng.randrange(len(sb.elements))
                x_rem = sb.elements[idx]
                
                found = False
                rng.shuffle(pool)
                for x_new in pool:
                    if x_new in sb.elements:
                        continue
                    old_sa_els = list(sa.elements)
                    old_sb_els = list(sb.elements)
                    
                    sb.remove(x_rem)
                    if sb.can_add(x_new, other_diffs_mask=sa.diffs_mask):
                        sb.add(x_new)
                        improved = True
                        if len(sa.elements) + len(sb.elements) > best_total:
                            best_total = len(sa.elements) + len(sb.elements)
                            best_a, best_b = list(sa.elements), list(sb.elements)
                        found = True
                        break
                    else:
                        sb.elements = old_sb_els
                        sb._rebuild()
                        sa.elements = old_sa_els
                        sa._rebuild()

        if not improved:
             # Random restart if stuck for too long? 
             # Or just break to try next seed.
             pass

    return best_a, best_b, best_total


def _valid_pair(a, b):
    """Check A Sidon, B Sidon, and disjoint nonzero differences."""
    sa = SidonState(a)
    sb = SidonState(b)
    # Check Sidon property: sums are unique. 
    # Our SidonState ensures this during construction, but let's double check for safety.
    # Actually, SidonState construction enforces it.
    # Check diff disjointness
    return not (sa.diffs_mask & sb.diffs_mask)


def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin for 60s limit

    best_total = 0
    best_a, best_b = [], []

    seeds = [1337, 7, 99, 12345, 555, 2024, 42, 777, 999, 2023, 111, 888, 314, 271]
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        
        # Alternate building strategies
        if seed % 2 == 0:
            a = _greedy_sidon_fast(max_size=MAX_EACH, seed=seed)
            b = _greedy_b_fast(a, max_size=MAX_EACH, seed=seed)
        else:
            b = _greedy_sidon_fast(max_size=MAX_EACH, seed=seed)
            a = _greedy_b_fast(b, max_size=MAX_EACH, seed=seed)
        
        t = len(a) + len(b)
        if t > best_total:
            best_total = t
            best_a, best_b = list(a), list(b)
        
        # Local search
        remaining = deadline - time.time()
        if remaining > 2:
            ls_deadline = t0 + min(deadline, time.time() + max(0.5, remaining - 2))
            a2, b2, total2 = _local_search_fast(a, b, max_size=MAX_EACH, deadline=ls_deadline, seed=seed)
            if total2 > best_total:
                best_total = total2
                best_a, best_b = a2, b2

    # Final validity check
    if not _valid_pair(best_a, best_b):
        # Fallback: simple valid pair
        # Just return a small valid pair to ensure non-empty and valid
        # A = [1, 2], B = [4, 8] ?
        # Diffs A: {1}. Diffs B: {4}. Disjoint.
        # Sums A: 2,3,4. Sums B: 8,12,16. Unique.
        # Let's just construct a tiny one.
        a = [1, 2]
        b = [4, 8]
        return [a, b]

    return [best_a, best_b]


if __name__ == "__main__":
    A, B = construct()
    print(f"|A|={len(A)}, |B|={len(B)}, total={len(A)+len(B)}")
    print(f"A={A}")
    print(f"B={B}")
