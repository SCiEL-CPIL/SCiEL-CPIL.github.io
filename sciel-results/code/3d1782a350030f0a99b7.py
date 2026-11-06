"""Two Sidon sets with disjoint difference sets.
Combines efficient state tracking with greedy construction and local search
(1-opt additions and 1-for-2 swaps) to maximize combined size."""

import time
import random

N = 256
MAX_SIZE = 64

class SidonState:
    """Tracks pair sums and nonzero differences for a Sidon set."""
    def __init__(self):
        self.els = set()
        self.sums = set()
        self.diffs = set()
    
    def size(self):
        return len(self.els)
    
    def can_add(self, x, other_diffs=None):
        """Check if x can be added without violating Sidon or diff constraints."""
        # Sidon: x+x and x+a for all a in els must be new
        if (x + x) in self.sums:
            return False
        for a in self.els:
            if (x + a) in self.sums:
                return False
        # Diffs: x-a and a-x must be new within self, and not in other_diffs
        for a in self.els:
            d1 = x - a
            d2 = a - x
            if d1 in self.diffs or d2 in self.diffs:
                return False
            if other_diffs is not None:
                if d1 in other_diffs or d2 in other_diffs:
                    return False
        return True
    
    def add(self, x):
        """Add x to the set. Assumes can_add is true."""
        self.els.add(x)
        self.sums.add(x + x)
        for a in self.els:
            if a != x:
                self.sums.add(a + x)
                self.diffs.add(a - x)
                self.diffs.add(x - a)
    
    def remove(self, x):
        """Remove x from the set."""
        if x not in self.els:
            return
        # Collect sums and diffs to remove
        to_remove_sums = {x + x}
        to_remove_diffs = set()
        for a in self.els:
            if a != x:
                to_remove_sums.add(a + x)
                to_remove_diffs.add(a - x)
                to_remove_diffs.add(x - a)
        self.els.remove(x)
        self.sums -= to_remove_sums
        self.diffs -= to_remove_diffs
    
    def to_list(self):
        return sorted(self.els)

def _check_sidon(els):
    """Check all pair sums a+b (a<=b) are unique."""
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
    """Set of all nonzero differences (positive and negative)."""
    d = set()
    n = len(els)
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = els[j] - els[i]
                d.add(diff)
    return d

def _valid(A, B):
    """Check A,B are Sidon and their nonzero difference sets are disjoint."""
    if not _check_sidon(A):
        return False
    if not _check_sidon(B):
        return False
    if _nonzero_diffs(A) & _nonzero_diffs(B):
        return False
    return True

def _greedy_build(state_a, state_b, rng, deadline, strategy="alternate"):
    """Greedily build sets A and B using state tracking."""
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if state_a.size() >= MAX_SIZE and state_b.size() >= MAX_SIZE:
            break
        
        if strategy == "alternate":
            # Alternate between A and B
            if state_a.size() < MAX_SIZE:
                if state_a.can_add(x, state_b.diffs):
                    state_a.add(x)
                    continue
            if state_b.size() < MAX_SIZE:
                if state_b.can_add(x, state_a.diffs):
                    state_b.add(x)
                    continue
        elif strategy == "A_first":
            if state_a.size() < MAX_SIZE:
                if state_a.can_add(x):
                    state_a.add(x)
                    continue
            if state_b.size() < MAX_SIZE:
                if state_b.can_add(x, state_a.diffs):
                    state_b.add(x)
                    continue
        else:  # B_first
            if state_b.size() < MAX_SIZE:
                if state_b.can_add(x):
                    state_b.add(x)
                    continue
            if state_a.size() < MAX_SIZE:
                if state_a.can_add(x, state_b.diffs):
                    state_a.add(x)
                    continue
    
    return state_a, state_b

def _local_search(state_a, state_b, rng, deadline):
    """Local search with 1-opt additions and 1-for-2 swaps."""
    best_total = state_a.size() + state_b.size()
    best_a_els = set(state_a.els)
    best_b_els = set(state_b.els)
    all_vals = list(range(1, N + 1))
    
    while time.time() < deadline:
        progress = False
        
        # Phase 1: Try 1-opt additions
        if state_a.size() < MAX_SIZE:
            candidates = [x for x in all_vals if x not in state_a.els]
            rng.shuffle(candidates)
            for x in candidates:
                if state_a.can_add(x, state_b.diffs):
                    state_a.add(x)
                    progress = True
                    break
        
        if state_b.size() < MAX_SIZE:
            candidates = [x for x in all_vals if x not in state_b.els]
            rng.shuffle(candidates)
            for x in candidates:
                if state_b.can_add(x, state_a.diffs):
                    state_b.add(x)
                    progress = True
                    break
        
        if progress:
            current_total = state_a.size() + state_b.size()
            if current_total > best_total:
                best_total = current_total
                best_a_els = set(state_a.els)
                best_b_els = set(state_b.els)
            continue
        
        # Phase 2: Try 1-for-2 swaps
        # Remove one element from A, try to add two (to A and/or B)
        if state_a.size() > 1:
            a_candidates = list(state_a.els)
            rng.shuffle(a_candidates)
            for x_remove in a_candidates:
                if time.time() > deadline:
                    break
                old_a_els = set(state_a.els)
                old_a_sums = set(state_a.sums)
                old_a_diffs = set(state_a.diffs)
                old_b_els = set(state_b.els)
                old_b_sums = set(state_b.sums)
                old_b_diffs = set(state_b.diffs)
                
                state_a.remove(x_remove)
                
                added_count = 0
                # Try adding to A
                if state_a.size() < MAX_SIZE:
                    cands_a = [x for x in all_vals if x not in state_a.els and x != x_remove]
                    rng.shuffle(cands_a)
                    for x_add in cands_a:
                        if state_a.can_add(x_add, state_b.diffs):
                            state_a.add(x_add)
                            added_count += 1
                            break
                
                # Try adding to B
                if state_b.size() < MAX_SIZE:
                    cands_b = [x for x in all_vals if x not in state_b.els]
                    rng.shuffle(cands_b)
                    for x_add in cands_b:
                        if state_b.can_add(x_add, state_a.diffs):
                            state_b.add(x_add)
                            added_count += 1
                            break
                
                current_total = state_a.size() + state_b.size()
                if added_count >= 2 and current_total > best_total:
                    best_total = current_total
                    best_a_els = set(state_a.els)
                    best_b_els = set(state_b.els)
                    progress = True
                else:
                    # Revert
                    state_a.els = set(old_a_els)
                    state_a.sums = set(old_a_sums)
                    state_a.diffs = set(old_a_diffs)
                    state_b.els = set(old_b_els)
                    state_b.sums = set(old_b_sums)
                    state_b.diffs = set(old_b_diffs)
        
        # Remove one element from B, try to add two
        if state_b.size() > 1:
            b_candidates = list(state_b.els)
            rng.shuffle(b_candidates)
            for x_remove in b_candidates:
                if time.time() > deadline:
                    break
                old_a_els = set(state_a.els)
                old_a_sums = set(state_a.sums)
                old_a_diffs = set(state_a.diffs)
                old_b_els = set(state_b.els)
                old_b_sums = set(state_b.sums)
                old_b_diffs = set(state_b.diffs)
                
                state_b.remove(x_remove)
                
                added_count = 0
                # Try adding to B
                if state_b.size() < MAX_SIZE:
                    cands_b = [x for x in all_vals if x not in state_b.els and x != x_remove]
                    rng.shuffle(cands_b)
                    for x_add in cands_b:
                        if state_b.can_add(x_add, state_a.diffs):
                            state_b.add(x_add)
                            added_count += 1
                            break
                
                # Try adding to A
                if state_a.size() < MAX_SIZE:
                    cands_a = [x for x in all_vals if x not in state_a.els]
                    rng.shuffle(cands_a)
                    for x_add in cands_a:
                        if state_a.can_add(x_add, state_b.diffs):
                            state_a.add(x_add)
                            added_count += 1
                            break
                
                current_total = state_a.size() + state_b.size()
                if added_count >= 2 and current_total > best_total:
                    best_total = current_total
                    best_a_els = set(state_a.els)
                    best_b_els = set(state_b.els)
                    progress = True
                else:
                    # Revert
                    state_a.els = set(old_a_els)
                    state_a.sums = set(old_a_sums)
                    state_a.diffs = set(old_a_diffs)
                    state_b.els = set(old_b_els)
                    state_b.sums = set(old_b_sums)
                    state_b.diffs = set(old_b_diffs)
        
        if not progress:
            break
    
    # Restore best found
    state_a.els = set(best_a_els)
    state_b.els = set(best_b_els)
    # Rebuild sums and diffs
    state_a.sums = set()
    state_a.diffs = set()
    for a in state_a.els:
        state_a.sums.add(a + a)
        for b in state_a.els:
            if b != a:
                state_a.sums.add(a + b)
                state_a.diffs.add(a - b)
                state_a.diffs.add(b - a)
    state_b.sums = set()
    state_b.diffs = set()
    for a in state_b.els:
        state_b.sums.add(a + a)
        for b in state_b.els:
            if b != a:
                state_b.sums.add(a + b)
                state_b.diffs.add(a - b)
                state_b.diffs.add(b - a)
    
    return state_a, state_b

def construct():
    """Construct two Sidon sets with disjoint difference sets, maximizing combined size."""
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    best_A = [1]
    best_B = [2]
    best_score = 2
    
    rng = random.Random(42)
    
    def _attempt_build(seed_A=None, seed_B=None, strategy="alternate"):
        """Build A and B from optional seeds using a strategy."""
        state_a = SidonState()
        state_b = SidonState()
        
        if seed_A:
            for x in seed_A:
                state_a.els.add(x)
        if seed_B:
            for x in seed_B:
                state_b.els.add(x)
        
        # Ensure both nonempty
        if not state_a.els:
            state_a.els.add(rng.randint(1, N))
        if not state_b.els:
            state_b.els.add(rng.randint(1, N))
        
        # Build sums and diffs
        for x in state_a.els:
            state_a.sums.add(x + x)
            for y in state_a.els:
                if y != x:
                    state_a.sums.add(x + y)
                    state_a.diffs.add(x - y)
                    state_a.diffs.add(y - x)
        for x in state_b.els:
            state_b.sums.add(x + x)
            for y in state_b.els:
                if y != x:
                    state_b.sums.add(x + y)
                    state_b.diffs.add(x - y)
                    state_b.diffs.add(y - x)
        
        # Check initial validity
        A_list = state_a.to_list()
        B_list = state_b.to_list()
        if not _valid(A_list, B_list):
            return None, None, 0
        
        # Greedy build
        _greedy_build(state_a, state_b, rng, deadline - 10, strategy)
        
        # Local search
        _local_search(state_a, state_b, rng, deadline - 3)
        
        A_list = state_a.to_list()
        B_list = state_b.to_list()
        
        if _valid(A_list, B_list):
            return A_list, B_list, len(A_list) + len(B_list)
        return None, None, 0
    
    # Main search loop with different strategies
    strategies = ["alternate", "A_first", "B_first"]
    it = 0
    while time.time() < deadline - 5:
        strat = strategies[it % 3]
        
        if it % 5 == 0:
            # Random single seeds
            A0 = [rng.randint(1, N)]
            B0 = [rng.randint(1, N)]
            A, B, score = _attempt_build(A0, B0, strat)
        elif it % 5 == 1:
            # A first
            A0 = [rng.randint(1, N)]
            A, B, score = _attempt_build(A0, None, "A_first")
        elif it % 5 == 2:
            # B first
            B0 = [rng.randint(1, N)]
            A, B, score = _attempt_build(None, B0, "B_first")
        elif it % 5 == 3:
            # Small random starts
            sa = rng.randint(1, 3)
            sb = rng.randint(1, 3)
            A0 = sorted(rng.sample(range(1, N + 1), sa))
            B0 = sorted(rng.sample(range(1, N + 1), sb))
            A, B, score = _attempt_build(A0, B0, strat)
        else:
            # Wider spacing seeds
            A0 = [rng.randint(1, 128)]
            B0 = [rng.randint(129, N)]
            A, B, score = _attempt_build(A0, B0, strat)
        
        if A is not None and score > best_score:
            best_score = score
            best_A = list(A)
            best_B = list(B)
        
        it += 1
    
    # Final validation
    if not _valid(best_A, best_B):
        best_A = [1]
        best_B = [2]
        best_score = 2
    
    return [best_A, best_B]
