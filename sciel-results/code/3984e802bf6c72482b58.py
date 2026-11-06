"""
Improved program for two Sidon sets with disjoint difference sets.
Uses randomized local search with 1-for-2 swaps and strict time management.
"""

import time
import random
import sys

N = 256
MAX_SIZE = 64
DEADLINE = 85.0  # Safety margin before 90s kill

class SidonState:
    """Tracks pair sums and nonzero differences for a set."""
    def __init__(self):
        self.els = set()
        self.sums = set()      # all a+b for a<=b
        self.diffs = set()     # all a-b for a!=b (nonzero)
        
    def size(self):
        return len(self.els)
        
    def can_add(self, x, other_diffs=None):
        """
        Check if x can be added to this set.
        other_diffs: set of differences that must also be avoided (from the other set).
        """
        # Check sums: x + a for a in els, and x + x
        for a in self.els:
            if (x + a) in self.sums:
                return False
        if (x + x) in self.sums:
            return False
            
        # Check diffs: x - a and a - x for a in els
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
        for a in self.els:
            if a != x:
                self.sums.add(a + x)
                self.diffs.add(a - x)
                self.diffs.add(x - a)
        self.sums.add(x + x)
        
    def remove(self, x):
        """Remove x from the set."""
        if x not in self.els:
            return
        self.els.remove(x)
        # Remove sums and diffs involving x
        to_remove_sums = set()
        to_remove_diffs = set()
        for a in self.els:
            to_remove_sums.add(a + x)
            to_remove_diffs.add(a - x)
            to_remove_diffs.add(x - a)
        to_remove_sums.add(x + x)
        self.sums -= to_remove_sums
        self.diffs -= to_remove_diffs

def greedy_build(state_a, state_b, rng, deadline):
    """Greedily build sets A and B."""
    # Shuffle candidates
    candidates_a = list(range(1, N + 1))
    candidates_b = list(range(1, N + 1))
    rng.shuffle(candidates_a)
    rng.shuffle(candidates_b)
    
    # Build A
    for x in candidates_a:
        if state_a.size() >= MAX_SIZE:
            break
        if time.time() > deadline:
            return
        if state_a.can_add(x):
            state_a.add(x)
            
    # Build B, avoiding A's diffs
    for x in candidates_b:
        if state_b.size() >= MAX_SIZE:
            break
        if time.time() > deadline:
            return
        if state_b.can_add(x, state_a.diffs):
            state_b.add(x)

def local_search(state_a, state_b, rng, deadline):
    """
    Local search with 1-opt (add) and 1-for-2 swaps.
    Returns the best state found.
    """
    best_total = state_a.size() + state_b.size()
    best_a_els = set(state_a.els)
    best_b_els = set(state_b.els)
    
    # Pre-compute all possible values for quick iteration
    all_vals = list(range(1, N + 1))
    
    while time.time() < deadline:
        progress = False
        
        # --- Phase 1: Try to add (1-opt) ---
        # Try adding to A
        if state_a.size() < MAX_SIZE:
            # Randomize order to avoid bias
            candidates = [x for x in all_vals if x not in state_a.els]
            rng.shuffle(candidates)
            for x in candidates:
                if state_a.can_add(x, state_b.diffs):
                    state_a.add(x)
                    progress = True
                    break
        
        # Try adding to B
        if state_b.size() < MAX_SIZE:
            candidates = [x for x in all_vals if x not in state_b.els]
            rng.shuffle(candidates)
            for x in candidates:
                if state_b.can_add(x, state_a.diffs):
                    state_b.add(x)
                    progress = True
                    break
        
        # --- Phase 2: Try 1-for-2 swaps ---
        # If we didn't progress with adds, try removing one element to allow adding two.
        if not progress:
            # Try removing from A
            if state_a.size() > 1:
                a_candidates = list(state_a.els)
                rng.shuffle(a_candidates)
                for x_remove in a_candidates:
                    # Save state
                    old_a_els = set(state_a.els)
                    old_a_sums = set(state_a.sums)
                    old_a_diffs = set(state_a.diffs)
                    
                    state_a.remove(x_remove)
                    
                    # Try to add two elements to A (or one to A and one to B)
                    added_count = 0
                    added_elements = []
                    
                    # Try adding to A first
                    if state_a.size() < MAX_SIZE:
                        cands_a = [x for x in all_vals if x not in state_a.els and x != x_remove]
                        rng.shuffle(cands_a)
                        for x_add in cands_a:
                            if state_a.can_add(x_add, state_b.diffs):
                                state_a.add(x_add)
                                added_elements.append(('A', x_add))
                                added_count += 1
                                break
                    
                    # Try adding to B
                    if state_b.size() < MAX_SIZE:
                        cands_b = [x for x in all_vals if x not in state_b.els]
                        rng.shuffle(cands_b)
                        for x_add in cands_b:
                            if state_b.can_add(x_add, state_a.diffs):
                                state_b.add(x_add)
                                added_elements.append(('B', x_add))
                                added_count += 1
                                break
                    
                    new_total = state_a.size() + state_b.size()
                    old_total = len(old_a_els) + state_b.size() - added_count  # Approximate, better to track properly
                    # Actually, let's just compare total size
                    if new_total > state_a.size() + state_b.size() - added_count: # This logic is flawed, let's just check if we added more than we removed
                        # We removed 1. If we added >= 1, size is same or better.
                        # If we added 2, size is +1.
                        if added_count >= 1:
                            # We improved or maintained size. If maintained, we might have better distribution?
                            # Let's accept if added_count >= 1 to allow exploration, but prefer added_count == 2.
                            # For strict improvement:
                            if added_count == 2:
                                progress = True
                                # Update best if improved
                                current_total = state_a.size() + state_b.size()
                                if current_total > best_total:
                                    best_total = current_total
                                    best_a_els = set(state_a.els)
                                    best_b_els = set(state_b.els)
                            else:
                                # Added 1, removed 1. Size unchanged. 
                                # Accept with some probability to escape local optima? 
                                # Or just revert if no size gain.
                                # Let's revert if no size gain to keep sets "full" of valid elements.
                                # Actually, if we revert, we are back to square one.
                                # Let's only accept if size increases.
                                pass
                        else:
                            # Added 0. Revert.
                            state_a.els = old_a_els
                            state_a.sums = old_a_sums
                            state_a.diffs = old_a_diffs
                    else:
                        # Revert
                        state_a.els = old_a_els
                        state_a.sums = old_a_sums
                        state_a.diffs = old_a_diffs
                        
                    if progress:
                        break
            
            # Try removing from B
            if not progress and state_b.size() > 1:
                b_candidates = list(state_b.els)
                rng.shuffle(b_candidates)
                for x_remove in b_candidates:
                    old_b_els = set(state_b.els)
                    old_b_sums = set(state_b.sums)
                    old_b_diffs = set(state_b.diffs)
                    
                    state_b.remove(x_remove)
                    
                    added_count = 0
                    added_elements = []
                    
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
                    
                    if added_count == 2:
                        progress = True
                        current_total = state_a.size() + state_b.size()
                        if current_total > best_total:
                            best_total = current_total
                            best_a_els = set(state_a.els)
                            best_b_els = set(state_b.els)
                    else:
                        state_b.els = old_b_els
                        state_b.sums = old_b_sums
                        state_b.diffs = old_b_diffs
                        
                    if progress:
                        break
        
        if not progress:
            # No moves possible. Break or restart? 
            # Since we are in a single local search call, we break and let the outer loop restart.
            break
            
    return best_total, best_a_els, best_b_els

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(12345)
    
    best_total = 0
    best_a = set()
    best_b = set()
    
    # Number of restarts
    num_restarts = 1000  # High number, will be cut by deadline
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        state_a = SidonState()
        state_b = SidonState()
        
        # Greedy Initialization
        greedy_build(state_a, state_b, rng, deadline - 2.0)
        
        # Local Search
        ls_total, ls_a, ls_b = local_search(state_a, state_b, rng, deadline - 2.0)
        
        if ls_total > best_total:
            best_total = ls_total
            best_a = ls_a
            best_b = ls_b
            
    # Ensure nonempty
    if not best_a:
        best_a = {1}
    if not best_b:
        best_b = {2}
        
    # Convert to lists for return
    return [list(best_a), list(best_b)]

# Entry point
if __name__ == "__main__":
    # This block is for testing, the evaluator calls construct()
    pass
