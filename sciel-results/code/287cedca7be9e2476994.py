"""Improved program for small inclusion-maximal Sidon sets in [1,256]."""
import numpy as np
import time
import random

N = 256
DEADLINE = 55.0  # safety margin before 60s kill

# EVOLVE-BLOCK-START

def _build_conflict_matrix():
    """
    Build a boolean matrix conflict[i,j] (1-indexed conceptually, 0-indexed here for [1..N])
    where conflict[i-1][j-1] is True if elements i and j CANNOT coexist in a Sidon set.
    Two elements a, b cannot coexist if:
      - a == b and 2a collides with some existing sum (handled dynamically, but for pairwise conflict:
        actually for a Sidon set, the condition is that all pairwise sums a+b with a<=b are distinct.
        So a and b conflict if adding b to a set containing a would cause a sum collision.
        But for a static graph: vertices are elements of [1,N]. Edge between a and b (a<b) means
        they cannot both be in a Sidon set. When do two elements conflict?
        a and b conflict if 2a == 2b (impossible for a!=b), or 2a == a+b (i.e., a=b, no),
        or a+b == a+a (i.e., b=a, no), or a+b == b+b (i.e., a=b, no).
        Actually, for two distinct elements a < b, they conflict if there exist c,d in the set such that
        the sums collide. But for a pairwise conflict graph where we want the remaining vertices to form
        a Sidon set, we need: a and b are incompatible if {a,b} itself is not Sidon.
        {a,b} is Sidon iff all sums from {a+a, a+b, b+b} are distinct, i.e., 2a != a+b != 2b and 2a != 2b.
        2a != 2b always for a!=b. 2a != a+b iff a!=b. a+b != 2b iff a!=b.
        So any two distinct elements form a Sidon set of size 2! The conflict is not pairwise.
        
        Wait, the Sidon condition for a set S is that all pairwise sums s1+s2 with s1<=s2 are distinct.
        For |S|=2, S={a,b}, a<b: sums are 2a, a+b, 2b. These are distinct iff a!=b. So yes, any two
        distinct elements form a valid Sidon set. The conflict arises only when we have 3+ elements.
        
        So a pairwise conflict graph doesn't directly work. We need a different approach.
        
        Let me reconsider: the task says to model as maximum independent set where edges connect elements
        that cannot coexist. But as shown, any two elements CAN coexist. The issue is with triples or more.
        
        Actually, let me re-read: "edges connect elements that cannot coexist in a Sidon set". 
        If we're building a Sidon set, the constraint is global. But for a greedy approach, when we consider
        adding element x to a current set S, x conflicts with S if adding x creates a sum collision.
        
        Let me use a different strategy: greedy construction with local search (swap/remove) to minimize
        cardinality while maintaining maximality.
    """
    pass

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set."""
    if not selected:
        return True
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            b = selected[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected):
    """Check if selected is inclusion-maximal: adding any missing element destroys Sidon property."""
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Try adding x
        test_set = list(selected) + [x]
        if _is_valid_sidon(test_set):
            return False
    return True

def _greedy_construct(order):
    """Greedy construction following a given order of elements."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _try_remove(selected, sums):
    """Try to remove elements to reduce size while maintaining maximality."""
    selected = list(selected)
    improved = True
    while improved:
        improved = False
        for i in range(len(selected)):
            candidate = selected[i]
            # Temporarily remove
            new_selected = selected[:i] + selected[i+1:]
            if not _is_valid_sidon(new_selected):
                continue
            if not _is_maximal(new_selected):
                continue
            # Success: remove this element
            selected = new_selected
            improved = True
            break
    return selected

def _try_swap(selected):
    """Try to swap an element for a missing one to reduce size or enable removals."""
    # This is complex; let's focus on removal first
    return selected

def construct():
    start_time = time.time()
    
    best_result = None
    best_size = N + 1
    
    # First, get a baseline greedy result
    baseline = _greedy_construct(range(1, N + 1))
    baseline = _try_remove(baseline, set())
    best_result = baseline
    best_size = len(baseline)
    
    # Try multiple random orderings
    rng = random.Random(42)
    remaining_time = DEADLINE - (time.time() - start_time)
    
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Random ordering
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        result = _greedy_construct(order)
        result = _try_remove(result, set())
        
        if len(result) < best_size:
            best_size = len(result)
            best_result = result
            
            # Try more aggressive local search
            if time.time() - start_time < DEADLINE - 10:
                # Try multiple removal passes
                for _ in range(3):
                    new_result = _try_remove(result, set())
                    if len(new_result) < len(result):
                        result = new_result
                        best_result = result
                        best_size = len(result)
    
    return best_result

# EVOLVE-BLOCK-END
